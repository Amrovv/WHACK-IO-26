"""Single entry point: wires perception, brain, voice, the dashboard server and the bus log.

No business logic here. Each layer comes from its owner's package when it exists and falls back
to coach/stubs.py when it doesn't, so this always runs.

  uv run python run.py --source file:videos/test_climb.mp4 --problem demo --voice print
  uv run python run.py --replay-pose data/pose_cache/clip.jsonl --problem demo --voice print
  uv run python run.py --source cam:0 --problem pink-v3 --voice real
  then open http://localhost:8000/

--source and --replay-pose together: poses come from the replay, video from the source.
--fast (tests only): no real-time pacing, no server, exits when the source runs out.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib
import logging
import signal
import time
from pathlib import Path

import yaml

from coach import stubs
from coach.bus import Bus

log = logging.getLogger("run")

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "data" / "logs"
SHUTDOWN_TIMEOUT_S = 3.0


def load(module: str, name: str, fallback=None):
    """The owner's implementation if it exists, otherwise the stub (or None)."""
    try:
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as e:
        using = getattr(fallback, "__name__", "nothing")
        log.info("%s.%s not available (%s); using %s", module, name, e, using)
        return fallback


def source_spec(text: str) -> int | str:
    """cam:N -> camera index, file:PATH -> path."""
    kind, _, value = text.partition(":")
    if kind == "cam" and value.isdigit():
        return int(value)
    if kind == "file" and value:
        if not Path(value).exists():
            raise argparse.ArgumentTypeError(f"no such file: {value}")
        return value
    raise argparse.ArgumentTypeError("expected cam:N or file:PATH")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Climbing Coach: live cues and debriefs.")
    p.add_argument("--source", type=source_spec, help="cam:N or file:PATH")
    p.add_argument("--replay-pose", type=Path, metavar="PATH.jsonl", help="replay recorded poses")
    p.add_argument(
        "--problem", required=True, metavar="ID", help="problem id (data/holdmaps/<ID>.json)"
    )
    p.add_argument("--voice", choices=("print", "real"), default="print")
    p.add_argument("--no-llm", action="store_true", help="no Gemini debriefs")
    p.add_argument("--config", type=Path, default=ROOT / "config" / "default.yaml")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--fast", action="store_true", help=argparse.SUPPRESS)  # tests only
    args = p.parse_args(argv)
    if args.source is None and args.replay_pose is None:
        p.error("give --source, --replay-pose, or both")
    if args.replay_pose and not args.replay_pose.exists():
        p.error(f"no such file: {args.replay_pose}")
    return args


class QuietServer:
    """Runs uvicorn in our loop without letting it take over Ctrl+C; run.py owns shutdown."""

    def __init__(self, app, host: str, port: int) -> None:
        import uvicorn

        class _Server(uvicorn.Server):
            @contextlib.contextmanager
            def capture_signals(self):
                yield

        config = uvicorn.Config(
            app, host=host, port=port, log_level="warning", timeout_graceful_shutdown=1
        )
        self.server = _Server(config)
        self.url = f"http://{host}:{port}/"

    async def serve(self) -> None:
        try:
            await self.server.serve()
        except (OSError, SystemExit) as e:  # e.g. port already in use; keep the rest running
            log.error("dashboard server failed to start on %s: %s", self.url, e)

    def stop(self) -> None:
        self.server.should_exit = True


def build_perception(args, cfg, bus, states):
    """The workers that feed states (and frames) for this combination of flags."""
    workers = []
    common = dict(problem_id=args.problem, cfg=cfg, bus=bus, out=states, fast=args.fast)
    if args.replay_pose:
        replay = load("coach.sources", "PoseReplay", stubs.PoseReplay)
        workers.append(replay(args.replay_pose, **common))
        if args.source is not None:  # video only, for the dashboard
            workers.append(stubs.VideoPerception(args.source, **common, frames_only=True))
    else:
        perception = load("coach.perception", "Perception", stubs.VideoPerception)
        workers.append(perception(args.source, **common))
    return workers


def build_voice(args, cfg, bus):
    if args.voice == "real":
        make_voice = load("coach.voice", "make_voice")
        if make_voice is not None:
            return make_voice(cfg, bus, no_llm=args.no_llm)
        log.warning("real voice isn't built yet; printing lines instead")
    return stubs.PrintVoice(cfg, bus, no_llm=args.no_llm)


async def main(args: argparse.Namespace) -> int:
    cfg = yaml.safe_load(args.config.read_text())
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    bus = Bus()
    bus.bind_loop()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    bus_log = stubs.BusLog(bus, LOG_DIR / f"bus-{args.problem}-{stamp}.jsonl")

    latest_queue = load("coach.sources", "LatestQueue", stubs.LatestQueue)
    states = latest_queue(loop)
    voice = build_voice(args, cfg, bus)
    brain = load("coach.brain", "Brain", stubs.StubBrain)(cfg, bus, voice)

    server = None
    if not args.fast:
        create_app = load("coach.server", "create_app", stubs.create_app)
        hello = {"problem_id": args.problem, "no_llm": args.no_llm}
        server = QuietServer(create_app(bus, hello), args.host, args.port)

    tasks = [asyncio.create_task(bus_log.run(), name="bus-log")]
    tasks.append(asyncio.create_task(brain.run(states), name="brain"))
    if server:
        tasks.append(asyncio.create_task(server.serve(), name="server"))
        log.info("dashboard at %s", server.url)

    workers = build_perception(args, cfg, bus, states)
    for w in workers:
        w.start()

    waits = [asyncio.create_task(stop.wait())]
    if args.fast:  # finite sources: finish when they all run out
        waits.append(asyncio.create_task(asyncio.to_thread(_wait_all, workers)))
    await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)

    log.info("shutting down")
    for w in workers:
        w.stop()
    if server:
        server.stop()
    for t in waits:
        t.cancel()
    for t in tasks:
        if t.get_name() != "server":
            t.cancel()
    await asyncio.wait(tasks, timeout=SHUTDOWN_TIMEOUT_S)
    voice.stop()
    with contextlib.suppress(AttributeError):
        voice.close()  # the real voice closes its audio stream here
    bus_log.close()
    bus.close()
    return 0


def _wait_all(workers) -> None:
    for w in workers:
        w.done.wait()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    raise SystemExit(asyncio.run(main(parse_args())))
