"""Single entry point: wires perception, brain, voice, the dashboard server and the bus log.

No business logic here. Each layer comes from its owner's package when it exists and falls back
to coach/stubs.py when it doesn't, so this always runs. The interface between the layers is
docs/LAYER-INTERFACES.md.

  uv run python run.py --source file:videos/test_climb.mp4 --problem demo --voice print
  uv run python run.py --replay-pose data/pose_cache/clip.jsonl --problem demo --voice print
  uv run python run.py --source cam:0 --problem pink-v3 --voice real
  then open http://localhost:8000/

--transcript [PATH]: also write a readable transcript of the brain's decisions (every event,
every suppressed cue and why, what was spoken), for debugging cues; tail -f it while it runs.
--source and --replay-pose together: poses come from the replay, video from the source.
--fast (tests only): no real-time pacing, no server, exits when the source runs out.

The dashboard can switch the source while this runs (Session): any clip under videos/, or the
dashboard's own webcam, whose frames arrive over the server's /camera WebSocket.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import importlib
import inspect
import logging
import re
import signal
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import yaml

from coach import stubs
from coach.bus import Bus
from coach.server.camera import BROWSER_SPEC, BrowserCamera

log = logging.getLogger("run")

ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "data" / "logs"
VIDEO_DIR = ROOT / "videos"
# .mov is left out on purpose: the phone originals are HDR, and videos/mp4 has SDR copies.
VIDEO_EXTS = {".mp4", ".m4v", ".mkv", ".webm", ".avi"}
SWITCH_SETTLE_S = 0.05  # lets the old source's last frames reach the dashboard before "hello"
EMPTY_WALL_AT_S = 1.0  # hold maps are built from this far into an empty-wall clip
CLIP_END_MARGIN_S = 0.1  # a still is read at least this long before a clip's end (seeking to
# the very end reads nothing)
CLIP_NAME = re.compile(r"(.+?)_(\d{2})(?:-\d+)?_")  # <problem>_<clip no>_<name> (TEST_CLIMBS.md)
SHUTDOWN_TIMEOUT_S = 3.0
FRAME_POLL_S = 0.2  # pose thread re-checks for shutdown this often while waiting for a frame
JOIN_TIMEOUT_S = 2.0


def load(module: str, name: str, fallback=None):
    """The owner's implementation if it exists, otherwise the stub (or None)."""
    try:
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError) as e:
        using = getattr(fallback, "__name__", "nothing")
        log.info("%s.%s not available (%s); using %s", module, name, e, using)
        return fallback


def clip_info(path: str) -> dict:
    """Frame count, fps and picture size of a video file, so the dashboard can show clip
    progress and lay out a portrait clip before its first frame arrives."""
    import cv2

    cap = cv2.VideoCapture(path)
    try:
        frames, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
        ok, img = cap.read()  # the decoded frame's size is after any rotation metadata
    finally:
        cap.release()
    info = {"frames": frames, "fps": fps} if frames > 0 and fps > 0 else {}
    if ok:
        info.update(width=int(img.shape[1]), height=int(img.shape[0]))
    return info


def build_hello(args: argparse.Namespace, probe=clip_info) -> dict:
    """What the dashboard is told on connect: live camera, or real-time analysis of a clip."""
    hello = {"problem_id": args.problem, "no_llm": args.no_llm}
    if isinstance(args.source, BrowserCamera):
        hello.update(mode="live", source={"kind": "browser_camera"})
    elif isinstance(args.source, int) and not args.replay_pose:
        hello.update(mode="live", source={"kind": "camera", "index": args.source})
    elif isinstance(args.source, str):
        hello.update(mode="offline", source={"kind": "file", "name": Path(args.source).name})
        hello["source"].update(probe(args.source))
    else:
        hello.update(mode="offline", source={"kind": "poses", "name": args.replay_pose.name})
    return hello


def file_spec(path: str | Path) -> str:
    """file:<path>, relative to the repo when it's inside it, so the same clip has one spec."""
    full = Path(path).resolve()
    return f"file:{full.relative_to(ROOT) if full.is_relative_to(ROOT) else full}"


def spec_of(args: argparse.Namespace) -> str:
    """The source spec the dashboard shows as selected."""
    src = args.source
    if isinstance(src, BrowserCamera):
        return BROWSER_SPEC
    if isinstance(src, int):
        return f"cam:{src}"
    if isinstance(src, str):
        return file_spec(src)
    return f"poses:{args.replay_pose}"


def natural_key(path: Path) -> list:
    """climb_2 before climb_10."""
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", str(path))]


def list_clips(video_dir: Path = VIDEO_DIR) -> list[Path]:
    """Every video file under video_dir, in natural order."""
    if not video_dir.is_dir():
        return []
    clips = [p for p in video_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS and p.is_file()]
    return sorted(clips, key=natural_key)


def problem_for(clip: Path, default: str) -> str:
    """The problem id for a clip named <problem>_<clip no>_<name> (TEST_CLIMBS.md):
    "climb_2_05_hand_searching" -> "climb_2". Other names keep the default (--problem).

    A hold map only fits footage from the same camera position, so a clip never borrows a
    similarly named map (A's "climb2" is different footage from our "climb_2" clips).
    """
    m = CLIP_NAME.match(clip.stem)
    return m.group(1) if m else default


def is_empty_wall(clip: Path) -> bool:
    """Clip 00 of a problem films the empty wall (TEST_CLIMBS.md): hold maps are built from it."""
    m = CLIP_NAME.match(clip.stem)
    return bool(m) and m.group(2) == "00"


def clip_duration(path: str | Path) -> float | None:
    """A video file's length in seconds, or None when it can't be told."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        frames, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    return frames / fps if frames > 0 and fps > 0 else None


def frame_at(path: str | Path, seconds: float):
    """One BGR frame from a video file; ValueError if it can't be read."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        ok, img = cap.read()
    finally:
        cap.release()
    if not ok:
        raise ValueError(f"cannot read a frame at {seconds} s from {path}")
    return img


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
    p.add_argument(
        "--transcript",
        nargs="?",
        const=True,
        type=Path,
        metavar="PATH",
        help="write a readable transcript of the brain's decisions, suppressed cues included "
        "(default data/logs/transcript-<problem>-<time>.txt)",
    )
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
        self.app = app
        self.server = _Server(config)
        self.url = f"http://{host}:{port}/"

    async def serve(self) -> None:
        try:
            await self.server.serve()
        except (OSError, SystemExit) as e:  # e.g. port already in use; keep the rest running
            log.error("dashboard server failed to start on %s: %s", self.url, e)

    def stop(self) -> None:
        hub = getattr(self.app.state, "hub", None)
        if hub is not None:  # end open dashboards and video streams so shutdown is quick
            hub.close()
        self.server.should_exit = True


def a_parts() -> SimpleNamespace:
    """A's perception building blocks. Raises ImportError when A's code isn't on this branch."""
    from coach.perception.holds import load_holdmap_with_shapes
    from coach.perception.pose import PoseTracker, load_perception_config
    from coach.sources import LatestQueue, parse_source

    cfg = load_perception_config()
    return SimpleNamespace(
        open_source=parse_source,  # (spec, fast=) -> iterable of (t, frame_idx, bgr), .close()
        make_tracker=lambda holdmap, shapes: PoseTracker(cfg, holdmap=holdmap, shapes=shapes),
        load_holdmap=load_holdmap_with_shapes,  # problem_id -> (HoldMap, shapes); raises if missing
        frame_queue=LatestQueue,  # A's thread -> thread 1-slot queue
    )


TIME_FIELDS = ("t", "deadline")  # plus every field named t_*: monotonic times, shifted on replay
REPLAY_SKIP_TOPICS = {"frame", "holdmap"}  # the video is live on every pass; the map never changes


def shift_times(obj, dt: float, d_idx: int = 0):
    """A copy of a contracts dataclass with its monotonic times moved by dt seconds and its
    frame_idx by d_idx, nested dataclasses included. Anything else comes back as it is."""
    if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
        return obj
    changes = {}
    for f in dataclasses.fields(obj):
        v = getattr(obj, f.name)
        if (f.name in TIME_FIELDS or f.name.startswith("t_")) and isinstance(v, (int, float)):
            changes[f.name] = v + dt
        elif f.name == "frame_idx" and isinstance(v, int):
            changes[f.name] = v + d_idx
        elif dataclasses.is_dataclass(v):
            changes[f.name] = shift_times(v, dt, d_idx)
    return dataclasses.replace(obj, **changes)


class LoopReplay:
    """A looping clip is analysed live once, then replayed.

    A clip loops so the demo keeps going, but the climber walking back to the start isn't a new
    attempt. So the first pass runs live and everything the coach does is recorded, each at its
    time from the start of the pass: what perception and the brain publish (through TapBus) and
    what the brain asks the voice to do (through TapVoice). On later passes perception and the
    brain rest; the video plays and the recording is played back in step with it, its times
    shifted to now, so it looks and sounds like the first pass. Kept in memory only: a new
    source, a restart or a saved hold map analyses live again.

    Threads: perception records and advances from its threads, the brain records from the
    event loop; voice calls are replayed on the loop, like the brain makes them.
    """

    def __init__(self, bus, voice=None, loop=None, clock=time.monotonic, on_pass=None) -> None:
        self.bus, self.voice, self.loop, self.clock = bus, voice, loop, clock
        self.on_pass = on_pass  # called with the pass number as each replayed pass starts
        self.items: list[tuple[float, str, str | None, object]] = []  # (offset, kind, topic, x)
        self.passes = 0  # passes started so far
        self.recording = False
        self._t0 = 0.0  # when the live pass started
        self._pass_start = 0.0  # when the current replayed pass started
        self._idx_offset = 0  # frame_idx of the current pass's first frame
        self._next = 0  # the next item to play in this pass
        self._lock = threading.Lock()
        self._tasks: set = set()  # replayed debriefs, kept until done

    @property
    def replaying(self) -> bool:
        return self.passes >= 2

    def start_pass(self, idx_offset: int) -> None:
        """Perception calls this at the first frame of every pass of the clip."""
        with self._lock:
            leftover = self.items[self._next :] if self.passes >= 2 else []
            now = self.clock()
            if self.passes == 0:
                self._t0, self.recording = now, True
            elif self.passes == 1:
                self.recording = False
                log.info("clip looped: replaying the live pass (%d items)", len(self.items))
            self.passes += 1
            number = self.passes
        for item in leftover:  # what the coach did after the last frame of the clip
            self._play(item)
        if number >= 2 and self.on_pass is not None:  # dashboards start this pass fresh
            self.on_pass(number)
        with self._lock:
            self._pass_start, self._idx_offset, self._next = now, idx_offset, 0

    def record(self, kind: str, topic: str | None, payload) -> None:
        with self._lock:
            if self.recording:
                self.items.append((self.clock() - self._t0, kind, topic, payload))

    def advance(self) -> None:
        """Play everything due by now in this pass. Perception calls this every frame."""
        with self._lock:
            if not self.replaying:
                return
            elapsed = self.clock() - self._pass_start
            start = self._next
            while self._next < len(self.items) and self.items[self._next][0] <= elapsed:
                self._next += 1
            due = self.items[start : self._next]
        for item in due:
            self._play(item)

    def _play(self, item) -> None:
        offset, kind, topic, payload = item
        if kind == "bus":
            dt = self._pass_start - self._t0
            self.bus.publish(topic, shift_times(payload, dt, self._idx_offset))
        elif self.voice is not None and self.loop is not None:
            dt = self.clock() - (self._t0 + offset)  # the deadline keeps its slack from now
            self.loop.call_soon_threadsafe(self._voice_call, kind, shift_times(payload, dt))

    def _voice_call(self, kind: str, payload) -> None:
        if kind == "say":
            self.voice.say(payload)
        elif kind == "stop":
            self.voice.stop()
        elif kind == "debrief":
            result = self.voice.debrief(*payload)
            if inspect.isawaitable(result):
                task = asyncio.ensure_future(result)
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)


class TapBus:
    """The bus as perception and the brain see it: what they publish is also recorded."""

    def __init__(self, bus, replay: LoopReplay) -> None:
        self._bus, self._replay = bus, replay

    def publish(self, topic: str, payload) -> None:
        if topic not in REPLAY_SKIP_TOPICS:
            self._replay.record("bus", topic, payload)
        self._bus.publish(topic, payload)

    def __getattr__(self, name):
        return getattr(self._bus, name)


class TapVoice:
    """The voice as the brain sees it: what the brain asks of it is also recorded."""

    def __init__(self, voice, replay: LoopReplay) -> None:
        self._voice, self._replay = voice, replay

    def say(self, req) -> bool:
        self._replay.record("say", None, req)
        return self._voice.say(req)

    def stop(self) -> None:
        self._replay.record("stop", None, None)
        self._voice.stop()

    def debrief(self, history, habits, crux_hold):
        self._replay.record("debrief", None, (list(history), dict(habits), crux_hold))
        return self._voice.debrief(history, habits, crux_hold)

    def __getattr__(self, name):
        return getattr(self._voice, name)


class PoseThreads:
    """Real perception wired from A's parts.

    A reader thread puts frames into A's 1-slot LatestQueue (so a slow pose step drops old
    frames instead of falling behind); a pose thread runs PoseTracker on the newest frame and
    publishes "state" (to the bus and the brain's queue) and "frame" (~FRAME_FPS, for the
    dashboard). The hold map is published once at start. Files loop unless fast=True, which
    runs everything on one thread, drops nothing and stops at the end of the file. With a
    LoopReplay, only a file's first pass is analysed; later passes show the video and play
    the recording back (the reader thread does that; the pose thread rests).
    """

    def __init__(
        self, source, problem_id, cfg, bus, out, fast=False, parts=None, replay=None
    ) -> None:
        self.parts = parts or a_parts()
        self.replay: LoopReplay | None = replay
        # A frame source object (the browser camera) is read as it is, not opened by spec.
        self.custom = None if isinstance(source, (int, str)) else source
        if self.custom is not None:
            self.spec = BROWSER_SPEC
        else:
            self.spec = f"cam:{source}" if isinstance(source, int) else f"file:{source}"
        self.is_file = isinstance(source, str)
        self.bus, self.out, self.fast = bus, out, fast
        try:
            self.holdmap, shapes = self.parts.load_holdmap(problem_id)
        except FileNotFoundError:
            log.warning("no hold map for %r; poses will have no contacts", problem_id)
            self.holdmap, shapes = None, {}
        self.tracker = self.parts.make_tracker(self.holdmap, shapes)  # may raise: no pose model
        self.frames = self.parts.frame_queue()
        self.done = threading.Event()
        self._stop = threading.Event()
        self._read_done = threading.Event()
        self._last_frame_t = -float("inf")
        self.latest_frame = None  # newest BGR frame, for building a hold map from it
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        if self.holdmap is not None:
            self.bus.publish("holdmap", self.holdmap)
        targets = [self._fast] if self.fast else [self._read, self._pose]
        self._threads = [
            threading.Thread(target=self._guard, args=(f,), daemon=True) for f in targets
        ]
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        self._stop.set()
        if self.custom is not None:
            self.custom.close()  # wakes the reader if it's waiting for the next frame
        self.frames.close()
        for t in self._threads:
            t.join(JOIN_TIMEOUT_S)
        self.tracker.close()

    def _guard(self, target) -> None:
        try:
            target()
        except Exception:
            log.exception("perception thread %s crashed", target.__name__)
        finally:
            if target is not self._read:
                self.done.set()

    def _frames(self):
        """(t, frame_idx, bgr) from the source; files restart at the end unless fast."""
        offset = 0
        while not self._stop.is_set():
            if self.custom is not None:
                source = self.custom
            else:
                source = self.parts.open_source(
                    self.spec, **({"fast": self.fast} if self.is_file else {})
                )
            last = -1
            looping = self.replay is not None and self.is_file and not self.fast
            try:
                for t, idx, img in source:
                    if self._stop.is_set():
                        return
                    if looping and last < 0:  # first frame of this pass
                        self.replay.start_pass(offset)
                    last = idx
                    yield t, idx + offset, img
            finally:
                source.close()
            if self.fast or not self.is_file or last < 0:
                return
            offset += last + 1  # keep frame_idx growing so the dashboard can count loops

    def _read(self) -> None:
        try:
            for frame in self._frames():
                if self.replay is not None and self.replay.replaying:
                    self._show(*frame)  # a replayed pass: video only, the recording does the rest
                else:
                    self.frames.put(frame)
        finally:
            self._read_done.set()
            self.frames.close()

    def _pose(self) -> None:
        while not self._stop.is_set():
            frame = self.frames.get(timeout=FRAME_POLL_S)
            if frame is None:
                if self._read_done.is_set():
                    return
                continue
            self._process(*frame)

    def _fast(self) -> None:
        for frame in self._frames():
            self._process(*frame)
            if hasattr(self.out, "wait_taken"):
                self.out.wait_taken(JOIN_TIMEOUT_S)

    def _process(self, t: float, idx: int, img) -> None:
        state = self.tracker.process(t, idx, img)
        self.bus.publish("state", state)
        self.out.put(state)
        self._show(t, idx, img)

    def _show(self, t: float, idx: int, img) -> None:
        """Keep the frame for the hold map editor, send it to the dashboard (~FRAME_FPS), and
        on a replayed pass play what the coach did by now."""
        self.latest_frame = img
        if self.replay is not None:
            self.replay.advance()
        if t - self._last_frame_t >= 1.0 / stubs.FRAME_FPS:
            self._last_frame_t = t
            jpeg = stubs.encode_jpeg(img)
            if jpeg is not None:
                self.bus.publish("frame", jpeg)


def build_perception(args, cfg, bus, states, parts=None, replay=None):
    """The workers that feed states (and frames) for this combination of flags. A LoopReplay
    makes a looping clip analysed once and replayed after (PoseThreads only)."""
    common = dict(problem_id=args.problem, cfg=cfg, bus=bus, out=states, fast=args.fast)
    if args.replay_pose:
        workers = [stubs.PoseReplay(args.replay_pose, **common)]
        if args.source is not None:  # video only, for the dashboard
            workers.append(stubs.VideoPerception(args.source, **common, frames_only=True))
        return workers
    try:
        return [PoseThreads(args.source, **common, parts=parts, replay=replay)]
    except ImportError as e:
        log.info("A's perception isn't on this branch (%s); video only, no pose", e)
    except Exception as e:  # e.g. pose model missing: keep the demo running
        log.warning("pose pipeline failed to start (%s); video only, no pose", e)
    return [stubs.VideoPerception(args.source, **common)]


def build_voice(args, cfg, bus):
    if args.voice == "real":
        make_voice = load("coach.voice", "make_voice")
        if make_voice is not None:
            return make_voice(cfg, bus, no_llm=args.no_llm)
        log.warning("real voice isn't built yet; printing lines instead")
    return stubs.PrintVoice(cfg, bus, no_llm=args.no_llm)


class Session:
    """What's being analysed right now: one source, its perception workers and a brain.

    switch() swaps the source while the server, voice and bus log keep running. The brain
    is rebuilt each time, so attempts and the hold map start fresh for the new problem.
    (The bus can't unsubscribe, so each old brain leaves two idle queues behind; a few
    switches in a demo make that negligible.) This is the server's SourceControl, and its
    HoldsControl for building hold maps: stills to build from, and reload() to restart
    analysis so A's contacts and B's brain load a freshly saved map.
    """

    def __init__(
        self, args, cfg, bus, voice, make_brain, parts=None, video_dir: Path = VIDEO_DIR
    ) -> None:
        self.initial = self.args = args
        self.cfg, self.bus, self.voice, self.make_brain = cfg, bus, voice, make_brain
        self.parts, self.video_dir = parts, video_dir
        self.workers: list = []
        self.replay: LoopReplay | None = None  # a clip's recorded first pass
        self._loop: asyncio.AbstractEventLoop | None = None
        self.brain_task: asyncio.Task | None = None
        self.camera: BrowserCamera | None = None
        self.hello: dict = {}
        self.number = 0
        self.paused = False  # stopped while the hold map is edited
        self._paused_frame = None  # the last frame before the pause: the editor can build from it
        self._durations: dict[str, float | None] = {}  # clip spec -> seconds, for the stills
        self._listeners: list = []
        self._lock = asyncio.Lock()

    def listen(self, on_hello) -> None:
        self._listeners.append(on_hello)

    def options(self) -> list[dict]:
        """Every source the dashboard may pick: the one we started with, the clips, the webcam."""
        first = spec_of(self.initial)
        opts = []
        if not first.startswith("file:"):
            src = self.initial.source
            name = f"Camera {src} (server)" if isinstance(src, int) else f"Poses: {first[6:]}"
            opts.append({"spec": first, "name": name, "group": "Started with"})
        clips = list_clips(self.video_dir)
        if first.startswith("file:") and all(file_spec(p) != first for p in clips):
            clips.insert(0, ROOT / first[5:])
        for clip in clips:
            folder = clip.parent.resolve()
            group = folder.relative_to(ROOT) if folder.is_relative_to(ROOT) else folder
            opts.append({"spec": file_spec(clip), "name": clip.name, "group": str(group)})
        opts.append({"spec": BROWSER_SPEC, "name": "This device's camera", "group": "Camera"})
        return opts

    def sources(self) -> dict:
        return {"current": self.hello.get("spec"), "options": self.options()}

    def args_for(self, spec: str) -> argparse.Namespace:
        """Run arguments for an offered spec; ValueError for anything else."""
        if spec not in {o["spec"] for o in self.options()}:
            raise ValueError(f"not an offered source: {spec!r}")
        if spec == spec_of(self.initial):
            return self.initial
        base = {**vars(self.initial), "replay_pose": None}
        if spec == BROWSER_SPEC:
            return argparse.Namespace(**{**base, "source": BrowserCamera()})
        path = spec.removeprefix("file:")
        problem = problem_for(Path(path), self.initial.problem)
        return argparse.Namespace(**{**base, "source": str(ROOT / path), "problem": problem})

    # start, stop and switch share one lock: a switch the dashboard asks for while the first
    # source is still starting (or while shutting down) waits instead of running alongside it.
    async def start(self, args: argparse.Namespace) -> None:
        async with self._lock:
            await self._start(args)

    async def stop(self) -> None:
        async with self._lock:
            await self._stop()

    async def _start(self, args: argparse.Namespace) -> None:
        self.args = args
        self.number += 1
        self.paused, self._paused_frame = False, None
        self.camera = args.source if isinstance(args.source, BrowserCamera) else None
        loop = self._loop = asyncio.get_running_loop()
        states = stubs.AsyncLatestQueue(loop)
        # A clip loops: its first pass is analysed live and recorded, later passes replay that.
        bus, voice, self.replay = self.bus, self.voice, None
        if isinstance(args.source, str) and not args.fast and not args.replay_pose:
            self.replay = LoopReplay(self.bus, self.voice, loop, on_pass=self._replayed_pass)
            bus, voice = TapBus(self.bus, self.replay), TapVoice(self.voice, self.replay)
        # Model loading and clip probing block, so they run off the loop.
        workers = await asyncio.to_thread(
            build_perception, args, self.cfg, bus, states, self.parts, self.replay
        )
        hello = await asyncio.to_thread(build_hello, args) if not args.fast else {}
        # The brain subscribes to "holdmap" when built, so build it before perception starts.
        brain = self.make_brain(self.cfg, bus, voice)
        self.brain_task = asyncio.create_task(brain.run(states), name="brain")
        self.hello = {**hello, "session": self.number, "spec": spec_of(args)}
        for on_hello in self._listeners:
            on_hello(self.hello)
        self.workers = workers
        for w in workers:
            w.start()
        log.info("analysing %s (problem %s)", self.hello["spec"], args.problem)

    def _replayed_pass(self, number: int) -> None:
        """A clip started a replayed pass (perception's reader thread): over to the loop."""
        replay = self.replay
        self._loop.call_soon_threadsafe(self._announce_pass, replay, number)

    def _announce_pass(self, replay: LoopReplay, number: int) -> None:
        """Dashboards start each pass fresh, as on the first: a new hello (with the pass
        number) clears their attempts and timeline, then the hold map goes out again."""
        if replay is not self.replay:  # the source changed meanwhile
            return
        self.hello = {**self.hello, "pass": number}
        for on_hello in self._listeners:
            on_hello(self.hello)
        holdmap = next((w.holdmap for w in self.workers if getattr(w, "holdmap", None)), None)
        if holdmap is not None:
            self.bus.publish("holdmap", holdmap)

    async def _stop(self) -> None:
        self.camera = None  # webcam frames that arrive from now on are ignored
        workers, self.workers = self.workers, []
        for w in workers:
            await asyncio.to_thread(w.stop)
        if self.brain_task is not None:
            self.brain_task.cancel()
            await asyncio.gather(self.brain_task, return_exceptions=True)
            self.brain_task = None
        self.voice.stop()

    async def switch(self, spec: str) -> dict:
        async with self._lock:
            args = self.args_for(spec)
            if spec == self.hello.get("spec"):
                return self.hello
            await self._stop()
            await asyncio.sleep(SWITCH_SETTLE_S)
            await self._start(args)
            return self.hello

    def feed_camera(self, jpeg: bytes) -> bool:
        if self.camera is None:
            return False
        self.camera.put(jpeg)
        return True

    def wait_done(self) -> None:
        """Block until every worker of the current source has finished (finite sources)."""
        _wait_all(self.workers)

    # -- hold maps (coach/server/holds.py) ---------------------------------------------------

    @property
    def problem_id(self) -> str:
        return self.args.problem

    def latest_frame(self):
        """The newest frame perception saw (the last one before a pause), or None (pose
        replays have no video)."""
        for w in self.workers:
            img = getattr(w, "latest_frame", None)
            if img is not None:
                return img
        return self._paused_frame

    def stills(self) -> list[dict]:
        """Pictures a hold map can be built from: this problem's empty-wall clips first (the
        wall with nobody in front of it), then the clip being analysed, then the current frame.
        Videos carry their "duration" (seconds), so the editor can pick any moment of them;
        "live" marks the video being analysed. The builder moves the walls filmed from
        somewhere else to the end."""
        walls = [
            clip
            for clip in list_clips(self.video_dir)
            if is_empty_wall(clip) and problem_for(clip, "") == self.problem_id
        ]
        walls.sort(key=lambda clip: re.search(r" \(\d+\)$", clip.stem) is not None)  # copies last
        out = [self._video_still(clip, f"Empty wall · {clip.name}") for clip in walls]
        if isinstance(self.args.source, str):  # a clip: any moment of it can be the still
            spec = file_spec(self.args.source)
            mine = next((s for s in out if s["key"] == spec), None)
            if mine is None:
                mine = self._video_still(self.args.source, f"This video · {Path(spec).name}")
                out.append(mine)
            mine["live"] = True
        if self.latest_frame() is not None:
            out.append({"key": "frame", "name": "Current frame", "live": True})
        return out

    def _video_still(self, path: str | Path, name: str) -> dict:
        key = file_spec(path)
        if key not in self._durations:  # probing opens the file: once per clip
            self._durations[key] = clip_duration(ROOT / key.removeprefix("file:"))
        duration = self._durations[key]
        if not duration:
            return {"key": key, "name": name}
        at = min(EMPTY_WALL_AT_S, max(duration - CLIP_END_MARGIN_S, 0.0))  # the default moment
        return {"key": key, "name": name, "duration": duration, "at": at}

    def read_still(self, key: str, at: float | None = None):
        """The picture for a still key (BGR): a video's frame `at` seconds in (default
        EMPTY_WALL_AT_S, kept inside the clip). Blocking: call it off the loop."""
        if key == "frame":
            img = self.latest_frame()
            if img is None:
                raise ValueError("there's no frame yet")
            return img.copy()
        still = next((s for s in self.stills() if s["key"] == key), None)
        if still is None:
            raise ValueError(f"not an offered still: {key!r}")
        t = EMPTY_WALL_AT_S if at is None else float(at)
        if still.get("duration"):
            t = min(max(t, 0.0), max(still["duration"] - CLIP_END_MARGIN_S, 0.0))
        return frame_at(ROOT / key.removeprefix("file:"), t)

    async def pause(self) -> None:
        """Stop analysing (perception, brain and voice) while the hold map is edited, keeping
        the last frame for the editor. resume() or reload() starts again."""
        async with self._lock:
            if self.paused:
                return
            frame = self.latest_frame()
            await self._stop()
            self.paused, self._paused_frame = True, frame
            log.info("analysis paused while the hold map is edited")

    async def resume(self) -> dict:
        """Start analysing again after pause(): the same source from its start, with a fresh
        brain. (Picking up mid-clip would hand the brain a gap in state.t as long as the pause.)"""
        async with self._lock:
            if self.paused:
                await self._restart()
            return self.hello

    async def reload(self) -> dict:
        """Restart analysis on the same source, so perception and the brain reload the hold map."""
        async with self._lock:
            await self._restart()
            return self.hello

    async def _restart(self) -> None:
        args = self.args
        if isinstance(args.source, BrowserCamera):  # the old one is closed by _stop
            args = argparse.Namespace(**{**vars(args), "source": BrowserCamera()})
        await self._stop()
        await asyncio.sleep(SWITCH_SETTLE_S)
        await self._start(args)


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
    transcript = None
    if args.transcript:  # a readable log of the live brain's decisions, suppressed cues too
        path = args.transcript if isinstance(args.transcript, Path) else None
        path = path or LOG_DIR / f"transcript-{args.problem}-{stamp}.txt"
        transcript = stubs.Transcript(bus, path)
        log.info("writing the brain transcript to %s", path)

    # The state hand-off inside Session is always D's AsyncLatestQueue; A's
    # coach.sources.LatestQueue is the thread -> thread queue inside perception
    # (see docs/LAYER-INTERFACES.md).
    voice = build_voice(args, cfg, bus)
    make_brain = load("coach.brain", "Brain", stubs.StubBrain)
    session = Session(args, cfg, bus, voice, make_brain)
    if transcript is not None:
        session.listen(transcript.on_hello)  # a block per source; looped replays not repeated

    server = None
    if not args.fast:
        create_app = load("coach.server", "create_app", stubs.create_app)
        if "control" in inspect.signature(create_app).parameters:
            app = create_app(bus, None, control=session)
        else:  # the older stub server can't switch sources
            app = create_app(bus, build_hello(args))
        server = QuietServer(app, args.host, args.port)

    tasks = [asyncio.create_task(bus_log.run(), name="bus-log")]
    if transcript is not None:
        tasks.append(asyncio.create_task(transcript.run(), name="transcript"))
    if server:
        tasks.append(asyncio.create_task(server.serve(), name="server"))
        log.info("dashboard at %s", server.url)

    await session.start(args)

    waits = [asyncio.create_task(stop.wait())]
    if args.fast:  # finite sources: finish when they all run out
        waits.append(asyncio.create_task(asyncio.to_thread(session.wait_done)))
    await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)

    log.info("shutting down")
    await session.stop()
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
    if transcript is not None:
        transcript.close()
    bus.close()
    return 0


def _wait_all(workers) -> None:
    for w in workers:
        w.done.wait()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    raise SystemExit(asyncio.run(main(parse_args())))
