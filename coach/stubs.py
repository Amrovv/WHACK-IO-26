"""Stand-ins for layers that aren't on this branch yet, plus D's own plumbing.

Owned by D. run.py uses each owner's real code when it's importable and falls back to the
class here when it isn't. The interface every layer plugs into is docs/LAYER-INTERFACES.md.

Stand-ins:
  VideoPerception   for A's pose pipeline: frames only, every state has person_found=False
  PoseReplay        replays a pose JSONL (A's pose-cache format) at recorded speed
  StubBrain         for B's Brain: a fake "stalling" cue every STUB_EVENT_PERIOD_S
  PrintVoice        logs lines instead of speaking, and publishes "spoken" like the real voice

D's own plumbing (not stand-ins):
  AsyncLatestQueue  the perception -> brain hand-off: put() from any thread, await get().
                    Not A's coach.sources.LatestQueue, which is thread -> thread inside
                    perception (blocking get); the two are different hops.
  BusLog            writes every bus topic except "frame" to a JSONL file.
  encode_jpeg       BGR frame -> dashboard JPEG.
  create_app        older copy of the dashboard server; coach/server/app.py is the real one.

  uv run python -m coach.stubs     # demo: replays a few fake states through the stub brain
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from coach.bus import Bus, to_jsonable
from coach.contracts import (
    TOPICS,
    AttemptSummary,
    ClimberState,
    CoachEvent,
    Hold,
    HoldMap,
    Limb,
    SpeakRequest,
    SpokenLine,
    Suppressed,
)

log = logging.getLogger("coach.stubs")

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
HOLDMAP_DIR = ROOT / "data" / "holdmaps"

# Streaming settings for the dashboard (not coaching thresholds, so not in config/default.yaml).
FRAME_FPS = 10.0  # frames sent to the dashboard per second
FRAME_MAX_WIDTH = 960  # downscale wider frames before JPEG encoding
FRAME_JPEG_QUALITY = 75
DASHBOARD_STATE_HZ = 10.0  # ClimberState forwarded to the dashboard at most this often
CLIENT_QUEUE_MAX = 64  # per-dashboard backlog; the oldest message is dropped when full
CAMERA_RETRY_S = 0.5
JOIN_TIMEOUT_S = 2.0

# Stub brain.
STUB_EVENT_PERIOD_S = 10.0  # one fake "stalling" event per this many seconds of capture time
STUB_LINES = {"stalling": "Keep moving, commit to the next hold."}


# ---- Queues -----------------------------------------------------------------------------


class AsyncLatestQueue:
    """One-slot queue between a producer thread and the asyncio loop. A new put() replaces
    an item nobody has taken yet, so the consumer always gets the newest frame's state."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._lock = threading.Lock()
        self._item: Any = None
        self._has_item = False
        self._ready = asyncio.Event()
        self._taken = threading.Event()
        self.dropped = 0

    def put(self, item: Any) -> None:
        with self._lock:
            if self._has_item:
                self.dropped += 1
            self._item, self._has_item = item, True
            self._taken.clear()
        self._loop.call_soon_threadsafe(self._ready.set)

    async def get(self) -> Any:
        while True:
            await self._ready.wait()
            with self._lock:
                self._ready.clear()
                if self._has_item:
                    item, self._item, self._has_item = self._item, None, False
                    self._taken.set()
                    return item

    def wait_taken(self, timeout: float) -> bool:
        """Block until the consumer took the last item (used by --fast so tests drop nothing)."""
        return self._taken.wait(timeout)


# ---- Hold map and pose parsing ----------------------------------------------------------


def load_holdmap(problem_id: str) -> HoldMap | None:
    """data/holdmaps/<problem_id>.json -> HoldMap, or None if it isn't there."""
    path = HOLDMAP_DIR / f"{problem_id}.json"
    if not path.exists():
        log.warning("no hold map at %s; the dashboard will show no holds", path)
        return None
    d = json.loads(path.read_text())
    holds = [
        Hold(
            id=h["id"], center=tuple(h["center"]), bbox=tuple(h["bbox"]), role=h.get("role", "mid")
        )
        for h in d.get("holds", [])
    ]
    return HoldMap(
        problem_id=d.get("problem_id", problem_id),
        colour=d.get("colour", ""),
        holds=holds,
        image_size=tuple(d.get("image_size", (0, 0))),
        mode=d.get("mode", "manual"),
    )


def _limb(key: str) -> Limb:
    return Limb(key) if key in Limb._value2member_map_ else Limb[key]


def state_from_dict(d: dict) -> ClimberState:
    """Inverse of to_jsonable() for ClimberState (limb keys may be "left_hand" or "LH")."""
    return ClimberState(
        t=float(d["t"]),
        frame_idx=int(d.get("frame_idx", 0)),
        person_found=bool(d.get("person_found", True)),
        keypoints={k: tuple(v) for k, v in (d.get("keypoints") or {}).items()},
        limbs={_limb(k): tuple(v) for k, v in (d.get("limbs") or {}).items()},
        contacts={_limb(k): v for k, v in (d.get("contacts") or {}).items()},
        hip=tuple(d.get("hip") or (0.5, 0.5)),
    )


def read_pose_jsonl(path: Path) -> list[ClimberState]:
    """A pose cache is one ClimberState per line, or a bus log (only "state" lines are used)."""
    states = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if "topic" in rec:
            if rec["topic"] != "state":
                continue
            rec = rec["payload"]
        states.append(state_from_dict(rec))
    return states


# ---- Perception stand-ins ---------------------------------------------------------------


class _Worker:
    """A daemon thread with stop() and a done flag."""

    def __init__(self, name: str) -> None:
        self._stop = threading.Event()
        self.done = threading.Event()
        self._thread = threading.Thread(target=self._guarded, name=name, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(JOIN_TIMEOUT_S)

    def _guarded(self) -> None:
        try:
            self._run()
        except Exception:
            log.exception("%s crashed", self._thread.name)
        finally:
            self.done.set()

    def _run(self) -> None:
        raise NotImplementedError


class PoseReplay(_Worker):
    """Stub perception: replays a pose JSONL at its recorded speed, looping.

    Capture times are shifted to now, so latency figures downstream stay meaningful.
    With fast=True it plays once with no waiting and hands over every state.
    """

    def __init__(self, path, problem_id, cfg, bus: Bus, out: AsyncLatestQueue, fast: bool = False):
        super().__init__("pose-replay")
        self.path, self.problem_id, self.bus, self.out, self.fast = (
            Path(path),
            problem_id,
            bus,
            out,
            fast,
        )

    def _run(self) -> None:
        holdmap = load_holdmap(self.problem_id)
        if holdmap:
            self.bus.publish("holdmap", holdmap)
        states = read_pose_jsonl(self.path)
        if not states:
            log.error("no poses in %s", self.path)
            return
        log.info("replaying %d poses from %s", len(states), self.path)
        while not self._stop.is_set():
            t0, base = states[0].t, time.monotonic()
            for s in states:
                if self._stop.is_set():
                    return
                s = ClimberState(**{**s.__dict__, "t": base + (s.t - t0)})
                if not self.fast:
                    delay = s.t - time.monotonic()
                    if delay > 0:
                        self._stop.wait(delay)
                self.bus.publish("state", s)
                self.out.put(s)
                if self.fast and hasattr(self.out, "wait_taken"):
                    self.out.wait_taken(JOIN_TIMEOUT_S)
            if self.fast:
                return


class VideoPerception(_Worker):
    """Stub perception for --source: no pose yet, just the camera or file.

    Publishes the newest frame as JPEG on "frame" (~FRAME_FPS) and, unless frames_only, a
    ClimberState with person_found=False per frame so the brain still sees time pass.
    Files loop at real speed; with fast=True a file plays once as quickly as it decodes.
    """

    def __init__(
        self, source, problem_id, cfg, bus: Bus, out, fast: bool = False, frames_only: bool = False
    ):
        super().__init__("video")
        self.source, self.problem_id, self.bus, self.out = source, problem_id, bus, out
        self.fast, self.frames_only = fast, frames_only

    def _run(self) -> None:
        import cv2  # only needed when there's video

        if not self.frames_only:
            holdmap = load_holdmap(self.problem_id)
            if holdmap:
                self.bus.publish("holdmap", holdmap)
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            log.error("could not open video source %r; running with no video", self.source)
            return
        is_file = isinstance(self.source, str)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        skip = max(1, round(src_fps / FRAME_FPS)) if is_file else 1
        period = skip / src_fps if is_file else 1.0 / FRAME_FPS
        log.info("video %r at %.1f fps, sending every %d frame(s)", self.source, src_fps, skip)
        frame_idx, next_t = 0, time.monotonic()
        try:
            while not self._stop.is_set():
                ok, img = False, None
                for _ in range(skip):
                    ok, img = cap.read()
                    if not ok:
                        break
                    frame_idx += 1
                if not ok:
                    if not is_file:
                        log.warning("camera read failed, retrying")
                        self._stop.wait(CAMERA_RETRY_S)
                        continue
                    if self.fast:
                        return
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop the clip
                    continue
                t = time.monotonic()
                self._publish_frame(cv2, img)
                if not self.frames_only:
                    s = ClimberState(t, frame_idx, False, {}, {}, {}, (0.5, 0.5))
                    self.bus.publish("state", s)
                    self.out.put(s)
                if self.fast:
                    continue
                next_t += period
                delay = next_t - time.monotonic()
                if delay > 0:
                    self._stop.wait(delay)
                else:
                    next_t = time.monotonic()  # fell behind; don't try to catch up
        finally:
            cap.release()

    def _publish_frame(self, cv2, img) -> None:
        jpeg = encode_jpeg(img)
        if jpeg is not None:
            self.bus.publish("frame", jpeg)


def encode_jpeg(img) -> bytes | None:
    """BGR image -> JPEG for the dashboard, downscaled to FRAME_MAX_WIDTH."""
    import cv2

    if img.shape[1] > FRAME_MAX_WIDTH:
        h = round(img.shape[0] * FRAME_MAX_WIDTH / img.shape[1])
        img = cv2.resize(img, (FRAME_MAX_WIDTH, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, FRAME_JPEG_QUALITY])
    return buf.tobytes() if ok else None


# ---- Brain and voice stand-ins ----------------------------------------------------------


class StubBrain:
    """Emits a fake "stalling" CoachEvent every STUB_EVENT_PERIOD_S of capture time."""

    def __init__(self, cfg: dict, bus: Bus, voice) -> None:
        self.bus, self.voice = bus, voice
        self.ttl_s = float(cfg["arbiter"]["ttl_s"]["p1"])

    async def run(self, states: AsyncLatestQueue) -> None:
        last_t: float | None = None
        while True:
            s: ClimberState = await states.get()
            if last_t is None:
                last_t = s.t
            if s.t - last_t < STUB_EVENT_PERIOD_S:
                continue
            last_t = s.t
            held = [h for h in s.contacts.values() if h is not None]
            ev = CoachEvent(
                t=s.t, kind="stalling", priority=1, ttl_s=self.ttl_s,
                data={"hold_id": max(held)} if held else {},
            )  # fmt: skip
            self.bus.publish("event", ev)
            req = SpeakRequest(event=ev, level=0, mood="neutral", deadline=s.t + self.ttl_s)
            if not self.voice.say(req):
                self.bus.publish("suppressed", Suppressed(event=ev, reason="late"))


class PrintVoice:
    """Voice that logs lines instead of speaking, and reports them on the bus like the real one."""

    def __init__(self, cfg: dict, bus: Bus, no_llm: bool = True) -> None:
        self.bus = bus

    def say(self, req: SpeakRequest) -> bool:
        now = time.monotonic()
        if now > req.deadline:
            return False
        kind = req.event.kind
        text = STUB_LINES.get(kind, kind.replace("_", " ").capitalize() + ".")
        log.info("COACH: %s", text)
        self.bus.publish(
            "spoken", SpokenLine(t_event=req.event.t, t_audio_start=now, kind=kind, text=text)
        )
        return True

    def is_speaking(self) -> bool:
        return False

    def stop(self) -> None:
        pass

    async def debrief(
        self, history: list[AttemptSummary], habits: dict, crux_hold: int | None
    ) -> None:
        last = history[-1] if history else None
        lines = [f"Attempt {last.attempt_no}: {last.outcome}." if last else "No attempts yet."]
        for line in lines:
            log.info("DEBRIEF: %s", line)
            self.bus.publish("debrief_text", line)


# ---- Bus log ----------------------------------------------------------------------------


class BusLog:
    """Writes {"t", "topic", "payload"} JSON lines for every topic except "frame" (raw JPEG)."""

    def __init__(self, bus: Bus, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", encoding="utf-8")
        self._queue: asyncio.Queue = asyncio.Queue()
        self._pending = [(bus.subscribe(t), t) for t in TOPICS if t != "frame"]
        self._tasks: list[asyncio.Task] = []

    async def run(self) -> None:
        self._tasks = [asyncio.create_task(self._pump(q, t)) for q, t in self._pending]
        while True:
            topic, payload = await self._queue.get()
            self._write(topic, payload)

    async def _pump(self, q: asyncio.Queue, topic: str) -> None:
        while True:
            self._queue.put_nowait((topic, await q.get()))

    def _write(self, topic: str, payload: Any) -> None:
        record = {"t": time.monotonic(), "topic": topic, "payload": to_jsonable(payload)}
        self._file.write(json.dumps(record) + "\n")

    def close(self) -> None:
        """Write whatever is still queued, then flush and close."""
        for task in self._tasks:
            task.cancel()
        for q, topic in self._pending:
            while not q.empty():
                self._queue.put_nowait((topic, q.get_nowait()))
        while not self._queue.empty():
            self._write(*self._queue.get_nowait())
        self._file.flush()
        self._file.close()
        log.info("bus log written to %s", self.path)


# ---- Dashboard server -------------------------------------------------------------------


class Hub:
    """Fans bus topics out to every connected dashboard. Subscribes once per topic, so clients
    can come and go without leaking bus queues. Frames go out as binary JPEG, the rest as
    {"topic", "data"} JSON. Slow clients lose their oldest messages, never stall the loop."""

    def __init__(self, bus: Bus, hello: dict) -> None:
        self.hello = json.dumps({"topic": "hello", "data": hello})
        self._sources = [(topic, bus.subscribe(topic)) for topic in TOPICS]
        self.clients: set[asyncio.Queue] = set()
        self.holdmap_msg: str | None = None
        self.frame: bytes | None = None
        self.frame_seq = 0
        self._frame_cond = asyncio.Condition()
        self._last_state = 0.0
        self._tasks: list[asyncio.Task] = []

    def start(self) -> None:
        self._tasks = [asyncio.create_task(self._pump(t, q)) for t, q in self._sources]

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _pump(self, topic: str, q: asyncio.Queue) -> None:
        while True:
            payload = await q.get()
            if topic == "frame":
                await self._new_frame(payload)
                continue
            if topic == "state":
                now = time.monotonic()
                if now - self._last_state < 1.0 / DASHBOARD_STATE_HZ:
                    continue
                self._last_state = now
            msg = json.dumps({"topic": topic, "data": to_jsonable(payload)})
            if topic == "holdmap":
                self.holdmap_msg = msg
            self._broadcast(msg)

    async def _new_frame(self, jpeg: bytes) -> None:
        async with self._frame_cond:
            self.frame, self.frame_seq = jpeg, self.frame_seq + 1
            self._frame_cond.notify_all()
        self._broadcast(jpeg)

    def _broadcast(self, msg: str | bytes) -> None:
        for q in self.clients:
            if q.full():
                q.get_nowait()
            q.put_nowait(msg)

    async def frames(self) -> AsyncIterator[bytes]:
        """Each new frame as it arrives (for MJPEG)."""
        seen = self.frame_seq
        while True:
            async with self._frame_cond:
                await self._frame_cond.wait_for(lambda seen=seen: self.frame_seq != seen)
                seen, jpeg = self.frame_seq, self.frame
            yield jpeg


def create_app(bus: Bus, hello: dict):
    """Dashboard server: web/index.html, its fonts, /ws (bus topics + frames), /video (MJPEG)."""
    hub = Hub(bus, hello)

    @contextlib.asynccontextmanager
    async def lifespan(app):
        hub.start()
        yield
        await hub.stop()

    app = FastAPI(lifespan=lifespan)
    app.state.hub = hub

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/fonts", StaticFiles(directory=WEB_DIR / "fonts"), name="fonts")

    @app.websocket("/ws")
    async def ws(sock: WebSocket) -> None:
        await sock.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=CLIENT_QUEUE_MAX)
        hub.clients.add(q)
        log.info("dashboard connected (%d open)", len(hub.clients))
        try:
            await sock.send_text(hub.hello)
            if hub.holdmap_msg:
                await sock.send_text(hub.holdmap_msg)
            while True:
                msg = await q.get()
                if isinstance(msg, bytes):
                    await sock.send_bytes(msg)
                else:
                    await sock.send_text(msg)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            hub.clients.discard(q)
            log.info("dashboard disconnected (%d open)", len(hub.clients))

    @app.get("/video")
    async def video() -> StreamingResponse:
        async def gen():
            async for jpeg in hub.frames():
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                    + jpeg
                    + b"\r\n"
                )

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    return app


# ---- Demo -------------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    async def _demo() -> None:
        bus = Bus()
        bus.bind_loop()
        spoken = bus.subscribe("spoken")
        states = AsyncLatestQueue(asyncio.get_running_loop())
        cfg = {"arbiter": {"ttl_s": {"p1": 1.2}}}
        brain = asyncio.create_task(StubBrain(cfg, bus, PrintVoice(cfg, bus)).run(states))
        now = time.monotonic()
        for i in range(12):  # 11 s of fake capture time, no sleeping needed
            states.put(ClimberState(now + i, i, True, {}, {}, {Limb.LH: 3}, (0.5, 0.5)))
            await asyncio.sleep(0)
        line = await asyncio.wait_for(spoken.get(), 1)
        log.info("stub brain spoke: %r", line.text)
        brain.cancel()

    asyncio.run(_demo())
