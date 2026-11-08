"""Dashboard server: serves web/index.html, forwards every bus topic over /ws, streams /video.

Owned by D.

  GET /         web/index.html (and /fonts/* for its self-hosted fonts)
  WS  /ws       one JSON message per bus item: {"topic", "t", "payload"}
                  t: server monotonic time when the item reached the hub
                  payload: to_jsonable(payload); "frame" JPEGs are base64 strings
                  "state" is throttled to STATE_HZ per client
                  on connect: "hello" (if given) and the latest "holdmap"
  GET /video    MJPEG (multipart/x-mixed-replace) of the latest "frame", ~FRAME_FPS

With a SourceControl (run.py's Session), the dashboard can also change what's analysed:
  GET  /api/sources   {"current": spec, "options": [{"spec", "name", "group"}]}
  POST /api/source    {"spec": "..."} -> switches; every dashboard gets the new "hello"
  POST /api/replay    a clip that ended (hello "ended": true) plays again; 409 if none has
  WS   /camera       binary JPEG frames from a dashboard's webcam (one sender at a time)
and, when it is also a HoldsControl, build the problem's hold map from the dashboard:
  /api/holds/...      see coach/server/holds.py

run.py awaits serve(bus, host, port). Replay a bus log with no pipeline running:

  uv run python -m coach.server.app --replay data/bus_logs/x.jsonl [--loop]
  then open http://localhost:8000/
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Annotated, Any, Protocol

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from coach.bus import Bus, to_jsonable
from coach.contracts import TOPICS
from coach.server.holds import UPLOAD_MAX_BYTES, HoldMapBuilder, Unavailable

log = logging.getLogger("coach.server")

WEB_DIR = Path(__file__).resolve().parents[2] / "web"

# Streaming settings (dashboard plumbing, not coaching thresholds).
STATE_HZ = 10.0  # max "state" messages per second per client
FRAME_FPS = 10.0  # max MJPEG frames per second per /video client
CLIENT_BACKLOG = 64  # messages queued per client; the oldest is dropped when full
SHUTDOWN_GRACE_S = 1  # uvicorn waits this long for open streams on shutdown
CAMERA_BUSY_CODE = 4409  # /camera close code when another dashboard is already sending

Clock = Callable[[], float]


class SourceControl(Protocol):
    """What the server needs to let the dashboard pick the source (run.py's Session)."""

    def listen(self, on_hello: Callable[[dict], None]) -> None: ...

    def sources(self) -> dict: ...

    async def switch(self, spec: str) -> dict: ...  # ValueError for a spec it doesn't offer

    def feed_camera(self, jpeg: bytes) -> bool: ...  # False when the browser camera isn't on


def encode(topic: str, t: float, payload: Any) -> str:
    """One bus item as the dashboard's JSON message."""
    if isinstance(payload, (bytes, bytearray)):
        payload = base64.b64encode(payload).decode("ascii")
    else:
        payload = to_jsonable(payload)
    return json.dumps({"topic": topic, "t": t, "payload": payload})


def mjpeg_part(jpeg: bytes) -> bytes:
    """One multipart/x-mixed-replace part."""
    head = f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n"
    return head.encode() + jpeg + b"\r\n"


class Client:
    """One dashboard connection: a bounded backlog plus its own state throttle."""

    def __init__(self, clock: Clock) -> None:
        self.queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=CLIENT_BACKLOG)
        self.clock = clock
        self.last_state = -float("inf")
        self.dropped = 0

    def offer(self, topic: str, msg: str) -> None:
        if topic == "state":
            now = self.clock()
            if now - self.last_state < 1.0 / STATE_HZ:
                return
            self.last_state = now
        if self.queue.full():  # newest wins; never block the hub on a slow browser
            self.queue.get_nowait()
            self.dropped += 1
        self.queue.put_nowait(msg)

    def close(self) -> None:
        """Queue the end-of-stream marker (None), even if the backlog is full."""
        if self.queue.full():
            self.queue.get_nowait()
        self.queue.put_nowait(None)


class Hub:
    """Subscribes to every bus topic once and fans items out to the connected clients.

    Subscribing once (not per connection) means clients can come and go without leaving
    dead queues on the bus.
    """

    def __init__(self, bus: Bus, hello: dict | None = None, clock: Clock = time.monotonic) -> None:
        self.clock = clock
        self.hello = hello
        self.clients: set[Client] = set()
        self.holdmap_msg: str | None = None
        self.frame: bytes | None = None
        self.frame_seq = 0
        self._frame_cond = asyncio.Condition()
        self._sources = [(topic, bus.subscribe(topic)) for topic in TOPICS]
        self._tasks: list[asyncio.Task] = []
        self.closed = False

    def start(self) -> None:
        self._tasks = [asyncio.create_task(self._pump(t, q)) for t, q in self._sources]

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    def close(self) -> None:
        """Ask every open /ws and /video connection to finish, so shutdown doesn't hang on them."""
        self.closed = True
        for client in self.clients:
            client.close()
        asyncio.get_running_loop().create_task(self._wake_frames())

    async def _wake_frames(self) -> None:
        async with self._frame_cond:
            self._frame_cond.notify_all()

    def set_hello(self, hello: dict) -> None:
        """A new source: tell every dashboard, and forget the old problem's hold map."""
        self.hello = hello
        self.holdmap_msg = None
        msg = encode("hello", self.clock(), hello)
        for client in self.clients:
            client.offer("hello", msg)

    def connect(self) -> Client:
        client = Client(self.clock)
        if self.hello is not None:
            client.offer("hello", encode("hello", self.clock(), self.hello))
        if self.holdmap_msg is not None:
            client.offer("holdmap", self.holdmap_msg)
        self.clients.add(client)
        return client

    def disconnect(self, client: Client) -> None:
        self.clients.discard(client)

    async def _pump(self, topic: str, queue: asyncio.Queue) -> None:
        while True:
            payload = await queue.get()
            await self.dispatch(topic, payload)

    async def dispatch(self, topic: str, payload: Any) -> None:
        if topic == "frame":
            # Live frames are JPEG bytes; replayed ones are base64 text.
            jpeg = base64.b64decode(payload) if isinstance(payload, str) else bytes(payload)
            async with self._frame_cond:
                self.frame, self.frame_seq = jpeg, self.frame_seq + 1
                self._frame_cond.notify_all()
        msg = encode(topic, self.clock(), payload)
        if topic == "holdmap":
            self.holdmap_msg = msg
        for client in self.clients:
            client.offer(topic, msg)

    async def frames(self) -> AsyncIterator[bytes]:
        """Yields each new frame, skipping any that arrive faster than FRAME_FPS."""
        seen, last = self.frame_seq, -float("inf")
        while True:
            async with self._frame_cond:
                await self._frame_cond.wait_for(
                    lambda seen=seen: self.closed or self.frame_seq != seen
                )
                if self.closed:
                    return
                seen, jpeg = self.frame_seq, self.frame
            now = self.clock()
            if now - last < 1.0 / FRAME_FPS:
                continue
            last = now
            yield jpeg


def create_app(
    bus: Bus,
    hello: dict | None = None,
    clock: Clock = time.monotonic,
    control: SourceControl | None = None,
) -> FastAPI:
    hub = Hub(bus, hello, clock)
    if control is not None:
        control.listen(hub.set_hello)
    camera_senders: set[WebSocket] = set()
    # Hold maps are built against the session that's running, so they need its controls too.
    builder = HoldMapBuilder(control) if hasattr(control, "read_still") else None

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        hub.start()
        yield
        await hub.stop()

    app = FastAPI(title="Climbing Coach", lifespan=lifespan)
    app.state.hub = hub
    app.state.holds = builder

    @app.get("/")
    async def index() -> FileResponse:
        # no-cache: browsers check back each load, so a plain reload shows dashboard changes.
        return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/fonts", StaticFiles(directory=WEB_DIR / "fonts"), name="fonts")

    @app.websocket("/ws")
    async def ws(sock: WebSocket) -> None:
        await sock.accept()
        client = hub.connect()
        log.info("dashboard connected (%d open)", len(hub.clients))
        try:
            while (msg := await client.queue.get()) is not None:
                await sock.send_text(msg)
            await sock.close()
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            hub.disconnect(client)
            log.info("dashboard disconnected (%d open)", len(hub.clients))

    @app.get("/video")
    async def video() -> StreamingResponse:
        async def parts() -> AsyncIterator[bytes]:
            async for jpeg in hub.frames():
                yield mjpeg_part(jpeg)

        return StreamingResponse(parts(), media_type="multipart/x-mixed-replace; boundary=frame")

    def need_control() -> SourceControl:
        if control is None:
            raise HTTPException(404, "this server can't change its source")
        return control

    @app.get("/api/sources")
    async def sources() -> dict:
        return need_control().sources()

    @app.post("/api/source")
    async def set_source(spec: str = Body(embed=True)) -> dict:
        ctl = need_control()
        try:
            return {"hello": await ctl.switch(spec)}
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/api/replay")
    async def replay() -> dict:
        """The Replay button: play an ended clip again (its recorded live pass)."""
        ctl = need_control()
        if not hasattr(ctl, "replay_clip"):
            raise HTTPException(404, "this server can't replay a clip")
        try:
            return {"hello": await ctl.replay_clip()}
        except ValueError as e:
            raise HTTPException(409, str(e)) from e

    @app.websocket("/camera")
    async def camera(sock: WebSocket) -> None:
        await sock.accept()
        if control is None or camera_senders:
            await sock.close(code=CAMERA_BUSY_CODE)
            return
        camera_senders.add(sock)
        log.info("browser camera connected")
        try:
            while True:
                control.feed_camera(await sock.receive_bytes())
        except (WebSocketDisconnect, RuntimeError, KeyError):
            pass  # KeyError: a text message, which a camera never sends
        finally:
            camera_senders.discard(sock)
            log.info("browser camera disconnected")

    def need_builder() -> HoldMapBuilder:
        if builder is None:
            raise HTTPException(404, "this server can't build hold maps")
        return builder

    async def run_builder(call: Awaitable[dict]) -> dict:
        """Builder errors as HTTP errors: 409 for a bad request, 503 without A's code."""
        try:
            return await call
        except Unavailable as e:
            raise HTTPException(503, str(e)) from e
        except (ValueError, KeyError, TypeError) as e:
            raise HTTPException(409, str(e)) from e

    @app.get("/api/holds")
    async def holds_status() -> dict:
        return await asyncio.to_thread(need_builder().status)  # reads and compares stills

    @app.get("/api/holds/view")
    async def holds_view() -> dict:
        return need_builder().view()  # lets a reloaded dashboard get back into an open editor

    @app.get("/api/holds/shapes")
    async def holds_shapes() -> dict:
        return await asyncio.to_thread(need_builder().shapes)

    @app.get("/api/holds/still.jpg")
    async def holds_still() -> Response:
        jpeg = need_builder().jpeg
        if jpeg is None:
            raise HTTPException(404, "no hold map is open")
        return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/api/holds/open")
    async def holds_open(
        still: Annotated[str, Body(embed=True)],
        at: Annotated[float | None, Body(embed=True)] = None,
    ) -> dict:
        return await run_builder(need_builder().open(still, at))

    @app.post("/api/holds/upload")
    async def holds_upload(request: Request, name: str = "picture") -> dict:
        """The raw picture in the body (the page sends a JPEG); ?name= is its file name."""
        if int(request.headers.get("content-length") or 0) > UPLOAD_MAX_BYTES:
            raise HTTPException(413, "that picture is too big")
        builder = need_builder()
        return await run_builder(builder.upload(name, await request.body()))

    @app.post("/api/holds/op")
    async def holds_op(body: Annotated[dict, Body()]) -> dict:
        args = dict(body)
        return await run_builder(need_builder().op(str(args.pop("op", "")), args))

    @app.post("/api/holds/save")
    async def holds_save() -> dict:
        return await run_builder(need_builder().save())

    @app.post("/api/holds/close")
    async def holds_close() -> dict:
        await need_builder().close()  # restarts analysis
        return {"open": False}

    return app


class _Server(uvicorn.Server):
    """uvicorn without its own signal handling: the caller (run.py) owns Ctrl+C."""

    @contextlib.contextmanager
    def capture_signals(self):
        yield


async def serve(bus: Bus, host: str = "127.0.0.1", port: int = 8000, hello: dict | None = None):
    """Run the dashboard server in the current loop until cancelled, then shut down cleanly."""
    app = create_app(bus, hello)
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        timeout_graceful_shutdown=SHUTDOWN_GRACE_S,
    )
    server = _Server(config)
    task = asyncio.ensure_future(server.serve())
    log.info("dashboard at http://%s:%d/", host, port)
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        app.state.hub.close()
        server.should_exit = True
        await task
        raise


# ---- Replay -----------------------------------------------------------------------------


def read_bus_log(path: Path) -> list[tuple[float, str, Any]]:
    """Bus log lines {"t", "topic", "payload"} -> [(t, topic, payload)], sorted by t."""
    records = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            records.append((float(rec["t"]), rec["topic"], rec["payload"]))
        except (ValueError, KeyError) as e:
            log.warning("%s:%d skipped (%s)", path, n, e)
    records.sort(key=lambda r: r[0])
    return records


async def replay(
    bus: Bus,
    records: list[tuple[float, str, Any]],
    loop: bool = False,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    """Publish records at 1x, keeping their original spacing. Payloads stay plain JSON."""
    if not records:
        return
    while True:
        prev = records[0][0]
        for t, topic, payload in records:
            if t > prev:
                await sleep(t - prev)
            prev = t
            bus.publish(topic, payload)
        if not loop:
            return


async def _replay_main(args: argparse.Namespace) -> None:
    bus = Bus()
    bus.bind_loop()
    records = read_bus_log(args.replay) if args.replay else []
    if records:
        span = records[-1][0] - records[0][0]
        log.info("replaying %d items (%.1f s) from %s", len(records), span, args.replay)
    hello = {"problem_id": "none"}
    if args.replay:
        hello = {"mode": "session_replay", "source": {"kind": "bus_log", "name": args.replay.name}}
    server = asyncio.create_task(serve(bus, args.host, args.port, hello))
    await asyncio.sleep(0)  # let the hub subscribe before the first publish
    try:
        await replay(bus, records, loop=args.loop)
        if records:
            log.info("replay finished; still serving (Ctrl+C to quit)")
        await server
    finally:
        server.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await server


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    p = argparse.ArgumentParser(description="Serve the dashboard; optionally replay a bus log.")
    p.add_argument("--replay", type=Path, metavar="PATH.jsonl", help="bus log to replay at 1x")
    p.add_argument("--loop", action="store_true", help="restart the replay when it ends")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    if args.replay and not args.replay.exists():
        p.error(f"no such file: {args.replay}")
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_replay_main(args))
