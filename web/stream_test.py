"""Test server for the dashboard's video path. No coach pipeline involved.

Reads a video file (looped) or a webcam and serves:
  /        web/index.html
  /ws      JPEG frames over the WebSocket (binary, or JSON {"topic": "frame"} with --base64)
  /video   the same frames as MJPEG, for the page's <img src="/video"> path

Only the newest frame is kept; slow clients skip frames rather than queueing them.

  uv run python web/stream_test.py --source data/clips/attempt1.mp4
  uv run python web/stream_test.py --source 0            # webcam 0
  then open http://localhost:8000/?video=ws   WebSocket video only
       or   http://localhost:8000/             MJPEG at /video first, then WebSocket frames
  Add --no-ws-video to test the MJPEG path on its own.
"""

import argparse
import asyncio
import base64
import json
import logging
import threading
import time
from pathlib import Path

import cv2
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

log = logging.getLogger("stream_test")
WEB_DIR = Path(__file__).resolve().parent


class LatestFrame:
    """Background reader that keeps only the newest JPEG-encoded frame."""

    def __init__(self, source: str, fps: float, width: int, quality: int) -> None:
        self.source = int(source) if source.isdigit() else source
        self.fps = fps
        self.width = width
        self.quality = quality
        self.jpeg: bytes | None = None
        self.seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def get(self) -> tuple[int, bytes | None]:
        with self._lock:
            return self.seq, self.jpeg

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise SystemExit(f"Could not open video source {self.source!r}")
        return cap

    def _run(self) -> None:
        cap = self._open()
        is_file = isinstance(self.source, str)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # For files, play at real speed and drop frames to hit the target fps.
        skip = max(1, round(src_fps / self.fps)) if is_file else 1
        period = skip / src_fps if is_file else 1.0 / self.fps
        log.info("source %r at %.1f fps, sending every %d frame(s)", self.source, src_fps, skip)
        next_t = time.monotonic()
        while not self._stop.is_set():
            for _ in range(skip):
                ok, img = cap.read()
                if not ok:
                    break
            if not ok:
                if is_file:  # loop the clip
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                log.warning("camera read failed, retrying")
                time.sleep(0.5)
                continue
            if self.width and img.shape[1] > self.width:
                h = round(img.shape[0] * self.width / img.shape[1])
                img = cv2.resize(img, (self.width, h), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            if ok:
                with self._lock:
                    self.jpeg = buf.tobytes()
                    self.seq += 1
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()  # fell behind; don't try to catch up
        cap.release()


def build_app(frames: LatestFrame, use_base64: bool, ws_video: bool) -> FastAPI:
    app = FastAPI()
    poll_s = 1.0 / (frames.fps * 2)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/fonts", StaticFiles(directory=WEB_DIR / "fonts"), name="fonts")

    @app.websocket("/ws")
    async def ws(sock: WebSocket) -> None:
        await sock.accept()
        log.info("ws client connected")
        await sock.send_text(json.dumps({"topic": "hello", "data": {"problem_id": "stream-test"}}))
        last = 0
        sent, t0 = 0, time.monotonic()
        try:
            while True:
                seq, jpeg = frames.get()
                if ws_video and jpeg is not None and seq != last:
                    last = seq
                    if use_base64:
                        b64 = base64.b64encode(jpeg).decode("ascii")
                        await sock.send_text(json.dumps({"topic": "frame", "data": b64}))
                    else:
                        await sock.send_bytes(jpeg)
                    sent += 1
                    if time.monotonic() - t0 >= 5:
                        log.info("ws: %.1f fps, %d KB/frame", sent / 5, len(jpeg) // 1024)
                        sent, t0 = 0, time.monotonic()
                await asyncio.sleep(poll_s)
        except (WebSocketDisconnect, RuntimeError):
            log.info("ws client disconnected")

    @app.get("/video")
    async def video() -> StreamingResponse:
        async def gen():
            last = 0
            while True:
                seq, jpeg = frames.get()
                if jpeg is not None and seq != last:
                    last = seq
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                        + jpeg
                        + b"\r\n"
                    )
                await asyncio.sleep(poll_s)

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    return app


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--source", required=True, help="video file path, or webcam index (e.g. 0)")
    p.add_argument("--fps", type=float, default=10.0, help="frames per second to send")
    p.add_argument("--width", type=int, default=960, help="downscale wider frames to this width")
    p.add_argument("--quality", type=int, default=75, help="JPEG quality 1-100")
    p.add_argument("--base64", action="store_true", help='send JSON {"topic":"frame"} not binary')
    p.add_argument("--no-ws-video", action="store_true", help="only serve video on /video")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    frames = LatestFrame(args.source, args.fps, args.width, args.quality)
    frames.start()
    app = build_app(frames, use_base64=args.base64, ws_video=not args.no_ws_video)
    log.info("open http://%s:%d/?video=ws", args.host, args.port)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        frames.stop()


if __name__ == "__main__":
    main()
