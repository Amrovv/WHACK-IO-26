"""Browser camera: JPEG frames a dashboard sends over WS /camera, read like any other camera.

Owned by D. The dashboard captures its webcam (getUserMedia), sends each frame as a binary
JPEG message, and perception iterates this object exactly like A's camera source:
(t, frame_idx, bgr) tuples until close(). t is time.monotonic() when the frame arrived.
Only the newest frame is kept, so a slow pose step drops frames instead of falling behind.

  uv run python -m coach.server.camera    # demo: feeds three frames from another thread
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator

import numpy as np

log = logging.getLogger("coach.server.camera")

BROWSER_SPEC = "browser"  # the source spec the dashboard sends to pick its own camera


class BrowserCamera:
    """put() JPEGs from the server's loop; iterate decoded frames on a perception thread."""

    kind = "browser_camera"

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._t = 0.0
        self._closed = False
        self.received = 0
        self.dropped = 0  # arrived before perception took the previous frame

    def put(self, jpeg: bytes) -> None:
        with self._cond:
            if self._jpeg is not None:
                self.dropped += 1
            self._jpeg, self._t = jpeg, self._clock()
            self.received += 1
            self._cond.notify()

    def __iter__(self) -> Iterator[tuple[float, int, np.ndarray]]:
        import cv2

        idx = 0
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._jpeg is not None or self._closed)
                if self._closed:
                    return
                jpeg, t, self._jpeg = self._jpeg, self._t, None
            img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                log.warning("dropped a browser frame that isn't a JPEG (%d bytes)", len(jpeg))
                continue
            yield t, idx, img
            idx += 1

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()


if __name__ == "__main__":
    import cv2

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cam = BrowserCamera()
    ok, buf = cv2.imencode(".jpg", np.zeros((48, 64, 3), np.uint8))

    def feed() -> None:
        for _ in range(3):
            cam.put(buf.tobytes())
            time.sleep(0.05)
        cam.close()

    threading.Thread(target=feed).start()
    for t, idx, img in cam:
        log.info("frame %d at t=%.3f, %dx%d", idx, t, img.shape[1], img.shape[0])
