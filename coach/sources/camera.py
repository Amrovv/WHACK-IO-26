"""Live camera source (cv2.VideoCapture)."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator

import cv2

from coach.sources.base import Frame

log = logging.getLogger(__name__)


class CameraSource:
    def __init__(self, index: int = 0) -> None:
        self.index = index
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open camera {index}")
        self._closed = False

    def __iter__(self) -> Iterator[Frame]:
        idx = 0
        while not self._closed:
            ok, frame = self._cap.read()
            if not ok:
                log.warning("camera %d: read failed, stopping", self.index)
                return
            yield time.monotonic(), idx, frame
            idx += 1

    def close(self) -> None:
        self._closed = True
        self._cap.release()
