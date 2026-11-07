"""Frame source interface and the 1-slot hand-off queue."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Protocol

import numpy as np

# (t_capture from time.monotonic(), frame index, BGR image)
Frame = tuple[float, int, np.ndarray]


class FrameSource(Protocol):
    """Yields frames until exhausted (files) or closed (cameras)."""

    def __iter__(self) -> Iterator[Frame]: ...

    def close(self) -> None: ...


class LatestQueue[T]:
    """Thread-safe 1-slot queue: a new put replaces (drops) the unread item."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._item: T | None = None
        self._has_item = False
        self._closed = False
        self.dropped = 0

    def put(self, item: T) -> None:
        with self._cond:
            if self._has_item:
                self.dropped += 1
            self._item = item
            self._has_item = True
            self._cond.notify()

    def get(self, timeout: float | None = None) -> T | None:
        """Block for the newest item. Returns None on timeout or once closed and empty."""
        with self._cond:
            self._cond.wait_for(lambda: self._has_item or self._closed, timeout)
            if not self._has_item:
                return None
            item = self._item
            self._item = None
            self._has_item = False
            return item

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
