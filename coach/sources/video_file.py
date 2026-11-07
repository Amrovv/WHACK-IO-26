"""Video file source paced to real time, plus a demo window.

python -m coach.sources.video_file --source file:clip.mp4
python -m coach.sources.video_file --source cam:0
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import cv2

from coach.sources.base import Frame, FrameSource

log = logging.getLogger(__name__)

DEFAULT_FPS = 30.0  # used only when the container reports no fps


class FileSource:
    """Plays a video at 1x speed. If the consumer falls behind, frames are skipped
    (not delayed). ``fast=True`` disables pacing and skipping (tests only)."""

    def __init__(
        self,
        path: str | Path,
        fast: bool = False,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.path = Path(path)
        self.fast = fast
        self._clock = clock
        self._sleep = sleep
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {self.path}")
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
        self.skipped = 0
        self._closed = False

    def __iter__(self) -> Iterator[Frame]:
        period = 1.0 / self.fps
        t0 = self._clock()
        idx = 0
        while not self._closed:
            if not self._cap.grab():
                return
            if not self.fast:
                due = t0 + idx * period
                now = self._clock()
                if now - due > period:  # a whole frame late: skip it
                    self.skipped += 1
                    idx += 1
                    continue
                if now < due:
                    self._sleep(due - now)
            ok, frame = self._cap.retrieve()
            if not ok:
                return
            yield self._clock(), idx, frame
            idx += 1

    def close(self) -> None:
        self._closed = True
        self._cap.release()


def parse_source(spec: str, **kwargs) -> FrameSource:
    """``cam:<n>`` or ``file:<path>`` -> a FrameSource."""
    from coach.sources.camera import CameraSource

    kind, _, value = spec.partition(":")
    if kind == "cam" and value.isdigit():
        return CameraSource(int(value))
    if kind == "file" and value:
        return FileSource(value, **kwargs)
    raise ValueError(f"bad source {spec!r}; use cam:<n> or file:<path>")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a frame source and measure fps")
    parser.add_argument("--source", required=True, help="cam:<n> or file:<path>")
    parser.add_argument("--no-window", action="store_true", help="print fps only")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    source = parse_source(args.source)
    count, t_start, t_last = 0, None, None
    try:
        for t, idx, frame in source:
            t_start = t if t_start is None else t_start
            t_last = t
            count += 1
            if count % 30 == 0 and t > t_start:
                log.info("frame %d  measured %.1f fps", idx, (count - 1) / (t - t_start))
            if not args.no_window:
                cv2.imshow("source", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
        cv2.destroyAllWindows()
    if t_start is not None and t_last is not None and t_last > t_start:
        log.info(
            "done: %d frames, %.1f fps measured, skipped %s",
            count,
            (count - 1) / (t_last - t_start),
            getattr(source, "skipped", 0),
        )


if __name__ == "__main__":
    main()
