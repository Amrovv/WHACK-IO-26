"""Take photos from a webcam (setup tool, for empty-wall frames to build hold maps from).

    uv run --frozen python -m coach.perception.snap --camera 1
    uv run --frozen python -m coach.perception.snap --camera 1 --name gym1 --out data/footage

Live preview. Space or Enter = save a photo, q or Esc = quit. Photos are saved at the camera's
full resolution as <out>/<name>.jpg (then <name>_2.jpg, ... so nothing is overwritten).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

SETTLE_FRAMES = 30  # throw away the first frames so auto-exposure can settle


def next_path(out_dir: Path, name: str) -> Path:
    """First free <name>.jpg, <name>_2.jpg, <name>_3.jpg, ... in out_dir."""
    path = out_dir / f"{name}.jpg"
    n = 2
    while path.exists():
        path = out_dir / f"{name}_{n}.jpg"
        n += 1
    return path


def save_photo(frame: np.ndarray, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = next_path(out_dir, name)
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"could not write {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", type=int, default=0, help="camera index (cam:N)")
    parser.add_argument("--name", default="wall", help="file name without extension")
    parser.add_argument("--out", type=Path, default=Path("data/footage"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {args.camera}")
    try:
        for _ in range(SETTLE_FRAMES):
            cap.read()
        log.info("space/Enter = save photo · q/Esc = quit")
        saved_msg, shown = "", 0
        while True:
            ok, frame = cap.read()
            if not ok:
                log.warning("camera %d: read failed", args.camera)
                break
            preview = frame.copy()
            h, w = preview.shape[:2]
            cv2.putText(
                preview, f"{w}x{h}  space = save   q = quit", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
            )  # fmt: skip
            if shown > 0:
                cv2.putText(
                    preview, saved_msg, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )
                shown -= 1
            cv2.imshow("snap", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (32, 13):  # photo is the clean frame, without the text
                path = save_photo(frame, args.out, args.name)
                log.info("saved %s (%dx%d)", path, w, h)
                saved_msg, shown = f"saved {path.name}", 45
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
