"""Build a HoldMap from one frame of the empty wall.

Two ways to find the route's holds:
  * detector mode: a Roboflow detector finds all holds, then only boxes the same colour as the
    clicked hold are kept (`detect_holds` + `filter_by_colour`)
  * colour-only mode: threshold the clicked hue directly (`colour_only_holds`); the default for
    the paper wall and the fallback for the gym

Setup-time only: `detect_holds` uses the network and must never run in the live loop.

Pixel boxes are (x0, y0, x1, y1). OpenCV hue is 0-179 and circular (red is both 0 and 179).

    uv run python -m coach.perception.holds --video data/footage/clip.mp4 --at 1.0 \
        --click 420 150 --problem demo --out-image holds.jpg
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np

from coach.contracts import Hold, HoldMap
from coach.perception.config import REPO_ROOT, load_config

log = logging.getLogger(__name__)

Box = tuple[int, int, int, int]  # x0, y0, x1, y1 in pixels
HUE_RANGE = 180

# Spoken colour names by OpenCV hue (upper bound exclusive). Wraps: hues >= 170 are red again.
_HUE_NAMES = (
    (8, "red"),
    (20, "orange"),
    (35, "yellow"),
    (85, "green"),
    (100, "teal"),
    (130, "blue"),
    (145, "purple"),
    (170, "pink"),
    (180, "red"),
)


def colour_name(hue: float) -> str:
    for upper, name in _HUE_NAMES:
        if hue < upper:
            return name
    return "red"


def hue_distance(a: float, b: float) -> float:
    """Circular distance between two OpenCV hues (0-179)."""
    d = abs(a - b) % HUE_RANGE
    return min(d, HUE_RANGE - d)


def load_holds_config(path: str | Path | None = None) -> dict:
    return load_config(path)["holds"]


def sample_colour(image_bgr: np.ndarray, point: tuple[int, int], cfg: dict) -> tuple[float, float]:
    """Median (hue, saturation) in a small patch around a clicked point, ignoring pixels whose
    saturation is below `min_saturation` (chalk, grey wall). Hue uses the circular median."""
    x, y = point
    r = cfg["patch_radius_px"]
    h, w = image_bgr.shape[:2]
    patch = image_bgr[max(y - r, 0) : min(y + r + 1, h), max(x - r, 0) : min(x + r + 1, w)]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    sat = hsv[:, 1]
    keep = hsv[sat >= cfg["min_saturation"]]
    if len(keep) == 0:
        raise ValueError(f"no saturated pixels near {point}: click on the coloured hold")
    return circular_median_hue(keep[:, 0]), float(np.median(keep[:, 1]))


def circular_median_hue(hues: np.ndarray) -> float:
    """Median of circular hue values: rotate so the biggest gap sits at the cut, then take the
    ordinary median (so reds at 2 and 177 give ~179.5/0.5, not 90)."""
    hs = np.sort(hues.astype(float))
    gaps = np.diff(np.concatenate([hs, [hs[0] + HUE_RANGE]]))
    cut = int(np.argmax(gaps)) + 1  # start just after the biggest gap
    rotated = np.concatenate([hs[cut:], hs[:cut] + HUE_RANGE])
    return float(np.median(rotated) % HUE_RANGE)


def box_hue(image_bgr: np.ndarray, box: Box, cfg: dict) -> float | None:
    """Median hue of the saturated pixels inside a box, or None if there are none."""
    x0, y0, x1, y1 = box
    hsv = cv2.cvtColor(image_bgr[max(y0, 0) : y1, max(x0, 0) : x1], cv2.COLOR_BGR2HSV)
    hsv = hsv.reshape(-1, 3)
    hsv = hsv[hsv[:, 1] >= cfg["min_saturation"]]
    return circular_median_hue(hsv[:, 0]) if len(hsv) else None


def filter_by_colour(
    image_bgr: np.ndarray, boxes: list[Box], ref_hue: float, cfg: dict
) -> list[Box]:
    """Keep boxes whose median hue is within `hue_tolerance` of ref_hue (wrap-around aware)."""
    kept = []
    for box in boxes:
        hue = box_hue(image_bgr, box, cfg)
        if hue is not None and hue_distance(hue, ref_hue) <= cfg["hue_tolerance"]:
            kept.append(box)
    return kept


def hue_mask(image_bgr: np.ndarray, ref_hue: float, cfg: dict) -> np.ndarray:
    """Binary mask of pixels within hue_tolerance of ref_hue and saturated enough."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    tol, min_sat = cfg["hue_tolerance"], cfg["min_saturation"]
    lo, hi = ref_hue - tol, ref_hue + tol
    if lo >= 0 and hi < HUE_RANGE:
        mask = cv2.inRange(hsv, (lo, min_sat, 0), (hi, 255, 255))
    else:  # range wraps around 0/179: two inRange calls
        mask = cv2.inRange(hsv, (lo % HUE_RANGE, min_sat, 0), (HUE_RANGE - 1, 255, 255))
        mask |= cv2.inRange(hsv, (0, min_sat, 0), (hi % HUE_RANGE, 255, 255))
    return mask


def colour_only_holds(image_bgr: np.ndarray, ref_hue: float, cfg: dict) -> list[Box]:
    """No detector: threshold the clicked hue, clean up, one box per blob inside the area limits
    (`min_area`/`max_area` are fractions of the image area)."""
    mask = hue_mask(image_bgr, ref_hue, cfg)
    k = cfg["morph_kernel_px"]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = image_bgr.shape[:2]
    lo, hi = cfg["min_area"] * w * h, cfg["max_area"] * w * h
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < lo:
            continue
        if area <= hi:
            x, y, bw, bh = cv2.boundingRect(c)
            boxes.append((x, y, x + bw, y + bh))
        else:  # several touching holds merged into one blob: try to split them
            boxes.extend(b for b, a in _split_blob(c, cfg) if lo <= a <= hi)
    return boxes


def _split_blob(contour: np.ndarray, cfg: dict) -> list[tuple[Box, int]]:
    """Split one merged blob with a distance-transform watershed. A blob with a single peak is
    just one big thing (not several holds), so it yields one region and is filtered by size.
    Returns (box, pixel_area) per piece."""
    x, y, bw, bh = cv2.boundingRect(contour)
    pad = 2
    blob = np.zeros((bh + 2 * pad, bw + 2 * pad), np.uint8)
    shifted = contour - np.array([x - pad, y - pad])
    cv2.drawContours(blob, [shifted], -1, 255, -1)
    dist = cv2.distanceTransform(blob, cv2.DIST_L2, 5)
    peaks = (dist > cfg["split_peak_ratio"] * dist.max()).astype(np.uint8)
    n, markers = cv2.connectedComponents(peaks)
    if n <= 2:  # background + one peak: nothing to split
        return [((x, y, x + bw, y + bh), int(cv2.contourArea(contour)))]
    markers = markers + 1
    markers[blob == 0] = 0  # background outside the blob stays unlabelled for watershed
    cv2.watershed(cv2.cvtColor(blob, cv2.COLOR_GRAY2BGR), markers)
    pieces = []
    for label in range(2, n + 1):
        ys, xs = np.where(markers == label)
        if len(xs):
            box = (
                x - pad + int(xs.min()),
                y - pad + int(ys.min()),
                x - pad + int(xs.max()) + 1,
                y - pad + int(ys.max()) + 1,
            )
            pieces.append((box, len(xs)))
    return pieces


_detect_cache: dict[str, list[Box]] = {}


def detect_holds(image_bgr: np.ndarray, cfg: dict) -> list[Box]:
    """Roboflow hold detector (network; setup-time only). Cached per image hash."""
    key = hashlib.sha256(image_bgr.tobytes()).hexdigest() + cfg["detector_model"]
    if key in _detect_cache:
        return _detect_cache[key]
    from inference_sdk import InferenceHTTPClient

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY is not set (put it in .env), or use colour-only mode")
    client = InferenceHTTPClient(api_url=cfg["api_url"], api_key=api_key)
    result = client.infer(image_bgr, model_id=cfg["detector_model"])
    h, w = image_bgr.shape[:2]
    boxes = []
    for p in result.get("predictions", []):  # centre x/y + width/height in pixels
        x0, y0 = round(p["x"] - p["width"] / 2), round(p["y"] - p["height"] / 2)
        boxes.append(
            (
                max(x0, 0),
                max(y0, 0),
                min(round(x0 + p["width"]), w),
                min(round(y0 + p["height"]), h),
            )
        )
    _detect_cache[key] = boxes
    return boxes


def build_holdmap(
    problem_id: str,
    colour: str,
    boxes: list[Box],
    image_size: tuple[int, int],
    mode: str,
    cfg: dict,
    start_boxes: set[Box] | None = None,
    top_boxes: set[Box] | None = None,
) -> HoldMap:
    """Number holds bottom (1) to top (N); mark the lowest `start_count` as start, highest as top.
    Coordinates are normalised 0-1, origin top-left, y down. `image_size` is (width, height).
    `start_boxes` / `top_boxes` (from the editor) replace the automatic start/top choice."""
    w, h = image_size
    ordered = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2, reverse=True)  # lowest first
    holds = []
    for i, (x0, y0, x1, y1) in enumerate(ordered, start=1):
        box = (x0, y0, x1, y1)
        if start_boxes or top_boxes:  # manual roles win over the automatic ones
            role = (
                "start"
                if box in (start_boxes or ())
                else "top"
                if box in (top_boxes or ())
                else "mid"
            )
        elif len(ordered) > 1 and i == len(ordered):
            role = "top"
        elif i <= cfg["start_count"]:
            role = "start"
        else:
            role = "mid"
        holds.append(
            Hold(
                id=i,
                center=((x0 + x1) / 2 / w, (y0 + y1) / 2 / h),
                bbox=(x0 / w, y0 / h, x1 / w, y1 / h),
                role=role,
            )
        )
    return HoldMap(
        problem_id=problem_id, colour=colour, holds=holds, image_size=image_size, mode=mode
    )


def holdmap_path(problem_id: str, root: str | Path | None = None) -> Path:
    return Path(root or REPO_ROOT / "data" / "holdmaps") / f"{problem_id}.json"


def save_holdmap(holdmap: HoldMap, root: str | Path | None = None) -> Path:
    path = holdmap_path(holdmap.problem_id, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "problem_id": holdmap.problem_id,
        "colour": holdmap.colour,
        "image_size": list(holdmap.image_size),
        "mode": holdmap.mode,
        "holds": [
            {"id": h.id, "center": list(h.center), "bbox": list(h.bbox), "role": h.role}
            for h in holdmap.holds
        ],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def load_holdmap(problem_id: str, root: str | Path | None = None) -> HoldMap:
    d = json.loads(holdmap_path(problem_id, root).read_text(encoding="utf-8"))
    holds = [
        Hold(id=h["id"], center=tuple(h["center"]), bbox=tuple(h["bbox"]), role=h["role"])
        for h in d["holds"]
    ]
    return HoldMap(
        problem_id=d["problem_id"],
        colour=d["colour"],
        holds=holds,
        image_size=tuple(d["image_size"]),
        mode=d["mode"],
    )


def draw_holds(image_bgr: np.ndarray, holdmap: HoldMap) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    out = image_bgr.copy()
    for hold in holdmap.holds:
        x0, y0, x1, y1 = (int(v * s) for v, s in zip(hold.bbox, (w, h, w, h), strict=True))
        colour = {"start": (0, 255, 0), "top": (0, 0, 255)}.get(hold.role, (255, 255, 0))
        cv2.rectangle(out, (x0, y0), (x1, y1), colour, 2)
        cv2.putText(
            out, str(hold.id), (x0, max(y0 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2
        )
    return out


def read_frame(video: str | Path, at_s: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, at_s * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"cannot read frame at {at_s}s from {video}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a hold map from one frame (no GUI)")
    parser.add_argument("--video", required=True)
    parser.add_argument("--at", type=float, default=1.0, help="seconds into the video")
    parser.add_argument("--click", type=int, nargs=2, required=True, metavar=("X", "Y"))
    parser.add_argument("--problem", required=True)
    parser.add_argument("--mode", choices=("colour", "detector"), default="colour")
    parser.add_argument("--out-image", help="write an annotated JPEG here")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_holds_config()
    frame = read_frame(args.video, args.at)
    hue, sat = sample_colour(frame, tuple(args.click), cfg)
    if args.mode == "detector":
        boxes = filter_by_colour(frame, detect_holds(frame, cfg), hue, cfg)
    else:
        boxes = colour_only_holds(frame, hue, cfg)
    h, w = frame.shape[:2]
    hm = build_holdmap(args.problem, colour_name(hue), boxes, (w, h), args.mode, cfg)
    log.info(
        "clicked hue %.0f sat %.0f (%s): %d holds, mode %s",
        hue,
        sat,
        hm.colour,
        len(hm.holds),
        hm.mode,
    )
    if not args.no_save:
        log.info("saved %s", save_holdmap(hm))
    if args.out_image:
        cv2.imwrite(args.out_image, draw_holds(frame, hm))


if __name__ == "__main__":
    main()
