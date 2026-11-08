"""Build a HoldMap from one frame of the empty wall.

Two ways to find the route's holds:
  * detector mode: a Roboflow detector finds all holds, then only boxes the same colour as the
    clicked hold are kept (`detect_holds` + `filter_by_colour`)
  * colour-only mode: threshold the clicked hue directly (`colour_only_holds`); explicit offline
    mode for the paper wall and fallback when the detector is unavailable

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
from dotenv import load_dotenv

from coach.contracts import Hold, HoldMap
from coach.perception.config import REPO_ROOT, load_config
from coach.perception.shapes import point_in_polygon, polygon_centroid

log = logging.getLogger(__name__)

Box = tuple[int, int, int, int]  # x0, y0, x1, y1 in pixels
PixelPolygon = list[tuple[int, int]]  # a diagonal hold's outline, in pixels
Shapes = dict[int, list[tuple[float, float]]]  # hold id -> normalised polygon
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
    if hue < 0:
        return "white"
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
    """Median (hue, saturation) around a clicked point, ignoring low-saturation pixels.

    A wider configured patch helps when a hand or shadow covers part of a hold. Hue uses the
    circular median.
    """
    hue, saturation, _ = sample_colour_hsv(image_bgr, point, cfg)
    return hue, saturation


def sample_colour_hsv(
    image_bgr: np.ndarray, point: tuple[int, int], cfg: dict
) -> tuple[float, float, float]:
    """HSV reference at a click with robust value filtering; adds value to legacy sampling."""
    x, y = point
    r = cfg["patch_radius_px"]
    h, w = image_bgr.shape[:2]
    patch = image_bgr[max(y - r, 0) : min(y + r + 1, h), max(x - r, 0) : min(x + r + 1, w)]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    sat = hsv[:, 1]
    value = hsv[:, 2]
    usable_value = (value >= cfg["colour_min_value"]) & (value <= cfg["colour_max_value"])
    keep = hsv[(sat >= cfg["min_saturation"]) & usable_value]
    if len(keep) == 0:
        pale = hsv[(sat >= cfg["pale_min_saturation"]) & usable_value]
        if len(pale) >= cfg["pale_min_fraction"] * len(hsv):  # a pale hold (e.g. light teal)
            return (
                circular_median_hue(pale[:, 0]),
                float(np.median(pale[:, 1])),
                float(np.median(pale[:, 2])),
            )
        white = (sat <= cfg["white_max_saturation"]) & (value >= cfg["white_min_value"])
        if np.any(white):
            return -1.0, float(np.median(sat[white])), float(np.median(value[white]))
        raise ValueError(f"no coloured or white pixels near {point}: click on a hold")
    return (
        circular_median_hue(keep[:, 0]),
        float(np.median(keep[:, 1])),
        float(np.median(keep[:, 2])),
    )


def circular_median_hue(hues: np.ndarray) -> float:
    """Median of circular hue values: rotate so the biggest gap sits at the cut, then take the
    ordinary median (so reds at 2 and 177 give ~179.5/0.5, not 90)."""
    hs = np.sort(hues.astype(float))
    gaps = np.diff(np.concatenate([hs, [hs[0] + HUE_RANGE]]))
    cut = int(np.argmax(gaps)) + 1  # start just after the biggest gap
    rotated = np.concatenate([hs[cut:], hs[:cut] + HUE_RANGE])
    return float(np.median(rotated) % HUE_RANGE)


def box_hue(
    image_bgr: np.ndarray, box: Box, cfg: dict, polygon: PixelPolygon | None = None
) -> float | None:
    """Median hue in the hold pixels inside a box or supplied polygon.

    Pixels count when they are saturated enough (`min_saturation`). A pale hold (light teal,
    median saturation ~40) has too few of those, so when they are under `pale_min_fraction` of the
    hold the lower `pale_min_saturation` is used instead; if even that is not met by enough of
    the hold, the hold has no usable colour (None).
    """
    measured = box_hsv(image_bgr, box, cfg, polygon)
    return measured[0] if measured is not None else None


def box_hsv(
    image_bgr: np.ndarray, box: Box, cfg: dict, polygon: PixelPolygon | None = None
) -> tuple[float, float, float] | None:
    """Robust median HSV from hold pixels; polygons exclude wall pixels in the bounding box."""
    x0, y0, x1, y1 = box
    x0, y0 = max(x0, 0), max(y0, 0)
    crop = image_bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hsv = hsv.reshape(-1, 3)
    if polygon:
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        pts = np.array([[(x - x0, y - y0) for x, y in polygon]], dtype=np.int32)
        cv2.fillPoly(mask, pts, 255)
        hsv = hsv[mask.reshape(-1) > 0]
    if len(hsv) == 0:
        return None
    valid = hsv[
        (hsv[:, 1] >= cfg["pale_min_saturation"])
        & (hsv[:, 2] >= cfg["colour_min_value"])
        & (hsv[:, 2] <= cfg["colour_max_value"])
    ]
    strong = valid[valid[:, 1] >= cfg["min_saturation"]]
    if len(strong) >= cfg["pale_min_fraction"] * len(hsv):
        valid = strong
    pale = hsv[hsv[:, 1] >= cfg["pale_min_saturation"]]
    if not len(valid) and len(pale) >= cfg["pale_min_fraction"] * len(hsv):
        valid = pale
    if len(valid) == 0:
        return None
    return (
        circular_median_hue(valid[:, 0]),
        float(np.median(valid[:, 1])),
        float(np.median(valid[:, 2])),
    )


def detected_hold_hue(
    image_bgr: np.ndarray,
    point: tuple[int, int],
    boxes: list[Box],
    polygons: dict[Box, PixelPolygon],
    cfg: dict,
) -> float | None:
    """Hue of the detected hold under a click, measured on the whole hold (its polygon pixels).

    A small click patch is unreliable on pale or chalky holds (saturation can drop to the wall's
    level); the hold's own pixels are not. The smallest detection containing the point is used.
    None if no detection is under the click or it has no usable hue (white, grey, black).
    """
    x, y = point
    hits = [
        b
        for b in boxes
        if (point_in_polygon(point, polygons[b]) if b in polygons else b[0] <= x <= b[2])
        and b[1] <= y <= b[3]
    ]
    if not hits:
        return None
    hit = min(hits, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    colour = box_hsv(image_bgr, hit, cfg, polygons.get(hit))
    return colour[0] if colour is not None else None


def detected_hold_colour(image_bgr, point, boxes, polygons, cfg):
    """Median HSV of the detected hold under the click; white retains sentinel hue -1."""
    x, y = point
    hits = [
        b
        for b in boxes
        if (point_in_polygon(point, polygons[b]) if b in polygons else b[0] <= x <= b[2])
        and b[1] <= y <= b[3]
    ]
    if not hits:
        return None
    hit = min(hits, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    color = box_hsv(image_bgr, hit, cfg, polygons.get(hit))
    if color is not None:
        return color
    region = image_bgr[max(hit[1], 0) : hit[3], max(hit[0], 0) : hit[2]]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    white = (hsv[:, 1] <= cfg["white_max_saturation"]) & (hsv[:, 2] >= cfg["white_min_value"])
    if len(hsv) and white.mean() >= cfg["white_min_fraction"]:
        return (-1.0, float(np.median(hsv[white, 1])), float(np.median(hsv[white, 2])))
    return None


def filter_by_colour(
    image_bgr: np.ndarray,
    boxes: list[Box],
    ref_hue: float,
    cfg: dict,
    polygons: dict[Box, PixelPolygon] | None = None,
    ref_saturation: float | None = None,
    ref_value: float | None = None,
) -> list[Box]:
    """Keep boxes whose median hue is within `hue_tolerance` of ref_hue (wrap-around aware).
    A white reference (hue < 0) keeps boxes where at least `white_min_fraction` of the hold
    pixels (polygon if given, else box) are white."""
    kept = []
    for box in boxes:
        if ref_hue < 0:
            x0, y0, x1, y1 = box
            region = image_bgr[max(y0, 0) : y1, max(x0, 0) : x1]
            if polygons and box in polygons:
                mask = np.zeros(region.shape[:2], np.uint8)
                pts = np.array(
                    [[(x - max(x0, 0), y - max(y0, 0)) for x, y in polygons[box]]], np.int32
                )
                cv2.fillPoly(mask, pts, 255)
                pixels = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)[mask > 0]
            else:
                pixels = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).reshape(-1, 3)
            white = (pixels[:, 1] <= cfg["white_max_saturation"]) & (
                pixels[:, 2] >= cfg["white_min_value"]
            )
            if len(pixels) and white.mean() >= cfg["white_min_fraction"]:
                kept.append(box)
            continue
        measured = box_hsv(image_bgr, box, cfg, (polygons or {}).get(box))
        if measured is None:
            continue
        hue, sat, val = measured
        hue_ok = hue_distance(hue, ref_hue) <= cfg["hue_tolerance"]
        if ref_saturation is not None and ref_value is not None:
            hue_ok &= sat >= ref_saturation - cfg["colour_saturation_tolerance"]
            hue_ok &= abs(val - ref_value) <= cfg["colour_value_tolerance"]
        if hue_ok:
            kept.append(box)
    return kept


def hue_mask(image_bgr: np.ndarray, ref_hue: float, cfg: dict) -> np.ndarray:
    """Binary mask of pixels within hue_tolerance of ref_hue and saturated enough."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    if ref_hue < 0:
        return cv2.inRange(
            hsv, (0, 0, cfg["white_min_value"]), (HUE_RANGE - 1, cfg["white_max_saturation"], 255)
        )
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


_detect_cache: dict[str, list[tuple[Box, PixelPolygon | None]]] = {}


def simplify_polygon(points: list[tuple[float, float]], cfg: dict) -> PixelPolygon | None:
    """Reduce a detector outline (often 40+ points) to a few corners that are easy to edit.
    Tolerance = `polygon_epsilon` x the outline's perimeter. None if it collapses to < 3 corners."""
    contour = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    eps = cfg["polygon_epsilon"] * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2)
    if len(approx) < 3:
        return None
    return [(round(float(x)), round(float(y))) for x, y in approx]


def _detections(image_bgr: np.ndarray, cfg: dict) -> list[tuple[Box, PixelPolygon | None]]:
    """Roboflow call (network; setup-time only). One (box, outline) per hold; the outline is
    None for a plain detection model that returns no `points`. Cached per image hash."""
    key = hashlib.sha256(image_bgr.tobytes()).hexdigest() + cfg["detector_model"]
    if key in _detect_cache:
        return _detect_cache[key]
    from inference_sdk import InferenceHTTPClient

    load_dotenv(REPO_ROOT / ".env")  # the key lives in .env; the shell does not export it
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY is not set (put it in .env), or use colour-only mode")
    client = InferenceHTTPClient(api_url=cfg["api_url"], api_key=api_key)
    result = client.infer(image_bgr, model_id=cfg["detector_model"])
    h, w = image_bgr.shape[:2]
    found: list[tuple[Box, PixelPolygon | None]] = []
    for p in result.get("predictions", []):  # centre x/y + width/height in pixels
        x0, y0 = round(p["x"] - p["width"] / 2), round(p["y"] - p["height"] / 2)
        box = (
            max(x0, 0),
            max(y0, 0),
            min(round(x0 + p["width"]), w),
            min(round(y0 + p["height"]), h),
        )
        outline = [(min(max(pt["x"], 0), w), min(max(pt["y"], 0), h)) for pt in p.get("points", [])]
        found.append((box, simplify_polygon(outline, cfg) if len(outline) >= 3 else None))
    _detect_cache[key] = found
    return found


def detect_holds(image_bgr: np.ndarray, cfg: dict) -> list[Box]:
    """Roboflow hold detector: the bounding boxes."""
    return [box for box, _ in _detections(image_bgr, cfg)]


def detect_hold_polygons(image_bgr: np.ndarray, cfg: dict) -> dict[Box, PixelPolygon]:
    """Outlines for the detected holds (segmentation models only), keyed by the box that
    `detect_holds` returns for the same hold."""
    return {box: poly for box, poly in _detections(image_bgr, cfg) if poly is not None}


def build_holdmap_with_shapes(
    problem_id: str,
    colour: str,
    boxes: list[Box],
    image_size: tuple[int, int],
    mode: str,
    cfg: dict,
    start_boxes: set[Box] | None = None,
    top_boxes: set[Box] | None = None,
    polygons: dict[Box, PixelPolygon] | None = None,
) -> tuple[HoldMap, Shapes]:
    """Number holds bottom (1) to top (N); mark the lowest `start_count` as start, highest as top.
    Coordinates are normalised 0-1, origin top-left, y down. `image_size` is (width, height).
    `start_boxes` / `top_boxes` (from the editor) replace the automatic start/top choice.

    A diagonal hold is given as its axis-aligned box (its bounding box, which is what the shared
    HoldMap contract carries) plus an entry in `polygons[box]` with its real outline. The result
    is the HoldMap and a `Shapes` dict {hold id: normalised polygon} for the polygon holds only.
    """
    w, h = image_size
    polygons = polygons or {}
    ordered = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2, reverse=True)  # lowest first
    holds = []
    shapes: Shapes = {}
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
        center = ((x0 + x1) / 2 / w, (y0 + y1) / 2 / h)
        if box in polygons:
            shapes[i] = [(px / w, py / h) for px, py in polygons[box]]
            center = polygon_centroid(shapes[i])
        holds.append(Hold(id=i, center=center, bbox=(x0 / w, y0 / h, x1 / w, y1 / h), role=role))
    holdmap = HoldMap(
        problem_id=problem_id, colour=colour, holds=holds, image_size=image_size, mode=mode
    )
    return holdmap, shapes


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
    """Boxes only (see `build_holdmap_with_shapes` for diagonal holds)."""
    return build_holdmap_with_shapes(
        problem_id, colour, boxes, image_size, mode, cfg, start_boxes, top_boxes
    )[0]


def holdmap_path(problem_id: str, root: str | Path | None = None) -> Path:
    return Path(root or REPO_ROOT / "data" / "holdmaps") / f"{problem_id}.json"


def save_holdmap(
    holdmap: HoldMap, root: str | Path | None = None, shapes: Shapes | None = None
) -> Path:
    """Write the map as JSON. `shapes` adds an optional "polygon" to the holds that have one;
    the shared fields (id, center, bbox, role) are the same as for a map without shapes."""
    path = holdmap_path(holdmap.problem_id, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    shapes = shapes or {}
    holds = []
    for h in holdmap.holds:
        entry = {"id": h.id, "center": list(h.center), "bbox": list(h.bbox), "role": h.role}
        if h.id in shapes:
            entry["polygon"] = [list(pt) for pt in shapes[h.id]]
        holds.append(entry)
    data = {
        "problem_id": holdmap.problem_id,
        "colour": holdmap.colour,
        "image_size": list(holdmap.image_size),
        "mode": holdmap.mode,
        "holds": holds,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def load_holdmap(problem_id: str, root: str | Path | None = None) -> HoldMap:
    return load_holdmap_with_shapes(problem_id, root)[0]


def load_hold_shapes(problem_id: str, root: str | Path | None = None) -> Shapes:
    """Polygons of the diagonal holds (empty for a map made of plain boxes)."""
    return load_holdmap_with_shapes(problem_id, root)[1]


def load_holdmap_with_shapes(
    problem_id: str, root: str | Path | None = None
) -> tuple[HoldMap, Shapes]:
    d = json.loads(holdmap_path(problem_id, root).read_text(encoding="utf-8"))
    holds = [
        Hold(id=h["id"], center=tuple(h["center"]), bbox=tuple(h["bbox"]), role=h["role"])
        for h in d["holds"]
    ]
    shapes: Shapes = {
        h["id"]: [tuple(pt) for pt in h["polygon"]] for h in d["holds"] if "polygon" in h
    }
    holdmap = HoldMap(
        problem_id=d["problem_id"],
        colour=d["colour"],
        holds=holds,
        image_size=tuple(d["image_size"]),
        mode=d["mode"],
    )
    return holdmap, shapes


def draw_holds(image_bgr: np.ndarray, holdmap: HoldMap, shapes: Shapes | None = None) -> np.ndarray:
    """Draw every hold with its id: its outline if it has a polygon, otherwise its box."""
    h, w = image_bgr.shape[:2]
    shapes = shapes or {}
    out = image_bgr.copy()
    for hold in holdmap.holds:
        colour = {"start": (0, 255, 0), "top": (0, 0, 255)}.get(hold.role, (255, 255, 0))
        if hold.id in shapes:
            pts = np.array([(round(x * w), round(y * h)) for x, y in shapes[hold.id]], np.int32)
            cv2.polylines(out, [pts], True, colour, 2)
            x0, y0 = int(pts[:, 0].min()), int(pts[:, 1].min())
        else:
            x0, y0, x1, y1 = (int(v * k) for v, k in zip(hold.bbox, (w, h, w, h), strict=True))
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
    outlines: dict[Box, PixelPolygon] = {}
    if args.mode == "detector":
        boxes = filter_by_colour(frame, detect_holds(frame, cfg), hue, cfg)
        found = detect_hold_polygons(frame, cfg)
        outlines = {b: found[b] for b in boxes if b in found}
    else:
        boxes = colour_only_holds(frame, hue, cfg)
    h, w = frame.shape[:2]
    hm, shapes = build_holdmap_with_shapes(
        args.problem, colour_name(hue), boxes, (w, h), args.mode, cfg, polygons=outlines
    )
    log.info(
        "clicked hue %.0f sat %.0f (%s): %d holds, mode %s",
        hue,
        sat,
        hm.colour,
        len(hm.holds),
        hm.mode,
    )
    if not args.no_save:
        log.info("saved %s", save_holdmap(hm, shapes=shapes))
    if args.out_image:
        cv2.imwrite(args.out_image, draw_holds(frame, hm, shapes))


if __name__ == "__main__":
    main()
