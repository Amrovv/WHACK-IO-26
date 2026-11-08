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
    x, y = point
    r = cfg["patch_radius_px"]
    h, w = image_bgr.shape[:2]
    patch = image_bgr[max(y - r, 0) : min(y + r + 1, h), max(x - r, 0) : min(x + r + 1, w)]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    sat = hsv[:, 1]
    keep = hsv[sat >= cfg["min_saturation"]]
    if len(keep) == 0:
        pale = hsv[sat >= cfg["pale_min_saturation"]]
        if len(pale) >= cfg["pale_min_fraction"] * len(hsv):  # a pale hold (e.g. light teal)
            return circular_median_hue(pale[:, 0]), float(np.median(pale[:, 1]))
        value = hsv[:, 2]
        white = (sat <= cfg["white_max_saturation"]) & (value >= cfg["white_min_value"])
        if np.any(white):
            return -1.0, float(np.median(value[white]))
        raise ValueError(f"no coloured or white pixels near {point}: click on a hold")
    return circular_median_hue(keep[:, 0]), float(np.median(keep[:, 1]))


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
    strong = hsv[hsv[:, 1] >= cfg["min_saturation"]]
    if len(strong) >= cfg["pale_min_fraction"] * len(hsv):
        return circular_median_hue(strong[:, 0])
    pale = hsv[hsv[:, 1] >= cfg["pale_min_saturation"]]
    if len(pale) >= cfg["pale_min_fraction"] * len(hsv):
        return circular_median_hue(pale[:, 0])
    return circular_median_hue(strong[:, 0]) if len(strong) else None


def detected_hold_at(
    point: tuple[int, int], boxes: list[Box], polygons: dict[Box, PixelPolygon]
) -> Box | None:
    """The smallest detection under a click (its outline when it has one), or None."""
    x, y = point
    hits = [
        b
        for b in boxes
        if (point_in_polygon(point, polygons[b]) if b in polygons else b[0] <= x <= b[2])
        and b[1] <= y <= b[3]
    ]
    return min(hits, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])) if hits else None


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
    hit = detected_hold_at(point, boxes, polygons)
    return box_hue(image_bgr, hit, cfg, polygons.get(hit)) if hit is not None else None


# ---- Colour matching in CIE Lab --------------------------------------------------------------
# Hue alone throws away lightness and saturation, so pale yellows sink into the beige wall and
# reds, pinks and oranges (neighbouring hues) run together. Each hold's colour is instead its own
# dominant colour in Lab, and holds are compared with CIEDE2000 (a perceptual colour difference)
# with lightness weighted down, since the same hold is lighter or darker across the wall.
# Measured on A's 14 hold maps (click every route hold): F1 0.756 with hue -> 0.896.


def _lab(pixels_bgr: np.ndarray) -> np.ndarray:
    """(n, 3) BGR uint8 -> (n, 3) Lab floats (L 0-100, a and b about -128..127)."""
    px = pixels_bgr.reshape(-1, 1, 3).astype(np.float32) / 255.0
    return cv2.cvtColor(px, cv2.COLOR_BGR2LAB).reshape(-1, 3)


def _core_colour(lab: np.ndarray, cfg: dict) -> np.ndarray | None:
    """The median of the most colourful `colour_core_fraction` of the pixels: chalk, shadow and
    bits of wall are greyer than the hold, so they drop out."""
    if len(lab) == 0:
        return None
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    core = lab[chroma >= np.quantile(chroma, 1 - cfg["colour_core_fraction"])]
    return np.median(core, axis=0)


def hold_lab(
    image_bgr: np.ndarray, box: Box, cfg: dict, polygon: PixelPolygon | None = None
) -> np.ndarray | None:
    """A hold's own colour in Lab: the core colour of the pixels inside its outline (or, for a
    plain box, the ellipse inside it), less `colour_erode_px` at the edge, which is often wall.
    None for an empty region."""
    x0, y0, x1, y1 = box
    x0, y0 = max(x0, 0), max(y0, 0)
    crop = image_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    mask = np.zeros(crop.shape[:2], np.uint8)
    if polygon:
        cv2.fillPoly(mask, np.array([[(x - x0, y - y0) for x, y in polygon]], np.int32), 255)
    else:
        h, w = mask.shape
        cv2.ellipse(mask, (w // 2, h // 2), (max(w // 2, 1), max(h // 2, 1)), 0, 0, 360, 255, -1)
    k = 2 * cfg["colour_erode_px"] + 1
    eroded = cv2.erode(mask, np.ones((k, k), np.uint8))
    if eroded.any():  # a tiny hold keeps its uneroded pixels
        mask = eroded
    return _core_colour(_lab(crop[mask > 0]), cfg)


def patch_lab(image_bgr: np.ndarray, point: tuple[int, int], cfg: dict) -> np.ndarray | None:
    """The core colour around a click (`patch_radius_px`), for a hold the detector missed."""
    x, y = point
    r = cfg["patch_radius_px"]
    h, w = image_bgr.shape[:2]
    patch = image_bgr[max(y - r, 0) : min(y + r + 1, h), max(x - r, 0) : min(x + r + 1, w)]
    return _core_colour(_lab(patch), cfg) if patch.size else None


def ciede2000(ref: np.ndarray, labs: np.ndarray, k_l: float = 1.0) -> np.ndarray:
    """CIEDE2000 colour difference from one Lab colour to each row of `labs` (Sharma et al.
    2005). k_l > 1 makes lightness count less than chroma and hue."""
    labs = np.atleast_2d(labs).astype(float)
    l1, a1, b1 = (float(v) for v in ref)
    l2, a2, b2 = labs[:, 0], labs[:, 1], labs[:, 2]
    c_bar = (np.hypot(a1, b1) + np.hypot(a2, b2)) / 2
    g = 0.5 * (1 - np.sqrt(c_bar**7 / (c_bar**7 + 25.0**7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    both = c1p * c2p != 0
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(both, dh, 0.0)
    d_l, d_c = l2 - l1, c2p - c1p
    d_h = 2 * np.sqrt(c1p * c2p) * np.sin(np.radians(dh / 2))
    l_bar, cp_bar = (l1 + l2) / 2, (c1p + c2p) / 2
    h_sum = h1p + h2p
    h_bar = np.where(
        ~both,
        h_sum,
        np.where(
            np.abs(h1p - h2p) <= 180,
            h_sum / 2,
            np.where(h_sum < 360, (h_sum + 360) / 2, (h_sum - 360) / 2),
        ),
    )
    t = (
        1
        - 0.17 * np.cos(np.radians(h_bar - 30))
        + 0.24 * np.cos(np.radians(2 * h_bar))
        + 0.32 * np.cos(np.radians(3 * h_bar + 6))
        - 0.20 * np.cos(np.radians(4 * h_bar - 63))
    )
    d_theta = 30 * np.exp(-(((h_bar - 275) / 25) ** 2))
    r_c = 2 * np.sqrt(cp_bar**7 / (cp_bar**7 + 25.0**7))
    s_l = 1 + 0.015 * (l_bar - 50) ** 2 / np.sqrt(20 + (l_bar - 50) ** 2)
    s_c = 1 + 0.045 * cp_bar
    s_h = 1 + 0.015 * cp_bar * t
    r_t = -np.sin(np.radians(2 * d_theta)) * r_c
    tl, tc, th = d_l / (k_l * s_l), d_c / s_c, d_h / s_h
    return np.sqrt(tl**2 + tc**2 + th**2 + r_t * tc * th)


def match_colour(
    image_bgr: np.ndarray,
    boxes: list[Box],
    ref_lab: np.ndarray,
    cfg: dict,
    polygons: dict[Box, PixelPolygon] | None = None,
    ref_box: Box | None = None,
) -> list[Box]:
    """The detections the colour of `ref_lab` (the clicked hold's), in two steps:

    1. holds within `colour_match_delta` (CIEDE2000, lightness weight `colour_match_kl`) of the
       clicked hold;
    2. `colour_refine_rounds` times: holds within `colour_refine_delta` of the median colour of
       what matched. One hold may be shadowed or chalky; the route's typical colour isn't, so
       this picks up the route holds the first step missed and drops near misses.

    The clicked detection (`ref_box`) is always kept.
    """
    polygons = polygons or {}
    colours = {b: hold_lab(image_bgr, b, cfg, polygons.get(b)) for b in boxes}
    keys = [b for b, c in colours.items() if c is not None]
    if not keys:
        return []
    labs = np.array([colours[b] for b in keys])
    k_l = cfg["colour_match_kl"]

    def within(ref: np.ndarray, delta: float) -> list[Box]:
        kept = [b for b, d in zip(keys, ciede2000(ref, labs, k_l), strict=True) if d <= delta]
        if ref_box is not None and ref_box in colours and ref_box not in kept:
            kept.append(ref_box)
        return kept

    kept = within(ref_lab, cfg["colour_match_delta"])
    for _ in range(cfg["colour_refine_rounds"]):
        if not kept:
            break
        typical = np.median(np.array([colours[b] for b in kept]), axis=0)
        kept = within(typical, cfg["colour_refine_delta"])
    return kept


def filter_by_colour(
    image_bgr: np.ndarray,
    boxes: list[Box],
    ref_hue: float,
    cfg: dict,
    polygons: dict[Box, PixelPolygon] | None = None,
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
        hue = box_hue(image_bgr, box, cfg, (polygons or {}).get(box))
        if hue is not None and hue_distance(hue, ref_hue) <= cfg["hue_tolerance"]:
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
