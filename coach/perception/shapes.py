"""Polygon helpers for diagonal holds (an upright box around a slanted slab is mostly empty wall).

A hold keeps its axis-aligned `bbox` in the shared HoldMap contract; its polygon is an extra,
perception-only detail stored beside it in the map file. Points are (x, y) tuples; the functions
work in whatever unit you give them (the tracker uses normalised 0-1 image coordinates).
"""

from __future__ import annotations

from collections.abc import Sequence

Point = tuple[float, float]
Polygon = Sequence[Point]

MIN_AREA = 1e-9  # a polygon with less area than this is a line or a point, not a hold


def polygon_area(poly: Polygon) -> float:
    """Unsigned area (shoelace formula)."""
    total = 0.0
    for (x0, y0), (x1, y1) in zip(poly, [*poly[1:], poly[0]], strict=True):
        total += x0 * y1 - x1 * y0
    return abs(total) / 2


def is_valid_polygon(poly: Polygon) -> bool:
    return len(poly) >= 3 and polygon_area(poly) > MIN_AREA


def polygon_bounds(poly: Polygon) -> tuple[float, float, float, float]:
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def polygon_centroid(poly: Polygon) -> Point:
    """Area centroid; falls back to the vertex mean for a degenerate polygon."""
    area2 = 0.0
    cx = cy = 0.0
    for (x0, y0), (x1, y1) in zip(poly, [*poly[1:], poly[0]], strict=True):
        cross = x0 * y1 - x1 * y0
        area2 += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    if abs(area2) < MIN_AREA:
        return sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly)
    return cx / (3 * area2), cy / (3 * area2)


def point_in_polygon(p: Point, poly: Polygon) -> bool:
    """Ray casting: is the point inside (points exactly on an edge may go either way)?"""
    x, y = p
    inside = False
    for (x0, y0), (x1, y1) in zip(poly, [*poly[1:], poly[0]], strict=True):
        if (y0 > y) != (y1 > y) and x < (x1 - x0) * (y - y0) / (y1 - y0) + x0:
            inside = not inside
    return inside


def polygon_intersects_rect(poly: Polygon, rect: tuple[float, float, float, float]) -> bool:
    """Return whether a polygon touches an axis-aligned rectangle, including boundaries."""
    x0, y0, x1, y1 = rect
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    if any(x0 <= x <= x1 and y0 <= y <= y1 for x, y in poly):
        return True
    if any(point_in_polygon(corner, poly) for corner in corners):
        return True

    def orientation(a: Point, b: Point, c: Point) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a: Point, b: Point, p: Point) -> bool:
        return (
            min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])
            and orientation(a, b, p) == 0
        )

    def crosses(a: Point, b: Point, c: Point, d: Point) -> bool:
        oa, ob = orientation(a, b, c), orientation(a, b, d)
        oc, od = orientation(c, d, a), orientation(c, d, b)
        return (
            (oa == 0 and on_segment(a, b, c))
            or (ob == 0 and on_segment(a, b, d))
            or (oc == 0 and on_segment(c, d, a))
            or (od == 0 and on_segment(c, d, b))
            or (oa < 0 < ob or ob < 0 < oa)
            and (oc < 0 < od or od < 0 < oc)
        )

    rect_edges = list(zip(corners, corners[1:] + corners[:1], strict=True))
    poly_edges = list(zip(poly, poly[1:] + poly[:1], strict=True))
    return any(crosses(a, b, c, d) for a, b in poly_edges for c, d in rect_edges)


def box_intersects_rect(
    box: tuple[float, float, float, float], rect: tuple[float, float, float, float]
) -> bool:
    """Return whether two axis-aligned boxes overlap or touch."""
    return box[0] <= rect[2] and box[2] >= rect[0] and box[1] <= rect[3] and box[3] >= rect[1]


def _distance_to_segment(p: Point, a: Point, b: Point) -> float:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    t = 0.0 if length2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / length2))
    return ((p[0] - (ax + t * dx)) ** 2 + (p[1] - (ay + t * dy)) ** 2) ** 0.5


def distance_to_polygon(p: Point, poly: Polygon) -> float:
    """0 if the point is inside the polygon, otherwise the distance to its nearest edge."""
    if point_in_polygon(p, poly):
        return 0.0
    return min(
        _distance_to_segment(p, a, b) for a, b in zip(poly, [*poly[1:], poly[0]], strict=True)
    )
