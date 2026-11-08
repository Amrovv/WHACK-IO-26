"""Plain OpenCV hold-map editor (setup tool, never part of the live loop).

    uv run python -m coach.perception.holdmap_editor --video data/footage/clip.mp4 --at 1.0 \
        --problem pink-v3 [--colour-name pink]
    uv run python -m coach.perception.holdmap_editor --image wall.jpg --problem paper-wall
    uv run python -m coach.perception.holdmap_editor --video clip.mp4 --problem pink-v3 --edit

Click a route hold to filter Roboflow detections by its colour; detected polygons are retained.
If Roboflow is unavailable, manual placement remains available.
Use a frame from the SAME video as the climb (first seconds, wall empty) so colours match.
`--edit` loads the saved map for --problem and lets you fix it (polygons included).

Mouse:  click a hold to toggle it in/out · drag = erase · right-click a hold = delete it
Keys:   r = toggle drag between erase and plain-box placement ·
        p = polygon: click the corners of a diagonal hold, Enter = close, Esc = cancel,
            right-click = remove the last corner ·
        u = undo · s + click = mark start · t + click = mark top · c = re-pick colour ·
        Enter = save · Esc = quit
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from coach.contracts import HoldMap
from coach.perception.holds import (
    Box,
    PixelPolygon,
    Shapes,
    build_holdmap_with_shapes,
    colour_name,
    detect_hold_polygons,
    detect_holds,
    detected_hold_colour,
    filter_by_colour,
    load_holdmap_with_shapes,
    load_holds_config,
    read_frame,
    sample_colour_hsv,
    save_holdmap,
)
from coach.perception.shapes import (
    box_intersects_rect,
    is_valid_polygon,
    point_in_polygon,
    polygon_bounds,
    polygon_intersects_rect,
)

log = logging.getLogger(__name__)

KEY_ENTER, KEY_ESC = 13, 27
HELP_LINES = (
    "click=toggle  right-click=delete  drag=erase  r=toggle erase/plain-box placement  u=undo",
    "p=polygon (click corners, Enter=close, Esc=cancel)  s/t+click=start/top",
    "c=colour  Enter=save  Esc=quit",
)
HELP_BAR_PX = 26
STATUS_ROWS = 2  # the status line wraps to at most this many rows
DISPLAY_H, DISPLAY_W = 800, 1400  # the window is sized to fit this; text and lines are drawn for it


def ui_scale(image_h: int) -> float:
    """Factor that makes text and lines look the same size in the window for any image size: a
    tall 1080x1920 photo is shown shrunk to fit the screen, so its text is drawn bigger."""
    return max(1.0, image_h / DISPLAY_H)


def _px(base: float, scale: float) -> int:
    return max(1, round(base * scale))


def wrap_text(text: str, max_w: int, font_scale: float, thickness: int) -> list[str]:
    """Split text into lines that each fit `max_w` pixels (a narrow portrait image would
    otherwise cut the help text off at the right edge)."""
    lines: list[str] = []
    cur = ""
    for word in text.split(" "):
        trial = f"{cur} {word}" if cur else word
        width = cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0]
        if cur and width > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return [*lines, cur] if cur else lines


MIN_BOX_PX = 3
POLY_COLOUR = (0, 200, 255)  # BGR: the polygon being drawn


def _contains(box: Box, p: tuple[int, int]) -> bool:
    return box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3]


def _area(box: Box) -> int:
    return (box[2] - box[0]) * (box[3] - box[1])


class EditorState:
    """All editing logic, with no window: easy to test with synthetic images and clicks."""

    def __init__(self, image: np.ndarray, cfg: dict, colour: str | None = None) -> None:
        self.image, self.cfg = image, cfg
        self.colour_override = colour
        self.ref_hue: float | None = None
        self.ref_saturation: float | None = None
        self.ref_value: float | None = None
        self.detector_used = False
        self.auto: list[Box] = []
        self.excluded: set[Box] = set()  # auto boxes switched off (shown dim, click to restore)
        self.deleted: set[Box] = set()  # auto boxes removed for good (hidden; undo restores)
        self._undo: list[tuple] = []
        self.manual: list[Box] = []
        self.polygons: dict[Box, PixelPolygon] = {}  # a manual box's real outline (diagonal hold)
        self.start: set[Box] = set()
        self.top: set[Box] = set()
        self.message = "click a hold of the route colour; press r to place a box"

    # -- colour selection ---------------------------------------------------------------------
    def pick_colour(self, point: tuple[int, int]) -> None:
        hue = None
        try:  # the colour of the whole detected hold under the click (robust on pale holds)
            colour = detected_hold_colour(
                self.image,
                point,
                detect_holds(self.image, self.cfg),
                detect_hold_polygons(self.image, self.cfg),
                self.cfg,
            )
        except Exception as e:  # no key / network: the small patch below still works
            log.info("Roboflow not available to read the clicked hold's colour: %s", e)
            colour = None
        hue = colour[0] if colour is not None else None
        if hue is not None:
            self.ref_hue = hue
            self.ref_saturation, self.ref_value = colour[1], colour[2]
        else:
            try:
                self.ref_hue, self.ref_saturation, self.ref_value = sample_colour_hsv(
                    self.image, point, self.cfg
                )
            except ValueError as e:
                self.message = f"{e}; manual placement is available with r"
                return
        self.detector_used = False
        self.recompute()

    def clear_colour(self) -> None:
        self.ref_hue = None
        self.detector_used = False
        self.message = "click a hold to pick a route colour"

    def recompute(self) -> None:
        """Re-run Roboflow detection for the current colour."""
        assert self.ref_hue is not None
        for box in self.auto:
            if box not in self.manual:
                self.polygons.pop(box, None)
        self.auto = []
        self.excluded.clear()
        self.deleted.clear()
        self._undo.clear()
        try:
            found = detect_holds(self.image, self.cfg)
            outlines = detect_hold_polygons(self.image, self.cfg)
            self.auto = filter_by_colour(
                self.image,
                found,
                self.ref_hue,
                self.cfg,
                outlines,
                self.ref_saturation,
                self.ref_value,
            )
            self.polygons.update({box: outlines[box] for box in self.auto if box in outlines})
            self.detector_used = True
        except Exception as e:
            log.warning("Roboflow detection failed: %s", e)
            self.detector_used = False
            self._prune_marks()
            self.message = f"Roboflow unavailable: {e}; use r to place boxes or p for polygons"
            return
        self._prune_marks()
        self.message = f"Roboflow found {len(self.auto)} holds"

    def _prune_marks(self) -> None:
        keep = set(self.included())
        self.start &= keep
        self.top &= keep

    def load_existing(self, holdmap: HoldMap, shapes: Shapes) -> None:
        """Start from a saved map (every hold becomes an editable manual hold)."""
        h, w = self.image.shape[:2]
        self.ref_hue = None
        self.ref_saturation = self.ref_value = None
        self.detector_used = False
        self.manual, self.polygons = [], {}
        self.start, self.top = set(), set()
        for hold in holdmap.holds:
            x0, y0, x1, y1 = hold.bbox
            box = (round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h))
            self.manual.append(box)
            if hold.id in shapes:
                self.polygons[box] = [(round(x * w), round(y * h)) for x, y in shapes[hold.id]]
            if hold.role == "start":
                self.start.add(box)
            elif hold.role == "top":
                self.top.add(box)
        self.colour_override = self.colour_override or holdmap.colour
        self._undo.clear()
        self.message = f"loaded {len(self.manual)} holds from the saved map"

    # -- editing ------------------------------------------------------------------------------
    def included(self) -> list[Box]:
        return [
            b for b in self.auto if b not in self.excluded and b not in self.deleted
        ] + self.manual

    def visible_auto(self) -> list[Box]:
        return [b for b in self.auto if b not in self.deleted]

    def _hit(self, box: Box, point: tuple[int, int]) -> bool:
        """Is the point on the hold? A polygon hold is hit by its outline, not its bounding box."""
        if box in self.polygons:
            return point_in_polygon(point, self.polygons[box])
        return _contains(box, point)

    def _shape_area(self, box: Box) -> int:
        return _area(box)

    def erase_rect(self, p0: tuple[int, int], p1: tuple[int, int]) -> None:
        """Remove every auto/manual hold touching a rectangle in a single undo step."""
        rect = (min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1]))
        hits = [
            box
            for box in self.included()
            if box in self.polygons and polygon_intersects_rect(self.polygons[box], rect)
        ]
        hits.extend(
            box
            for box in self.included()
            if box not in self.polygons and box_intersects_rect(box, rect)
        )
        if not hits:
            self.message = f"erase: no holds touched ({len(self.included())} kept)"
            return
        self._snapshot()
        for box in hits:
            if box in self.manual:
                self.manual.remove(box)
                self.polygons.pop(box, None)
            else:
                self.deleted.add(box)
            self.start.discard(box)
            self.top.discard(box)
        self.message = f"erased {len(hits)} holds ({len(self.included())} kept)"

    def box_at(self, point: tuple[int, int]) -> Box | None:
        """The smallest hold (auto or manual) under the point."""
        hits = [b for b in [*self.visible_auto(), *self.manual] if self._hit(b, point)]
        return min(hits, key=self._shape_area) if hits else None

    def _snapshot(self) -> None:
        self._undo.append(
            (
                set(self.excluded),
                set(self.deleted),
                list(self.manual),
                set(self.start),
                set(self.top),
                dict(self.polygons),
            )
        )

    def undo(self) -> None:
        if not self._undo:
            self.message = "nothing to undo"
            return
        (
            self.excluded,
            self.deleted,
            self.manual,
            self.start,
            self.top,
            self.polygons,
        ) = self._undo.pop()
        self.message = "undone"

    def delete_at(self, point: tuple[int, int]) -> None:
        """Remove the hold under the point for good (drawn: dropped, detected: hidden)."""
        box = self.box_at(point)
        if box is None:
            self.message = "no box here to delete"
            return
        self._snapshot()
        if box in self.manual:
            self.manual.remove(box)
            self.polygons.pop(box, None)
        else:
            self.deleted.add(box)
        self.excluded.discard(box)
        self.start.discard(box)
        self.top.discard(box)
        self.message = f"deleted a hold ({len(self.included())} left)"

    def toggle_at(self, point: tuple[int, int]) -> None:
        box = self.box_at(point)
        if box is None:
            self.message = "no box here (drag to draw one)"
            return
        self._snapshot()
        if box in self.manual:
            self.manual.remove(box)
            self.polygons.pop(box, None)
            self.start.discard(box)
            self.top.discard(box)
        elif box in self.excluded:
            self.excluded.discard(box)
        else:
            self.excluded.add(box)
            self.start.discard(box)
            self.top.discard(box)

    def _clamp(self, p: tuple[int, int]) -> tuple[int, int]:
        h, w = self.image.shape[:2]
        return min(max(p[0], 0), w - 1), min(max(p[1], 0), h - 1)

    def add_box(self, p0: tuple[int, int], p1: tuple[int, int]) -> None:
        (x0, y0), (x1, y1) = self._clamp(p0), self._clamp(p1)
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        if x1 - x0 < MIN_BOX_PX or y1 - y0 < MIN_BOX_PX:
            return
        self._snapshot()
        self.manual.append((x0, y0, x1, y1))

    def add_polygon(self, points: list[tuple[int, int]]) -> bool:
        """Add a diagonal hold from the corners the user clicked. Returns False if the outline
        is not a proper shape (fewer than 3 corners, or all in a line)."""
        pts = [self._clamp(p) for p in points]
        if not is_valid_polygon(pts):
            self.message = "a polygon needs at least 3 corners that are not in a line"
            return False
        x0, y0, x1, y1 = (round(v) for v in polygon_bounds(pts))
        box = (x0, y0, x1, y1)
        if box in self.manual:  # same bounds as an existing hold: nudge so the two stay distinct
            box = (x0, y0, x1 + 1, y1 + 1)
        self._snapshot()
        self.manual.append(box)
        self.polygons[box] = pts
        self.message = f"added a polygon hold ({len(self.included())} holds)"
        return True

    def mark(self, role: str, point: tuple[int, int]) -> None:
        box = self.box_at(point)
        if box is None or box not in self.included():
            self.message = f"click a kept hold to mark as {role}"
            return
        self._snapshot()
        target = self.start if role == "start" else self.top
        target.symmetric_difference_update({box})  # toggle
        (self.top if role == "start" else self.start).discard(box)

    # -- result -------------------------------------------------------------------------------
    def colour_label(self) -> str:
        if self.colour_override:
            return self.colour_override
        return colour_name(self.ref_hue) if self.ref_hue is not None else "unknown"

    def build_with_shapes(self, problem_id: str) -> tuple[HoldMap, Shapes]:
        h, w = self.image.shape[:2]
        mode = "detector" if self.detector_used else "manual"
        kept = set(self.included())
        return build_holdmap_with_shapes(
            problem_id,
            self.colour_label(),
            self.included(),
            (w, h),
            mode,
            self.cfg,
            start_boxes=self.start or None,
            top_boxes=self.top or None,
            polygons={b: p for b, p in self.polygons.items() if b in kept},
        )

    def build(self, problem_id: str) -> HoldMap:
        return self.build_with_shapes(problem_id)[0]


class Editor:
    """Window + input handling around an EditorState."""

    def __init__(self, state: EditorState, problem_id: str, root: Path | None = None) -> None:
        self.state, self.problem_id, self.root = state, problem_id, root
        self.pending: str | None = None  # "start" / "top" after pressing s / t
        self.poly: list[tuple[int, int]] | None = None  # corners while drawing a polygon
        self.down: tuple[int, int] | None = None
        self.drag: tuple[int, int] | None = None
        self.saved: Path | None = None
        self.done = False
        self.erase_mode = True
        self.ref_pick = state.ref_hue is None

    # -- input (called by the window, or directly by tests) -----------------------------------
    def on_mouse(self, event: int, x: int, y: int) -> None:
        st = self.state
        if self.poly is not None:  # drawing a polygon: every click adds a corner
            if event == cv2.EVENT_LBUTTONUP:
                self.poly.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN:  # undo the last corner (none left: leave)
                if self.poly:
                    self.poly.pop()
                else:
                    self.poly = None
                    st.message = "polygon cancelled"
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.down, self.drag = (x, y), (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.down is not None:
            self.drag = (x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            st.delete_at((x, y))
        elif event == cv2.EVENT_LBUTTONUP and self.down is not None:
            start, self.down, self.drag = self.down, None, None
            moved = max(abs(x - start[0]), abs(y - start[1]))
            if moved >= st.cfg["drag_min_px"]:
                if self.erase_mode:
                    st.erase_rect(start, (x, y))
                else:
                    st.add_box(start, (x, y))
            elif self.pending:
                st.mark(self.pending, (x, y))
                self.pending = None
            elif self.ref_pick:
                st.pick_colour((x, y))
                self.ref_pick = False
            else:
                st.toggle_at((x, y))

    def _close_polygon(self) -> None:
        if self.poly is not None and self.state.add_polygon(self.poly):
            self.poly = None
        elif self.poly is not None and len(self.poly) < 3:
            self.state.message = "click at least 3 corners, then Enter (Esc cancels)"

    def on_key(self, key: int) -> None:
        st = self.state
        if self.poly is not None:  # polygon mode has its own Enter / Esc / p
            if key == KEY_ESC:
                self.poly = None
                st.message = "polygon cancelled"
            elif key in (KEY_ENTER, ord("p")):
                self._close_polygon()
            return
        if key == KEY_ESC:
            self.done = True
        elif key == KEY_ENTER:
            if st.included():
                hm, shapes = st.build_with_shapes(self.problem_id)
                self.saved = save_holdmap(hm, self.root, shapes)
                self.done = True
            else:
                st.message = "nothing to save yet"
        elif key == ord("u"):
            st.undo()
        elif key == ord("r"):
            self.erase_mode = not self.erase_mode
            st.message = f"drag action: {'erase' if self.erase_mode else 'plain box placement'}"
        elif key == ord("c"):
            st.clear_colour()
            self.ref_pick = True
        elif key == ord("p"):
            self.poly = []
            self.pending = None
            self.erase_mode = False
            st.message = "polygon: click corners, Enter = close, Esc = cancel"
        elif key in (ord("s"), ord("t")):
            self.pending = "start" if key == ord("s") else "top"
            st.message = f"click a hold to mark as {self.pending}"

    # -- drawing / window ---------------------------------------------------------------------
    def render(self) -> np.ndarray:
        st = self.state
        canvas = st.image.copy()
        k = ui_scale(canvas.shape[0])
        for box in st.visible_auto():
            if box in st.excluded:
                cv2.rectangle(canvas, box[:2], box[2:], (0, 0, 160), _px(1, k))  # dim red: excluded
        if st.included():
            hm, shapes = st.build_with_shapes(self.problem_id)
            h, w = canvas.shape[:2]
            for hold in hm.holds:
                colour = {"start": (0, 255, 0), "top": (0, 0, 255)}.get(hold.role, (255, 255, 0))
                x0, y0, x1, y1 = (int(v * k) for v, k in zip(hold.bbox, (w, h, w, h), strict=True))
                if hold.id in shapes:
                    pts = np.array([(round(x * w), round(y * h)) for x, y in shapes[hold.id]])
                    cv2.polylines(canvas, [pts.astype(np.int32)], True, colour, _px(2, k))
                else:
                    cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, _px(2, k))
                label = (
                    f"{hold.id}{'S' if hold.role == 'start' else 'T' if hold.role == 'top' else ''}"
                )
                cv2.putText(
                    canvas,
                    label,
                    (x0, max(y0 - _px(4, k), _px(12, k))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 * k,
                    colour,
                    _px(2, k),
                )
        if self.poly:  # the polygon being drawn: corners and the edges so far
            for a, b in zip(self.poly, self.poly[1:], strict=False):
                cv2.line(canvas, a, b, POLY_COLOUR, _px(2, k))
            for pt in self.poly:
                cv2.circle(canvas, pt, _px(4, k), POLY_COLOUR, -1)
        if self.down is not None and self.drag is not None:
            erase = self.erase_mode
            colour = (0, 0, 255) if erase else (255, 255, 255)
            cv2.rectangle(canvas, self.down, self.drag, colour, _px(2 if erase else 1, k))
            if erase:
                rect = (*self.down, *self.drag)
                x0, x1 = sorted((rect[0], rect[2]))
                y0, y1 = sorted((rect[1], rect[3]))
                for box in st.included():
                    touches = (
                        polygon_intersects_rect(st.polygons[box], (x0, y0, x1, y1))
                        if box in st.polygons
                        else box_intersects_rect(box, (x0, y0, x1, y1))
                    )
                    if touches:
                        if box in st.polygons:
                            pts = np.asarray(st.polygons[box], np.int32)
                            cv2.polylines(canvas, [pts], True, (0, 0, 255), _px(3, k))
                        else:
                            cv2.rectangle(canvas, box[:2], box[2:], (0, 0, 255), _px(3, k))
        bar_px = _px(HELP_BAR_PX, k)
        action = "erase" if self.erase_mode else "plain box placement"
        status = f"{st.message} | {len(st.included())} kept | drag: {action}"
        width = canvas.shape[1] - 2 * _px(6, k)
        status_lines = wrap_text(status, width, 0.5 * k, _px(1, k))[:STATUS_ROWS]
        status_lines += [""] * (STATUS_ROWS - len(status_lines))  # fixed height: no window jumps
        rows = [(line, 0.5 * k, (0, 255, 255)) for line in status_lines]
        for text in HELP_LINES:
            rows += [
                (line, 0.45 * k, (255, 255, 255))
                for line in wrap_text(text, width, 0.45 * k, _px(1, k))
            ]
        bar = np.zeros((bar_px * len(rows) + _px(6, k), canvas.shape[1], 3), np.uint8)
        for i, (line, scale, colour) in enumerate(rows):
            y = _px(18, k) + i * bar_px
            cv2.putText(
                bar, line, (_px(6, k), y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, _px(1, k)
            )
        return np.vstack([canvas, bar])

    def run(self) -> Path | None:
        cv2.namedWindow("holdmap", cv2.WINDOW_NORMAL)  # resizable
        total_h, img_w = self.render().shape[:2]
        fit = min(1.0, DISPLAY_H / total_h, DISPLAY_W / img_w)
        cv2.resizeWindow("holdmap", round(img_w * fit), round(total_h * fit))  # fits the screen
        cv2.setMouseCallback("holdmap", lambda ev, x, y, *_: self.on_mouse(ev, x, y))
        try:
            while not self.done:
                cv2.imshow("holdmap", self.render())
                key = cv2.waitKey(30) & 0xFF
                if key != 0xFF:
                    self.on_key(key)
        finally:
            cv2.destroyAllWindows()
        return self.saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Pick a hold colour, edit the map, and save it.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image")
    src.add_argument("--video")
    parser.add_argument("--at", type=float, default=1.0, help="seconds into --video")
    parser.add_argument("--problem", required=True)
    parser.add_argument("--colour-name", help="spoken colour name, e.g. pink")
    parser.add_argument(
        "--edit", action="store_true", help="start from the saved map for --problem"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            parser.error(f"cannot read image {args.image}")
    else:
        image = read_frame(args.video, args.at)
    state = EditorState(image, load_holds_config(), args.colour_name)
    if args.edit:
        state.load_existing(*load_holdmap_with_shapes(args.problem))
    saved = Editor(state, args.problem).run()
    log.info("saved %s", saved) if saved else log.info("quit without saving")


if __name__ == "__main__":
    main()
