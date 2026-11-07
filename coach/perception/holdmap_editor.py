"""Plain OpenCV hold-map editor (setup tool, never part of the live loop).

    uv run python -m coach.perception.holdmap_editor --video data/footage/clip.mp4 --at 1.0 \
        --problem pink-v3 [--mode detector|colour] [--colour-name pink]
    uv run python -m coach.perception.holdmap_editor --image wall.jpg --problem paper-wall
    uv run python -m coach.perception.holdmap_editor --video clip.mp4 --problem pink-v3 --edit

Use a frame from the SAME video as the climb (first seconds, wall empty) so colours match.
`--edit` loads the saved map for --problem and lets you fix it (polygons included).

Mouse:  first click picks the route colour · click a hold to toggle it in/out ·
        drag on empty wall to draw a missing box · right-click a hold = delete it
Keys:   n = no colour (white or grey holds): draw everything by hand ·
        p = polygon: click the corners of a diagonal hold, Enter = close, Esc = cancel,
            right-click = remove the last corner ·
        u = undo · s + click = mark start · t + click = mark top · c = re-pick colour ·
        m = switch detector/colour mode · Enter = save · Esc = quit without saving
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
    colour_only_holds,
    detect_holds,
    filter_by_colour,
    load_holdmap_with_shapes,
    load_holds_config,
    read_frame,
    sample_colour,
    save_holdmap,
)
from coach.perception.shapes import is_valid_polygon, point_in_polygon, polygon_area, polygon_bounds

log = logging.getLogger(__name__)

KEY_ENTER, KEY_ESC = 13, 27
HELP_LINES = (
    "click=toggle  right-click=delete  drag=box  u=undo  n=no colour (white holds)",
    "p=polygon (click corners, Enter=close, Esc=cancel)  s/t+click=start/top",
    "c=colour  m=mode  Enter=save  Esc=quit",
)
HELP_BAR_PX = 26
MIN_BOX_PX = 3
POLY_COLOUR = (0, 200, 255)  # BGR: the polygon being drawn


def _contains(box: Box, p: tuple[int, int]) -> bool:
    return box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3]


def _area(box: Box) -> int:
    return (box[2] - box[0]) * (box[3] - box[1])


class EditorState:
    """All editing logic, with no window: easy to test with synthetic images and clicks."""

    def __init__(
        self, image: np.ndarray, cfg: dict, mode: str = "colour", colour: str | None = None
    ) -> None:
        self.image, self.cfg, self.mode = image, cfg, mode
        self.colour_override = colour
        self.ref_hue: float | None = None
        self.manual_only = False  # no colour: every hold is drawn by hand
        self.auto: list[Box] = []
        self.excluded: set[Box] = set()  # auto boxes switched off (shown dim, click to restore)
        self.deleted: set[Box] = set()  # auto boxes removed for good (hidden; undo restores)
        self._undo: list[tuple] = []
        self.manual: list[Box] = []
        self.polygons: dict[Box, PixelPolygon] = {}  # a manual box's real outline (diagonal hold)
        self.start: set[Box] = set()
        self.top: set[Box] = set()
        self.message = "click a hold of the route colour (white holds: press n)"

    @property
    def ready(self) -> bool:
        """Can holds be drawn and edited yet? Yes once a colour is picked or manual mode is on."""
        return self.ref_hue is not None or self.manual_only

    # -- colour / mode ------------------------------------------------------------------------
    def pick_colour(self, point: tuple[int, int]) -> None:
        try:
            self.ref_hue, _ = sample_colour(self.image, point, self.cfg)
        except ValueError as e:
            self.message = f"{e} (white or grey holds: press n)"
            return
        self.manual_only = False
        self.recompute()

    def start_manual(self) -> None:
        """No colour: skip detection and draw every hold (box or polygon) by hand."""
        self.ref_hue = None
        self.manual_only = True
        self.message = "manual mode: drag = box, p = polygon for diagonal holds"

    def clear_colour(self) -> None:
        self.ref_hue = None
        self.manual_only = False
        self.message = "click a hold of the route colour (white holds: press n)"

    def toggle_mode(self) -> None:
        self.mode = "detector" if self.mode == "colour" else "colour"
        if self.ref_hue is not None:
            self.recompute()
        else:
            self.message = f"mode: {self.mode}"

    def recompute(self) -> None:
        """Re-run the automatic detection for the current colour and mode."""
        assert self.ref_hue is not None
        if self.mode == "detector":
            try:
                found = detect_holds(self.image, self.cfg)
                self.auto = filter_by_colour(self.image, found, self.ref_hue, self.cfg)
            except Exception as e:  # no API key, no network, bad model id: fall back
                log.warning("detector failed (%s): using colour-only mode", e)
                self.mode = "colour"
                self.auto = colour_only_holds(self.image, self.ref_hue, self.cfg)
                self.message = "detector unavailable, switched to colour mode"
                self._prune_marks()
                return
        else:
            self.auto = colour_only_holds(self.image, self.ref_hue, self.cfg)
        self.excluded.clear()
        self.deleted.clear()
        self._undo.clear()  # the automatic boxes changed, old snapshots no longer apply
        self._prune_marks()
        self.message = f"{self.mode} mode: {len(self.auto)} holds found"

    def _prune_marks(self) -> None:
        keep = set(self.included())
        self.start &= keep
        self.top &= keep

    def load_existing(self, holdmap: HoldMap, shapes: Shapes) -> None:
        """Start from a saved map (every hold becomes an editable manual hold)."""
        h, w = self.image.shape[:2]
        self.start_manual()
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

    def _shape_area(self, box: Box) -> float:
        return polygon_area(self.polygons[box]) if box in self.polygons else _area(box)

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
        mode = "manual" if self.manual else self.mode
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
        elif event == cv2.EVENT_RBUTTONDOWN and st.ready:
            st.delete_at((x, y))
        elif event == cv2.EVENT_LBUTTONUP and self.down is not None:
            start, self.down, self.drag = self.down, None, None
            moved = max(abs(x - start[0]), abs(y - start[1]))
            if moved >= st.cfg["drag_min_px"]:
                if st.ready:
                    st.add_box(start, (x, y))
            elif not st.ready:
                st.pick_colour((x, y))
            elif self.pending:
                st.mark(self.pending, (x, y))
                self.pending = None
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
        elif key == ord("c"):
            st.clear_colour()
        elif key == ord("n"):
            st.start_manual()
        elif key == ord("m"):
            st.toggle_mode()
        elif key == ord("p"):
            if st.ready:
                self.poly = []
                self.pending = None
                st.message = "polygon: click the corners, Enter = close, Esc = cancel"
            else:
                st.message = "pick a colour first, or press n to draw by hand"
        elif key in (ord("s"), ord("t")):
            self.pending = "start" if key == ord("s") else "top"
            st.message = f"click a hold to mark as {self.pending}"

    # -- drawing / window ---------------------------------------------------------------------
    def render(self) -> np.ndarray:
        st = self.state
        canvas = st.image.copy()
        for box in st.visible_auto():
            if box in st.excluded:
                cv2.rectangle(canvas, box[:2], box[2:], (0, 0, 160), 1)  # dim red: excluded
        if st.included():
            hm, shapes = st.build_with_shapes(self.problem_id)
            h, w = canvas.shape[:2]
            for hold in hm.holds:
                colour = {"start": (0, 255, 0), "top": (0, 0, 255)}.get(hold.role, (255, 255, 0))
                x0, y0, x1, y1 = (int(v * k) for v, k in zip(hold.bbox, (w, h, w, h), strict=True))
                if hold.id in shapes:
                    pts = np.array([(round(x * w), round(y * h)) for x, y in shapes[hold.id]])
                    cv2.polylines(canvas, [pts.astype(np.int32)], True, colour, 2)
                else:
                    cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, 2)
                label = (
                    f"{hold.id}{'S' if hold.role == 'start' else 'T' if hold.role == 'top' else ''}"
                )
                cv2.putText(
                    canvas, label, (x0, max(y0 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2
                )
        if self.poly:  # the polygon being drawn: corners and the edges so far
            for a, b in zip(self.poly, self.poly[1:], strict=False):
                cv2.line(canvas, a, b, POLY_COLOUR, 2)
            for pt in self.poly:
                cv2.circle(canvas, pt, 4, POLY_COLOUR, -1)
        if self.down is not None and self.drag is not None:
            cv2.rectangle(canvas, self.down, self.drag, (255, 255, 255), 1)
        bar = np.zeros((HELP_BAR_PX * (1 + len(HELP_LINES)), canvas.shape[1], 3), np.uint8)
        cv2.putText(bar, st.message, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        for i, line in enumerate(HELP_LINES, start=1):
            y = 18 + i * HELP_BAR_PX
            cv2.putText(bar, line, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return np.vstack([canvas, bar])

    def run(self) -> Path | None:
        cv2.namedWindow("holdmap")
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
    parser = argparse.ArgumentParser(description="Hold map editor")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image")
    src.add_argument("--video")
    parser.add_argument("--at", type=float, default=1.0, help="seconds into --video")
    parser.add_argument("--problem", required=True)
    parser.add_argument("--mode", choices=("detector", "colour"), default="colour")
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
    state = EditorState(image, load_holds_config(), args.mode, args.colour_name)
    if args.edit:
        state.load_existing(*load_holdmap_with_shapes(args.problem))
    saved = Editor(state, args.problem).run()
    log.info("saved %s", saved) if saved else log.info("quit without saving")


if __name__ == "__main__":
    main()
