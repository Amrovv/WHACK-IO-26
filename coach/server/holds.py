"""Hold map builder for the dashboard: A's EditorState, driven over HTTP. Owned by D.

The dashboard shows a still of the empty wall. Clicking a hold of the route's colour runs A's
Roboflow detector and keeps the holds of that colour; further clicks and drags leave holds out,
put them back, erase, add boxes and mark start and top holds. Picking the colour again starts
over (Undo brings the earlier holds back). Saving writes data/holdmaps/<problem>.json and
restarts analysis, so A's contact tracking and B's brain both load the new map.

Analysis pauses while the editor is open and restarts when it closes, so the coach doesn't talk
over the editing. A map only lines up with video filmed from the same spot, so each still is
compared with the video's newest frame (ORB features, about 10 ms a still): the ones that match
are offered first, and the others are flagged.

A video still can be any moment of its clip: the "frame" op moves to another moment and keeps
the holds (same clip, same camera). Your own pictures can be uploaded too; they're kept in
data/stills/<problem>/ and offered with the others.

Roboflow is a network call (about 1.5 s). It runs only when a colour is picked: setup time,
never the live loop, and always off the event loop.

  GET  /api/holds            the problem, whether it has a map, the stills to build from and
                             the one used last
  GET  /api/holds/view       the open editor's view ({"open": false} when none is open)
  POST /api/holds/open       {"still": key, "at"?: seconds} -> editor view (pauses analysis)
  POST /api/holds/upload     ?name=<file name>, body: the picture (JPEG/PNG) -> editor view on it
  GET  /api/holds/still.jpg  the picture being edited
  POST /api/holds/op         {"op": ..., "x", "y" | "x0", "y0", "x1", "y1" | "at"} -> editor view
  POST /api/holds/save       save, restart analysis -> {"holds", "path", "hello"}
  POST /api/holds/close      leave without saving (restarts analysis)
  GET  /api/holds/shapes     {"problem_id", "shapes": {hold id: polygon}} for the live overlay

All coordinates between the page and the server are normalised 0-1, origin top-left, y down.

  uv run python -m coach.server.holds --video videos/mp4/climb_1_00_empty_wall.mp4 \
      --click 0.23 0.36       # demo: one colour pick on a real frame (needs ROBOFLOW_API_KEY)
  uv run python -m coach.server.holds --video "videos/mp4/climb_2_00_empty_wall.mp4" \
      --against videos/mp4/climb_2_05_hand_searching.mp4   # does the still match the clip?
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

import numpy as np

log = logging.getLogger("coach.server.holds")

STILL_JPEG_QUALITY = 90  # the editing picture: sharp enough to judge a hold's edge
SAME_SHAPE_TOLERANCE = 0.02  # a saved map is reused only for a picture of the same aspect ratio
ROLES = ("start", "top")
POINT_OPS = {"pick", "toggle", "delete", "start", "top"}
RECT_OPS = {"erase", "box"}
PLAIN_OPS = {"undo", "recolour", "keep", "frame"}
UPLOAD_DIR = Path(__file__).resolve().parents[2] / "data" / "stills"  # <problem>/<name>.jpg
UPLOAD_PREFIX = "upload:"  # still key of an uploaded picture: upload:<file name>
UPLOAD_MAX_BYTES = 25 * 1024 * 1024  # the page sends a JPEG of at most 2560 px: far below this
UPLOAD_MIN_SIDE_PX = 64  # smaller than this isn't a picture of a wall
UPLOAD_JPEG_QUALITY = 95  # kept on disk: as sharp as what was sent
UPLOAD_NAME_CHARS = 60  # longest file name stem kept
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
# Was a still filmed from where the video's camera is? On the test clips the same spot moves the
# wall by at most 0.008 of the picture's width, and a moved camera by 0.085 or more.
MATCH_WIDTH_PX = 480  # pictures are compared at this width
MATCH_FEATURES = 1500  # ORB corners found per picture
MATCH_MIN_INLIERS = 40  # fewer matches than this agree on one camera move: can't tell
MATCH_RANSAC = 0.01  # a match agrees with the camera move to within this fraction of the width
SAME_VIEW_MAX_SHIFT = 0.03  # same spot: no part of the wall moves further (fraction of width)
FIT_ORDER = {True: 0, None: 1, False: 2}  # matching stills first, then unknown, then the rest
PICK_PROMPT = "Click a hold of the route's colour."
FEATURE_CACHE_MAX = 64  # stills' features kept (about 60 kB each)


class Unavailable(RuntimeError):
    """A's hold map code isn't on this branch."""


class HoldsControl(Protocol):
    """What the builder needs from run.py's Session."""

    @property
    def problem_id(self) -> str: ...

    # [{"key", "name"}, plus "duration" (seconds) for a video and "live": True for the video
    # being analysed], the recommended one first
    def stills(self) -> list[dict]: ...

    # BGR, `at` seconds into a video (None: its default moment); ValueError for an unknown key
    def read_still(self, key: str, at: float | None = None) -> np.ndarray: ...

    def latest_frame(self) -> np.ndarray | None: ...  # the video's newest frame (BGR), if any

    async def pause(self) -> None: ...  # stop analysing while the map is edited

    async def resume(self) -> dict: ...  # start analysing again -> new hello

    async def reload(self) -> dict: ...  # restart analysis on the same source -> new hello


def a_parts() -> SimpleNamespace:
    """A's hold map building blocks. Raises Unavailable when A's code isn't on this branch."""
    try:
        from coach.perception.holdmap_editor import EditorState
        from coach.perception.holds import (
            holdmap_path,
            load_holdmap_with_shapes,
            load_holds_config,
            save_holdmap,
        )
    except ImportError as e:
        raise Unavailable(f"the hold map editor needs A's perception code ({e})") from e
    return SimpleNamespace(
        EditorState=EditorState,
        load_cfg=load_holds_config,
        path=holdmap_path,  # (problem_id, root) -> Path
        load=load_holdmap_with_shapes,  # (problem_id, root) -> (HoldMap, shapes)
        save=save_holdmap,  # (holdmap, root, shapes) -> Path
    )


def encode_jpeg(img: np.ndarray) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, STILL_JPEG_QUALITY])
    if not ok:
        raise ValueError("could not encode the still as JPEG")
    return buf.tobytes()


def same_shape(size_a: tuple[int, int], size_b: tuple[int, int]) -> bool:
    """Two (width, height) sizes with the same aspect ratio, so normalised boxes line up."""
    (wa, ha), (wb, hb) = size_a, size_b
    return ha > 0 and hb > 0 and abs(wa / ha - wb / hb) <= SAME_SHAPE_TOLERANCE


def first_clause(message: str) -> str:
    """A's messages end with desktop-editor key hints ("...; use r to place boxes"): drop them."""
    return message.split(";")[0].strip()


def safe_name(text: str, fallback: str) -> str:
    """A file or folder name made only of letters, digits, - and _ (no paths)."""
    name = re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-")[:UPLOAD_NAME_CHARS]
    return name or fallback


@dataclass(frozen=True)
class Features:
    """ORB corners of one picture, for telling whether two pictures share a camera position."""

    aspect: float  # height / width
    points: np.ndarray  # (n, 2): x and y as fractions of the width
    descriptors: np.ndarray | None


def features(img: np.ndarray) -> Features:
    import cv2

    h, w = img.shape[:2]
    small = cv2.resize(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        (MATCH_WIDTH_PX, max(1, round(h * MATCH_WIDTH_PX / w))),
        interpolation=cv2.INTER_AREA,
    )
    corners, descriptors = cv2.ORB_create(MATCH_FEATURES).detectAndCompute(small, None)
    points = np.float32([c.pt for c in corners]).reshape(-1, 2) / MATCH_WIDTH_PX
    return Features(h / w, points, descriptors)


def camera_shift(a: Features, b: Features) -> float | None:
    """How far the wall moves from picture a to picture b, as a fraction of the width: about 0
    when both were filmed from the same spot. None when too little matches to tell."""
    import cv2

    if a.descriptors is None or b.descriptors is None:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(a.descriptors, b.descriptors)
    if len(matches) < MATCH_MIN_INLIERS:
        return None
    src = a.points[[m.queryIdx for m in matches]]
    dst = b.points[[m.trainIdx for m in matches]]
    move, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=MATCH_RANSAC
    )
    if move is None or int(inliers.sum()) < MATCH_MIN_INLIERS:
        return None
    grid = np.float32([(x, y * a.aspect) for x in (0.1, 0.5, 0.9) for y in (0.1, 0.5, 0.9)])
    moved = grid @ move[:, :2].T + move[:, 2]
    return float(np.linalg.norm(moved - grid, axis=1).max())


def same_view(still: Features, frame: Features) -> bool | None:
    """Was the still filmed from where the video's camera is? None when it can't be told."""
    if abs(still.aspect - frame.aspect) > SAME_SHAPE_TOLERANCE:
        return False
    shift = camera_shift(still, frame)
    return None if shift is None else shift <= SAME_VIEW_MAX_SHIFT


class HoldMapBuilder:
    """One editing session at a time, shared by every dashboard."""

    def __init__(
        self,
        control: HoldsControl,
        parts: SimpleNamespace | None = None,
        root=None,
        uploads=None,
    ):
        self.control = control
        self._parts = parts
        self.root = Path(root) if root else None  # hold map folder; None = A's default
        self.uploads = Path(uploads) if uploads else UPLOAD_DIR  # uploaded pictures
        self.editor: Any = None  # A's EditorState while a map is open
        self.problem_id: str | None = None
        self.still_key: str | None = None
        self.at: float | None = None  # seconds into the open still's video
        self.duration: float | None = None  # the open still's video length (None: a picture)
        self.jpeg: bytes | None = None
        self.version = 0  # bumps with each new still, so the page reloads the picture
        self.message = ""
        self.from_saved = False  # the open map started from the saved file
        self.fits: bool | None = None  # the open still was filmed from the video's camera spot
        self.picking = False  # after "Pick the colour again": the next click picks a colour
        self._replaced: list[tuple[Any, bool]] = []  # (editor, from_saved) before each re-pick
        self._still_features: dict[tuple, Features] = {}  # (key, at) -> features: fixed pictures
        self._last: dict[str, dict] = {}  # problem -> {"key", "at"} of the still used last
        self._lock = asyncio.Lock()

    def parts(self) -> SimpleNamespace:
        if self._parts is None:
            self._parts = a_parts()
        return self._parts

    def _path(self, problem_id: str) -> Path:
        return self.parts().path(problem_id, self.root)

    # -- reading ----------------------------------------------------------------------------

    def status(self) -> dict:
        """Blocking (stills are read from video and compared): call it off the loop."""
        problem = self.control.problem_id
        try:
            has_map = self._path(problem).exists()
            available, reason = True, None
        except Unavailable as e:
            has_map, available, reason = False, False, str(e)
        stills = self.stills()
        last = self._last.get(problem)
        return {
            "available": available,
            "reason": reason,
            "problem_id": problem,
            "has_map": has_map,
            "stills": stills,
            "last": last if last and any(s["key"] == last["key"] for s in stills) else None,
            "open": self.editor is not None,
        }

    def stills(self) -> list[dict]:
        """The control's stills and the uploaded pictures, each with "fits": filmed from where
        the video's camera is (True), from somewhere else (False) or unknown (None). Matching
        ones come first."""
        frame = self.control.latest_frame()
        ref = features(frame) if frame is not None else None
        out = [{**s, "fits": self._fits(s, ref)} for s in self._all_stills()]
        return sorted(out, key=lambda s: FIT_ORDER[s["fits"]])  # stable: keeps the control's order

    def _all_stills(self) -> list[dict]:
        return [*self.control.stills(), *self._uploaded()]

    def _find(self, key: str) -> dict:
        still = next((s for s in self._all_stills() if s["key"] == key), None)
        if still is None:
            raise ValueError(f"not an offered still: {key!r}")
        return still

    def _read(self, key: str, at: float | None = None) -> np.ndarray:
        """The picture for a still key: an uploaded picture, or whatever the control has.
        Blocking."""
        if not key.startswith(UPLOAD_PREFIX):
            return self.control.read_still(key, at)
        import cv2

        self._find(key)  # only pictures that are offered: no paths from the page
        img = cv2.imread(str(self._upload_folder() / key.removeprefix(UPLOAD_PREFIX)))
        if img is None:
            raise ValueError(f"can't read the picture {key.removeprefix(UPLOAD_PREFIX)!r}")
        return img

    def _fits(
        self,
        still: dict,
        ref: Features | None,
        img: np.ndarray | None = None,
        at: float | None = None,
    ) -> bool | None:
        if still.get("live"):  # the video being analysed lines up with itself
            return True
        if ref is None:
            return None
        cache = (still["key"], at)
        if cache not in self._still_features:
            if len(self._still_features) >= FEATURE_CACHE_MAX:  # scrubbing visits many moments
                self._still_features.clear()
            try:
                pic = img if img is not None else self._read(still["key"], at)
            except ValueError as e:
                log.warning("can't read still %r: %s", still["key"], e)
                return None
            self._still_features[cache] = features(pic)
        return same_view(self._still_features[cache], ref)

    def _still_fits(self, key: str, img: np.ndarray, at: float | None = None) -> bool | None:
        still = next((s for s in self._all_stills() if s["key"] == key), {"key": key})
        frame = self.control.latest_frame()
        return self._fits(still, features(frame) if frame is not None else None, img, at)

    # -- uploaded pictures --------------------------------------------------------------------

    def _upload_folder(self, problem: str | None = None) -> Path:
        return self.uploads / safe_name(problem or self.control.problem_id, "problem")

    def _uploaded(self) -> list[dict]:
        """This problem's uploaded pictures, newest first."""
        folder = self._upload_folder()
        if not folder.is_dir():
            return []
        pics = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
        pics.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return [{"key": UPLOAD_PREFIX + p.name, "name": f"Your picture · {p.name}"} for p in pics]

    def _keep_upload(self, name: str, data: bytes) -> str:
        """Decode an uploaded picture, keep it as a JPEG in this problem's folder -> its key."""
        import cv2

        if len(data) > UPLOAD_MAX_BYTES:
            raise ValueError(f"that picture is over {UPLOAD_MAX_BYTES // (1024 * 1024)} MB")
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
        if img is None:
            raise ValueError("that file isn't a picture the editor can read: use a JPEG or PNG")
        if min(img.shape[:2]) < UPLOAD_MIN_SIDE_PX:
            raise ValueError("that picture is too small to show holds")
        folder = self._upload_folder()
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{safe_name(Path(name).stem, 'picture')}.jpg"
        if not cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, UPLOAD_JPEG_QUALITY]):
            raise ValueError("couldn't keep the picture")
        key = UPLOAD_PREFIX + path.name
        self._still_features = {k: v for k, v in self._still_features.items() if k[0] != key}
        log.info("kept an uploaded picture at %s (%dx%d)", path, img.shape[1], img.shape[0])
        return key

    async def upload(self, name: str, data: bytes) -> dict:
        """Keep an uploaded picture of the wall for this problem and open the editor on it."""
        key = await asyncio.to_thread(self._keep_upload, name, data)
        return await self.open(key)

    def shapes(self) -> dict:
        """Outlines of the current problem's diagonal holds, keyed by hold id."""
        problem = self.control.problem_id
        try:
            _, shapes = self.parts().load(problem, self.root)
        except (Unavailable, FileNotFoundError):
            shapes = {}
        return {
            "problem_id": problem,
            "shapes": {str(k): [list(p) for p in v] for k, v in shapes.items()},
        }

    def view(self) -> dict:
        """Everything the page needs to draw the editor."""
        ed = self.editor
        if ed is None:
            return {"open": False}
        h, w = ed.image.shape[:2]
        kept = ed.included()
        ids: dict[tuple, Any] = {}
        if kept:  # A numbers the kept holds bottom (1) to top (N); show those numbers now
            holdmap, _ = ed.build_with_shapes(self.problem_id)
            ids = {tuple(hold.bbox): hold for hold in holdmap.holds}
        holds = []
        for box in [*ed.visible_auto(), *ed.manual]:
            x0, y0, x1, y1 = box
            hold = ids.get((x0 / w, y0 / h, x1 / w, y1 / h)) if box in kept else None
            poly = ed.polygons.get(box)
            holds.append(
                {
                    "box": [x0 / w, y0 / h, x1 / w, y1 / h],
                    "polygon": [[x / w, y / h] for x, y in poly] if poly else None,
                    "kept": box in kept,
                    "id": hold.id if hold else None,
                    "role": hold.role if hold else None,
                }
            )
        return {
            "open": True,
            "problem_id": self.problem_id,
            "still": self.still_key,
            "at": self.at,
            "duration": self.duration,
            "version": self.version,
            "size": [w, h],
            "fits": self.fits,
            "picking": self.picking,
            "picked": ed.ref_hue is not None,
            "colour": ed.colour_label() if (ed.ref_hue is not None or ed.colour_override) else None,
            "detector": ed.detector_used,
            "origin": (
                "found by Roboflow"
                if ed.detector_used
                else "from the saved map"
                if self.from_saved
                else "placed by hand"
            ),
            "kept": len(kept),
            "holds": holds,
            "message": self.message,
        }

    # -- editing ----------------------------------------------------------------------------

    async def open(self, still_key: str, at: float | None = None) -> dict:
        """Open the editor on a still (a video's still `at` seconds in; None: its default
        moment), starting from the saved map when it fits the picture's shape."""
        parts = self.parts()
        async with self._lock:
            still = await asyncio.to_thread(self._find, still_key)
            duration = still.get("duration")
            at = (still.get("at") if at is None else float(at)) if duration else None
            if at is not None:
                at = min(max(at, 0.0), duration)
            img = await asyncio.to_thread(self._read, still_key, at)
            fits = await asyncio.to_thread(self._still_fits, still_key, img, at)
            # Analysis waits while the map is edited; close() or save() starts it again.
            await self.control.pause()
            problem = self.control.problem_id
            ed = parts.EditorState(img, parts.load_cfg())
            h, w = img.shape[:2]
            self.message = PICK_PROMPT
            self.from_saved, self.fits, self.picking, self._replaced = False, fits, False, []
            if self._path(problem).exists():
                holdmap, shapes = parts.load(problem, self.root)
                if same_shape(tuple(holdmap.image_size), (w, h)):
                    ed.load_existing(holdmap, shapes)
                    self.from_saved = True
                    self.message = f"Editing the saved map ({len(holdmap.holds)} holds)."
                else:
                    mw, mh = holdmap.image_size
                    self.message = (
                        f"The saved map ({mw}×{mh}) doesn't fit this picture. {PICK_PROMPT}"
                    )
            self.jpeg = await asyncio.to_thread(encode_jpeg, img)
            self.editor, self.problem_id, self.still_key = ed, problem, still_key
            self.at, self.duration = at, duration
            self._last[problem] = {"key": still_key, "at": at}
            self.version += 1
            return self.view()

    async def op(self, name: str, args: dict) -> dict:
        async with self._lock:
            if self.editor is None:
                raise ValueError("no hold map is open")
            # Picking a colour calls Roboflow over the network, so every op runs off the loop.
            self.message = await asyncio.to_thread(self._apply, name, args)
            return self.view()

    def _apply(self, name: str, args: dict) -> str:
        ed = self.editor
        h, w = ed.image.shape[:2]

        def px(key_x: str, key_y: str) -> tuple[int, int]:
            x, y = float(args[key_x]), float(args[key_y])
            if not (0 <= x <= 1 and 0 <= y <= 1):
                raise ValueError("coordinates are normalised 0-1")
            return round(x * (w - 1)), round(y * (h - 1))

        before = len(ed.included())
        if name in POINT_OPS:
            p = px("x", "y")
            if name == "pick":
                # Picking starts over in a fresh editor. A's pick_colour keeps every hold placed
                # by hand (and every hold loaded from the saved map) and adds the detected ones,
                # so picking over a saved map doubled it. A missed click changes nothing.
                fresh = self.parts().EditorState(ed.image, ed.cfg)
                fresh.pick_colour(p)
                # A reports success as "Roboflow found N holds"; judge by the message.
                if not fresh.message.startswith("Roboflow found"):
                    reason = first_clause(fresh.message)
                    return f"Couldn't find holds: {reason}."
                self._replaced.append((ed, self.from_saved))
                self.editor, self.from_saved, self.picking = fresh, False, False
                found = f"{len(fresh.included())} {fresh.colour_label()} holds"
                return f"{'Started over with' if before else 'Found'} {found}."
            box = ed.box_at(p)
            if name == "toggle":
                if box is None:
                    return "There's no hold there. Use Add box to draw one."
                ed.toggle_at(p)
                return "Put a hold back." if len(ed.included()) > before else "Left a hold out."
            if name == "delete":
                if box is None:
                    return "There's no hold there to delete."
                ed.delete_at(p)
                return "Deleted a hold."
            if box is None or box not in ed.included():  # start / top
                return f"Click a kept hold to mark it as the {name}."
            self._keep_automatic_roles(name)
            marks = ed.start if name == "start" else ed.top
            ed.mark(name, p)
            return f"Marked a {name} hold." if box in marks else f"Unmarked a {name} hold."
        if name in RECT_OPS:
            p0, p1 = px("x0", "y0"), px("x1", "y1")
            if name == "erase":
                ed.erase_rect(p0, p1)
                gone = before - len(ed.included())
                return (
                    f"Erased {gone} hold{'s' if gone != 1 else ''}."
                    if gone
                    else ("Nothing to erase there.")
                )
            manual = len(ed.manual)
            ed.add_box(p0, p1)
            return (
                "Added a hold." if len(ed.manual) > manual else "Drag a bigger box to add a hold."
            )
        if name == "frame":
            return self._move_to(float(args["at"]))
        if name == "undo":
            ed.undo()
            if ed.message != "nothing to undo":
                return "Undone."
            if self._replaced:  # back to the holds from before the last colour pick
                self.editor, self.from_saved = self._replaced.pop()
                return "Undone: the earlier holds are back."
            return "Nothing to undo."
        if name == "recolour":
            self.picking = True
            return PICK_PROMPT
        if name == "keep":
            self.picking = False
            return "Kept the holds you have."
        raise ValueError(f"unknown op {name!r}")

    def _move_to(self, at: float) -> str:
        """Show another moment of the open still's video. Same clip, same camera, so the holds
        stay where they are; a colour picked later is found in the new picture."""
        if not self.duration:
            raise ValueError("this picture isn't from a video")
        at = min(max(at, 0.0), self.duration)
        img = self._read(self.still_key, at)
        ed = self.editor
        if img.shape != ed.image.shape:
            raise ValueError("that moment of the video is a different size")
        for editor in [ed, *(old for old, _ in self._replaced)]:  # undo keeps the new picture
            editor.image = img
        self.at, self.jpeg = at, encode_jpeg(img)
        self.fits = self._still_fits(self.still_key, img, at)
        self._last[self.problem_id] = {"key": self.still_key, "at": at}
        self.version += 1
        return f"At {at:.1f} s."

    def _keep_automatic_roles(self, marking: str) -> None:
        """A's rule: once any start or top is set by hand, every role is manual. So the first
        time one is marked, carry over the automatic roles it isn't replacing: marking the top
        keeps the automatic starts, marking a start keeps the automatic top."""
        ed = self.editor
        if ed.start or ed.top:  # roles are already manual
            return
        h, w = ed.image.shape[:2]
        boxes = {
            (x0 / w, y0 / h, x1 / w, y1 / h): (x0, y0, x1, y1) for x0, y0, x1, y1 in ed.included()
        }
        holdmap, _ = ed.build_with_shapes(self.problem_id)
        for hold in holdmap.holds:
            box = boxes.get(tuple(hold.bbox))
            if box is None:
                continue
            if hold.role == "start":
                ed.start.add(box)
            elif hold.role == "top" and marking == "start":
                ed.top.add(box)

    async def save(self) -> dict:
        async with self._lock:
            ed = self.editor
            if ed is None:
                raise ValueError("no hold map is open")
            if not ed.included():
                raise ValueError("there are no holds to save")
            holdmap, shapes = ed.build_with_shapes(self.problem_id)
            path = await asyncio.to_thread(self.parts().save, holdmap, self.root, shapes)
            log.info("saved %d holds to %s", len(holdmap.holds), path)
            self._close()
        # A's contact tracking and B's brain read the map when analysis starts.
        hello = await self.control.reload()
        return {"holds": len(holdmap.holds), "path": str(path), "hello": hello}

    async def close(self) -> None:
        async with self._lock:
            was_open = self.editor is not None
            self._close()
        if was_open:
            await self.control.resume()

    def _close(self) -> None:
        self.editor, self.still_key, self.jpeg, self.message = None, None, None, ""
        self.fits, self.picking, self._replaced = None, False, []


if __name__ == "__main__":
    import cv2

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    p = argparse.ArgumentParser(description="Open a still, pick a colour, print what was found.")
    p.add_argument("--video", required=True)
    p.add_argument("--at", type=float, default=1.0, help="seconds into the video")
    p.add_argument("--click", type=float, nargs=2, metavar=("X", "Y"))
    p.add_argument("--against", help="a climbing clip: does the still match its camera position?")
    a = p.parse_args()

    def _frame(path: str) -> np.ndarray:
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_MSEC, a.at * 1000)
        ok, img = cap.read()
        cap.release()
        if not ok:
            raise ValueError(f"cannot read {path} at {a.at} s")
        return img

    class _Control:
        problem_id = "demo"

        def stills(self) -> list[dict]:
            return [{"key": "clip", "name": a.video}]

        def read_still(self, key: str, at: float | None = None) -> np.ndarray:
            return _frame(a.video)

        def latest_frame(self) -> np.ndarray | None:
            return _frame(a.against) if a.against else None

        async def pause(self) -> None:
            pass

        async def resume(self) -> dict:
            return {}

        async def reload(self) -> dict:
            return {}

    async def _demo() -> None:
        builder = HoldMapBuilder(_Control())
        if a.against:
            shift = camera_shift(features(_frame(a.video)), features(_frame(a.against)))
            fits = builder.stills()[0]["fits"]
            log.info("camera moved by %s of the width: fits %s", shift and f"{shift:.3f}", fits)
        if a.click:
            view = await builder.open("clip")
            log.info("%s", view["message"])
            view = await builder.op("pick", {"x": a.click[0], "y": a.click[1]})
            log.info("%s (%d holds kept)", view["message"], view["kept"])

    asyncio.run(_demo())
