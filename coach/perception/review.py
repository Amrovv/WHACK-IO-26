"""Frame-by-frame review of a clip with its saved pose (and contacts), for checking by eye.

    uv run python -m coach.perception.review --video data/footage/clip.mp4 \
        --cache data/pose_cache/clip.jsonl --problem pink-v3

Nothing is re-detected: the skeleton comes from the pose cache. If --problem is given, contacts
are recomputed from the cached poses with the CURRENT config, so what you see is what the code
does now. The bottom-left label shows the frame number and time.

Keys:  space = pause / play · d or . or Right = next frame · a or , or Left = previous frame
       w = forward 1 s · s = back 1 s · q or Esc = quit.  The slider at the top seeks.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from coach.contracts import ClimberState, HoldMap, Limb
from coach.perception.contacts import ContactTracker
from coach.perception.holds import load_holdmap_with_shapes
from coach.perception.pose import draw_overlay, load_perception_config
from coach.perception.serialize import state_from_line

log = logging.getLogger(__name__)

# cv2.waitKeyEx codes for the arrow keys on Windows
KEY_LEFT, KEY_RIGHT = 2424832, 2555904
KEY_ESC = 27
LIMB_ORDER = (Limb.LH, Limb.RH, Limb.LF, Limb.RF)


def key_name(code: int) -> str | None:
    """Map a raw key code to an action name (None = ignore)."""
    if code in (-1, 0xFF):
        return None
    if code == KEY_RIGHT:
        return "next"
    if code == KEY_LEFT:
        return "prev"
    if code == KEY_ESC:
        return "quit"
    ch = chr(code & 0xFF) if 0 <= code < 0x110000 else ""
    return {
        " ": "toggle",
        "d": "next",
        ".": "next",
        "a": "prev",
        ",": "prev",
        "w": "fwd",
        "s": "back",
        "q": "quit",
    }.get(ch)


class Reviewer:
    """Playback state with no window: easy to test."""

    def __init__(self, n_frames: int, fps: float) -> None:
        self.n, self.fps = n_frames, fps
        self.idx = 0
        self.paused = False
        self.done = False

    def seek(self, idx: int) -> None:
        self.idx = min(max(idx, 0), self.n - 1)

    def tick(self) -> None:
        """Called once per loop while playing: advance a frame, pause on the last one."""
        if self.paused:
            return
        if self.idx >= self.n - 1:
            self.paused = True
        else:
            self.idx += 1

    def on_key(self, name: str | None) -> None:
        if name is None:
            return
        if name == "quit":
            self.done = True
        elif name == "toggle":
            self.paused = not self.paused
        elif name in ("next", "prev", "fwd", "back"):
            self.paused = True  # any manual move pauses
            step = {"next": 1, "prev": -1, "fwd": round(self.fps), "back": -round(self.fps)}[name]
            self.seek(self.idx + step)


def contact_text(state: ClimberState) -> str:
    return "  ".join(
        f"{limb.name}:{state.contacts[limb] if state.contacts[limb] is not None else '-'}"
        for limb in LIMB_ORDER
    )


def recompute_contacts(
    states: list[ClimberState], holdmap: HoldMap, shapes: dict | None = None
) -> None:
    """Fill each state's contacts using the current config (run in order, like live)."""
    tracker = ContactTracker(load_perception_config(), holdmap, shapes)
    for s in states:
        s.contacts = tracker.update(s)


def load_states(cache: Path) -> list[ClimberState]:
    return [state_from_line(line) for line in cache.read_text(encoding="utf-8").splitlines()]


def render(
    frame: np.ndarray,
    state: ClimberState,
    holdmap: HoldMap | None,
    fps: float,
    shapes: dict | None = None,
) -> np.ndarray:
    out = draw_overlay(frame.copy(), state, state.frame_idx / fps, holdmap, shapes)
    if holdmap is not None and state.person_found:
        cv2.putText(
            out, contact_text(state), (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
    return out


def run(
    video: Path, states: list[ClimberState], holdmap: HoldMap | None, shapes: dict | None = None
) -> None:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(states))
    rv = Reviewer(n, fps)
    win = "review"
    cv2.namedWindow(win)
    syncing = False

    def on_slider(pos: int) -> None:
        if not syncing:
            rv.seek(pos)
            rv.paused = True

    cv2.createTrackbar("frame", win, 0, max(n - 1, 1), on_slider)
    last_read = -2
    frame = None
    try:
        while not rv.done:
            if rv.idx != last_read:
                if rv.idx != last_read + 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, rv.idx)
                ok, frame = cap.read()
                last_read = rv.idx
                if not ok:
                    break
                syncing = True
                cv2.setTrackbarPos("frame", win, rv.idx)
                syncing = False
            state = states[rv.idx]
            cv2.imshow(win, render(frame, state, holdmap, fps, shapes))
            code = cv2.waitKeyEx(1 if rv.paused else max(int(1000 / fps), 1))
            rv.on_key(key_name(code))
            rv.tick()
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Frame-by-frame review of a clip with saved pose")
    parser.add_argument("--video", required=True)
    parser.add_argument("--cache", required=True, help="pose cache .jsonl for that video")
    parser.add_argument("--problem", help="hold map id: shows holds and recomputes contacts")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    states = load_states(Path(args.cache))
    holdmap, shapes = load_holdmap_with_shapes(args.problem) if args.problem else (None, {})
    if holdmap is not None:
        recompute_contacts(states, holdmap, shapes)
    log.info("%d frames loaded. Keys: space pause · a/d step · w/s ±1 s · q quit", len(states))
    run(Path(args.video), states, holdmap, shapes)


if __name__ == "__main__":
    main()
