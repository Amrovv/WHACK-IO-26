"""MediaPipe pose -> ClimberState (contacts are filled in later by contacts.py).

python -m coach.perception.pose --source file:data/footage/clip.mp4
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from coach.contracts import ClimberState, HoldMap, Limb
from coach.perception.config import REPO_ROOT, load_config
from coach.perception.contacts import LIMB_LANDMARKS, ContactTracker
from coach.perception.one_euro import OneEuroPoint

log = logging.getLogger(__name__)


# MediaPipe's 33 landmark names, in index order.
LANDMARK_NAMES = (
    "nose, left_eye_inner, left_eye, left_eye_outer, right_eye_inner, right_eye, "
    "right_eye_outer, left_ear, right_ear, mouth_left, mouth_right, left_shoulder, "
    "right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_pinky, "
    "right_pinky, left_index, right_index, left_thumb, right_thumb, left_hip, right_hip, "
    "left_knee, right_knee, left_ankle, right_ankle, left_heel, right_heel, "
    "left_foot_index, right_foot_index"
).split(", ")

# Landmarks that are relabelled together so a pair never ends up half swapped.
# Each group is decided by the mean x of its "left" vs "right" members.
SWAP_GROUPS: dict[str, tuple[str, ...]] = {
    "shoulder": ("shoulder",),
    "elbow": ("elbow",),
    "hand": ("wrist", "pinky", "index", "thumb"),
    "hip": ("hip",),
    "knee": ("knee",),
    "foot": ("ankle", "heel", "foot_index"),
}

SKELETON = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"),
    ("left_heel", "left_foot_index"),
    ("right_ankle", "right_heel"),
    ("right_heel", "right_foot_index"),
)

LIMB_COLOURS = {  # BGR
    Limb.LH: (255, 128, 0),
    Limb.RH: (0, 128, 255),
    Limb.LF: (255, 0, 200),
    Limb.RF: (0, 220, 0),
}

Landmarks = list[tuple[float, float, float]]  # 33 x (x, y, visibility), normalised
Detector = Callable[[np.ndarray, int], list[Landmarks]]  # (rgb, timestamp_ms) -> poses


def load_perception_config(path: str | Path | None = None) -> dict:
    return load_config(path)["perception"]


class MediapipeDetector:
    """PoseLandmarker in VIDEO mode. Downscales to max_width, converts BGR -> RGB."""

    def __init__(self, cfg: dict) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        model = Path(cfg["pose_model"])
        if not model.is_absolute():
            model = REPO_ROOT / model
        if not model.exists():
            raise FileNotFoundError(f"pose model missing: {model} (see models/README.md)")
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=cfg["num_poses"],
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._max_width = cfg["max_width"]

    def __call__(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[Landmarks]:
        h, w = frame_bgr.shape[:2]
        if w > self._max_width:
            frame_bgr = cv2.resize(
                frame_bgr, (self._max_width, round(h * self._max_width / w)), cv2.INTER_AREA
            )
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        return [[(lm.x, lm.y, lm.visibility) for lm in pose] for pose in result.pose_landmarks]

    def close(self) -> None:
        self._landmarker.close()


def visible_body_score(pose: Landmarks) -> float:
    """How much of a body is really in view: the visibility of the landmarks that are inside the
    frame, averaged over all of them (so landmarks outside the frame count as zero)."""
    inside = sum(q[2] for q in pose if 0.0 <= q[0] <= 1.0 and 0.0 <= q[1] <= 1.0)
    return inside / len(pose)


def _box_area(pose: Landmarks) -> float:
    xs, ys = [q[0] for q in pose], [q[1] for q in pose]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def pick_climber(poses: list[Landmarks]) -> Landmarks | None:
    """Choose the climber among the detected people: the one most fully in view.

    The biggest box is NOT enough: a bystander close to the camera and cropped by the frame edge
    has a bigger box than the climber on the wall, but only part of them is in view and the
    detector is unsure about it (low visibility). Measured on two clips (one with such a
    bystander), the visible-in-frame score alone separated the climber from everyone else; the
    box area is only a tie-break (two equally visible people: the bigger one).
    """
    return max(poses, key=lambda p: (visible_body_score(p), _box_area(p))) if poses else None


class LeftRightStabilizer:
    """Fixes left/right labels by image x (smaller x = left; the climber faces the wall).

    MediaPipe's own labels are usually right, so they are kept unless the image x order clearly
    disagrees with them: the two sides must be at least `min_gap` apart in x (normalised units),
    and the new order must hold for `swap_frames` consecutive frames before it is committed.
    When limbs are stacked at nearly the same x (e.g. reaching straight up), x order is noise
    and the current labelling is kept.
    """

    def __init__(self, swap_frames: int, min_gap: float) -> None:
        self.swap_frames = swap_frames
        self.min_gap = min_gap
        self.reset()

    def reset(self) -> None:
        self._swapped = {g: False for g in SWAP_GROUPS}
        self._streak = {g: 0 for g in SWAP_GROUPS}

    def apply(self, lms: dict[str, tuple[float, float, float]]) -> dict[str, tuple]:
        out = dict(lms)
        for group, parts in SWAP_GROUPS.items():
            lx = sum(lms[f"left_{p}"][0] for p in parts) / len(parts)
            rx = sum(lms[f"right_{p}"][0] for p in parts) / len(parts)
            gap = lx - rx  # > 0: raw "left" is to the right of raw "right" in the image
            # In which raw-label state would x order put left on the left? Undefined if stacked.
            wants_swap = gap > 0 if abs(gap) >= self.min_gap else self._swapped[group]
            if wants_swap == self._swapped[group]:
                self._streak[group] = 0
            else:
                self._streak[group] += 1
                if self._streak[group] >= self.swap_frames:
                    self._swapped[group] = wants_swap
                    self._streak[group] = 0
            if self._swapped[group]:
                for p in parts:
                    out[f"left_{p}"], out[f"right_{p}"] = lms[f"right_{p}"], lms[f"left_{p}"]
        return out


def _mean(a: tuple, b: tuple) -> tuple[float, float]:
    return (a[0] + b[0]) / 2, (a[1] + b[1]) / 2


def raw_limb_points(kp: dict) -> dict[Limb, tuple[float, float]]:
    """hand = mean(wrist, index) · foot = mean(heel, foot_index)."""
    return {
        Limb.LH: _mean(kp["left_wrist"], kp["left_index"]),
        Limb.RH: _mean(kp["right_wrist"], kp["right_index"]),
        Limb.LF: _mean(kp["left_heel"], kp["left_foot_index"]),
        Limb.RF: _mean(kp["right_heel"], kp["right_foot_index"]),
    }


class PoseTracker:
    """Frame in, ClimberState out. Stateful: smoothing, L/R stabilising, timestamps."""

    def __init__(
        self,
        cfg: dict,
        detector: Detector | None = None,
        holdmap: HoldMap | None = None,
        shapes: dict | None = None,
    ) -> None:
        self._detector = detector or MediapipeDetector(cfg)
        self._contacts = ContactTracker(cfg, holdmap, shapes) if holdmap else None
        eu = cfg["one_euro"]
        self._stabilizer = LeftRightStabilizer(cfg["lr_swap_frames"], cfg["lr_min_x_gap"])
        self._filters = {limb: OneEuroPoint(**eu) for limb in Limb}
        self._hip_filter = OneEuroPoint(**eu)
        self._t0: float | None = None
        self._last_ts = -1
        self._last_hip = (0.5, 0.5)
        self._min_visibility = cfg["min_visibility"]
        self._resume_visibility = cfg["resume_visibility"]
        self._resume_frames = cfg["resume_frames"]
        self._freeze_max_frames = cfg.get("freeze_max_frames", 15)
        self._good_run = {limb: 0 for limb in Limb}  # consecutive clearly-visible frames
        self._lost_memory = cfg["lost_memory_frames"]
        self._lost_frames = 0
        self._recent = {limb: deque(maxlen=cfg["hold_lookback_frames"]) for limb in Limb}
        self._hidden = {limb: False for limb in Limb}
        self._hidden_frames = {limb: 0 for limb in Limb}
        self._frozen: dict[Limb, tuple[float, float]] = {}

    def _timestamp_ms(self, t: float) -> int:
        self._t0 = t if self._t0 is None else self._t0
        ts = int((t - self._t0) * 1000)
        self._last_ts = max(ts, self._last_ts + 1)  # MediaPipe needs strictly increasing
        return self._last_ts

    def process(self, t: float, frame_idx: int, frame_bgr: np.ndarray) -> ClimberState:
        pose = pick_climber(self._detector(frame_bgr, self._timestamp_ms(t)))
        if pose is None:
            self._stabilizer.reset()
            self._lost_frames += 1
            if self._lost_frames > self._lost_memory:  # a long gap: forget the frozen limb points
                for limb in Limb:
                    self._recent[limb].clear()
                    self._hidden[limb] = False
                    self._good_run[limb] = 0
            for f in (*self._filters.values(), self._hip_filter):
                f.reset()
            return self._finish(
                ClimberState(
                    t=t,
                    frame_idx=frame_idx,
                    person_found=False,
                    keypoints={},
                    limbs={},
                    contacts={limb: None for limb in Limb},
                    hip=self._last_hip,
                )
            )
        self._lost_frames = 0
        kp = self._stabilizer.apply(dict(zip(LANDMARK_NAMES, pose, strict=True)))
        limbs = self._smooth_limbs(t, kp)
        hip = self._hip_filter(t, _mean(kp["left_hip"], kp["right_hip"]))
        self._last_hip = hip
        return self._finish(
            ClimberState(
                t=t,
                frame_idx=frame_idx,
                person_found=True,
                keypoints=kp,
                limbs=limbs,
                contacts={limb: None for limb in Limb},
                hip=hip,
            )
        )

    def _smooth_limbs(self, t: float, kp: dict) -> dict[Limb, tuple[float, float]]:
        """Smooth each limb point. While a limb is hidden (e.g. a hand behind the head) MediaPipe's
        guess is wrong, so the point is frozen instead of following it.

        A limb becomes hidden when its landmark visibility drops below `min_visibility`, and
        resumes only once it has stayed above `resume_visibility` for `resume_frames` frames in a
        row (so flickering visibility, or one confident-but-wrong frame, doesn't unfreeze it).
        It freezes at the position from `hold_lookback_frames` visible frames ago, because the
        guess usually starts drifting a little before visibility falls.
        """
        limbs = {}
        for limb, point in raw_limb_points(kp).items():
            vis = min(kp[name][2] for name in LIMB_LANDMARKS[limb])
            recent = self._recent[limb]
            self._good_run[limb] = self._good_run[limb] + 1 if vis >= self._resume_visibility else 0
            if self._hidden[limb] and self._good_run[limb] >= self._resume_frames:
                self._hidden[limb] = False
                self._hidden_frames[limb] = 0
                recent.clear()  # the filter is still catching up from the frozen point: start fresh
            elif not self._hidden[limb] and vis < self._min_visibility and recent:
                self._hidden[limb] = True
                self._hidden_frames[limb] = 0
                self._frozen[limb] = recent[0]  # oldest of the last few visible frames
            if self._hidden[limb]:
                self._hidden_frames[limb] += 1
                # Keep the frozen point through the configured short visibility dip, then stop
                # exposing it as a usable contact point while the pose remains hidden.
                if self._hidden_frames[limb] <= self._freeze_max_frames:
                    limbs[limb] = self._frozen[limb]
                # else: leave the limb out of `limbs` (the contract has no "missing" point)
            else:
                limbs[limb] = self._filters[limb](t, point)
                if vis >= self._min_visibility:  # never remember a low-visibility guess as reliable
                    recent.append(limbs[limb])
        return limbs

    def _finish(self, state: ClimberState) -> ClimberState:
        """Fill in hold contacts when a hold map is loaded (otherwise they stay None)."""
        if self._contacts is not None:
            state.contacts = self._contacts.update(state)
        return state

    def close(self) -> None:
        close = getattr(self._detector, "close", None)
        if close:
            close()


def _poly_px(poly, w: int, h: int) -> np.ndarray:
    """Normalised polygon -> pixel points in the shape cv2.polylines wants."""
    return np.array([(round(x * w), round(y * h)) for x, y in poly], np.int32)


def draw_overlay(
    frame: np.ndarray,
    state: ClimberState,
    video_t: float | None = None,
    holdmap: HoldMap | None = None,
    shapes: dict | None = None,
) -> np.ndarray:
    """Draw skeleton, limb dots and a frame/time label. `video_t` is seconds into the clip
    (defaults to state.t, which is video time for pose-cache states). With a hold map the
    holds are drawn too (a diagonal hold by its polygon) and each limb's hold is highlighted."""
    shapes = shapes or {}
    h, w = frame.shape[:2]

    def px(p) -> tuple[int, int]:
        return int(p[0] * w), int(p[1] * h)

    cv2.putText(
        frame,
        f"f{state.frame_idx}  {state.t if video_t is None else video_t:.2f}s",
        (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    if holdmap is not None:
        for hold in holdmap.holds:
            x0, y0, x1, y1 = (int(v * k) for v, k in zip(hold.bbox, (w, h, w, h), strict=True))
            if hold.id in shapes:
                cv2.polylines(frame, [_poly_px(shapes[hold.id], w, h)], True, (160, 160, 160), 1)
            else:
                cv2.rectangle(frame, (x0, y0), (x1, y1), (160, 160, 160), 1)
            cv2.putText(
                frame,
                str(hold.id),
                (x0, y0 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (160, 160, 160),
                1,
            )
        by_id = {hold.id: hold for hold in holdmap.holds}
        for limb, hold_id in state.contacts.items():  # highlight each limb's hold in its colour
            if hold_id in shapes:
                cv2.polylines(frame, [_poly_px(shapes[hold_id], w, h)], True, LIMB_COLOURS[limb], 3)
            elif hold_id in by_id:
                x0, y0, x1, y1 = (
                    int(v * k) for v, k in zip(by_id[hold_id].bbox, (w, h, w, h), strict=True)
                )
                cv2.rectangle(frame, (x0, y0), (x1, y1), LIMB_COLOURS[limb], 3)
    if not state.person_found:
        cv2.putText(frame, "no person", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame
    for a, b in SKELETON:
        cv2.line(frame, px(state.keypoints[a]), px(state.keypoints[b]), (200, 200, 200), 2)
    for limb, p in state.limbs.items():
        cv2.circle(frame, px(p), 9, LIMB_COLOURS[limb], -1)
        cv2.putText(frame, limb.name, px(p), cv2.FONT_HERSHEY_SIMPLEX, 0.5, LIMB_COLOURS[limb], 2)
    cv2.circle(frame, px(state.hip), 7, (0, 255, 255), -1)
    return frame


def main() -> None:
    from coach.sources import parse_source

    parser = argparse.ArgumentParser(description="Pose overlay demo")
    parser.add_argument("--source", required=True, help="cam:<n> or file:<path>")
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--problem", help="hold map id (data/holdmaps/<id>.json): fills contacts")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from coach.perception.holds import load_holdmap_with_shapes

    holdmap, shapes = load_holdmap_with_shapes(args.problem) if args.problem else (None, {})
    tracker = PoseTracker(load_perception_config(), holdmap=holdmap, shapes=shapes)
    source = parse_source(args.source)
    fps = getattr(source, "fps", None)  # file sources know their fps
    if not args.no_window:
        cv2.namedWindow("pose", cv2.WINDOW_NORMAL)  # resizable
    t_first = time.monotonic()
    n, found, t_start = 0, 0, t_first
    try:
        for t, idx, frame in source:
            state = tracker.process(t, idx, frame)
            n += 1
            found += state.person_found
            if n % 30 == 0:
                log.info(
                    "%.1f fps · person in %.0f%% of frames",
                    n / (time.monotonic() - t_start),
                    100 * found / n,
                )
            if not args.no_window:
                video_t = idx / fps if fps else t - t_first
                cv2.imshow("pose", draw_overlay(frame, state, video_t, holdmap, shapes))
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
        tracker.close()
        cv2.destroyAllWindows()
    if n:
        log.info(
            "done: %d frames, %.1f fps, person found in %.0f%%",
            n,
            n / (time.monotonic() - t_start),
            100 * found / n,
        )


if __name__ == "__main__":
    main()
