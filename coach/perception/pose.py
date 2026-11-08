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

HAND_SIDES = {Limb.LH: "left", Limb.RH: "right"}
Point = tuple[float, float]


def _rotate(v: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    """Signed angle from a to b."""
    return float(np.arctan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]))


class ArmModel:
    """One arm's shape, learnt while its hand is in view, to place the hand while it isn't.

    A hand behind the chest (or the head) vanishes, but MediaPipe still sees that shoulder and
    elbow. While the hand is in view the model keeps the forearm's length and its angle to the
    upper arm (medians over `arm_model_frames`). While it's hidden:
      * the hand stays where it was (it's usually gripping a hold) as long as that point is still
        within reach: forearm length from the elbow and arm length from the shoulder, each with
        `hand_reach_slack`;
      * once it's out of reach the arm has moved, so the hand follows the elbow: forearm length
        from it, at the learnt angle to the upper arm.
    Lengths and angles use x scaled by the frame's aspect ratio (the real shape, not the
    normalised one).
    """

    def __init__(self, cfg: dict) -> None:
        self.mode = cfg["hidden_hand"]  # "freeze" | "keep_or_rotate" | "keep_or_project" | ...
        self.slack = cfg["hand_reach_slack"]
        self.length_percentile = cfg["arm_length_percentile"]
        self._forearm: deque[float] = deque(maxlen=cfg["arm_model_frames"])
        self._upper: deque[float] = deque(maxlen=cfg["arm_model_frames"])
        self._bend: deque[float] = deque(maxlen=cfg["arm_model_frames"])
        self.aspect = 1.0  # frame width / height
        self.following = False  # out of reach once: the hand follows the elbow until it's seen

    @property
    def known(self) -> bool:
        return len(self._forearm) > 0

    @property
    def forearm(self) -> float:
        """The forearm's length: a high percentile, since a forearm pointing at the camera only
        ever looks shorter than it is."""
        return float(np.percentile(self._forearm, self.length_percentile))

    def _iso(self, p: Point) -> np.ndarray:
        return np.array([p[0] * self.aspect, p[1]])

    def distance(self, a: Point, b: Point) -> float:
        """Distance in frame heights (x scaled by the aspect ratio)."""
        return float(np.linalg.norm(self._iso(a) - self._iso(b)))

    def _norm(self, v: np.ndarray) -> Point:
        return float(v[0] / self.aspect), float(v[1])

    def learn(self, shoulder: Point, elbow: Point, hand: Point) -> None:
        s, e, h = self._iso(shoulder), self._iso(elbow), self._iso(hand)
        upper, fore = e - s, h - e
        if np.linalg.norm(upper) < 1e-6 or np.linalg.norm(fore) < 1e-6:
            return
        self._forearm.append(float(np.linalg.norm(fore)))
        self._upper.append(float(np.linalg.norm(upper)))
        self._bend.append(_angle(upper, fore))
        self.following = False

    def reachable(self, hand: Point, shoulder: Point, elbow: Point) -> bool:
        """Could the hand be at this point, given where the shoulder and elbow are?"""
        s, e, h = self._iso(shoulder), self._iso(elbow), self._iso(hand)
        arm = float(np.percentile(self._upper, self.length_percentile)) + self.forearm
        return (
            np.linalg.norm(h - e) <= self.forearm * self.slack
            and np.linalg.norm(h - s) <= arm * self.slack
        )

    def place(self, frozen: Point, shoulder: Point, elbow: Point) -> Point:
        """Where the hidden hand is, from where it was last seen and where the arm is now."""
        if self.mode == "freeze" or not self.known:
            return frozen
        keep = self.mode.startswith("keep_or_")
        if keep and not self.following and self.reachable(frozen, shoulder, elbow):
            return frozen
        self.following = True
        s, e, f = self._iso(shoulder), self._iso(elbow), self._iso(frozen)
        how = self.mode.removeprefix("keep_or_")
        if how == "project":  # forearm length from the elbow, as near the old point as can be
            v = f - e
            n = np.linalg.norm(v)
            return self._norm(e + v / n * self.forearm) if n > 1e-6 else frozen
        upper = e - s  # "rotate": the learnt bend, carried round with the upper arm
        n = np.linalg.norm(upper)
        if n < 1e-6:
            return frozen
        bend = float(np.median(self._bend))
        return self._norm(e + _rotate(upper / n, bend) * self.forearm)


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
        self._arms = {limb: ArmModel(cfg) for limb in HAND_SIDES}  # hidden hands, from the elbow
        self._hand_max_frames = cfg["hand_estimate_max_frames"]
        self._hand_max_speed = cfg["hand_estimate_max_speed"]
        self._resume_reach = cfg["hand_resume_reach"]
        self._resume_patience = cfg["hand_resume_patience_frames"]
        self._resume_snap = cfg["hand_resume_snap"]
        self._shown: dict[Limb, tuple[float, Point]] = {}  # each limb's last output (t, point)

    def _timestamp_ms(self, t: float) -> int:
        self._t0 = t if self._t0 is None else self._t0
        ts = int((t - self._t0) * 1000)
        self._last_ts = max(ts, self._last_ts + 1)  # MediaPipe needs strictly increasing
        return self._last_ts

    def process(self, t: float, frame_idx: int, frame_bgr: np.ndarray) -> ClimberState:
        h, w = frame_bgr.shape[:2]
        for arm in self._arms.values():
            arm.aspect = w / h if h else 1.0
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

        A hidden hand is placed by its ArmModel from the shoulder and elbow while those are in
        view (kept where it was while that's within reach, otherwise following the elbow), for
        up to `hand_estimate_max_frames`. A hand only comes back into view at a point within
        `hand_resume_reach` forearm lengths of the elbow: MediaPipe's first guesses as a hand
        reappears can be anywhere.
        """
        limbs = {}
        for limb, point in raw_limb_points(kp).items():
            vis = min(kp[name][2] for name in LIMB_LANDMARKS[limb])
            recent = self._recent[limb]
            arm, shoulder, elbow = self._arm(limb, kp)
            self._good_run[limb] = self._good_run[limb] + 1 if vis >= self._resume_visibility else 0
            # A reappearing hand out of its elbow's reach waits; one that stays in view anyway is
            # believed after `hand_resume_patience_frames` more (the arm model can be wrong).
            patient = self._good_run[limb] >= self._resume_frames + self._resume_patience
            if (
                self._hidden[limb]
                and self._good_run[limb] >= self._resume_frames
                and (patient or self._plausible(arm, point, elbow))
            ):
                self._hidden[limb] = False
                self._hidden_frames[limb] = 0
                recent.clear()  # the filter is still catching up from the frozen point: start fresh
                shown = self._shown.get(limb)
                if (
                    arm is not None
                    and shown is not None
                    and arm.distance(shown[1], point) > (self._resume_snap)
                ):
                    # Back in view far from where it was placed: the hand is where MediaPipe
                    # sees it (within reach, checked above), so go there now rather than lag.
                    self._filters[limb].reset()
            elif not self._hidden[limb] and vis < self._min_visibility and recent:
                self._hidden[limb] = True
                self._hidden_frames[limb] = 0
                self._frozen[limb] = recent[0]  # oldest of the last few visible frames
            if self._hidden[limb]:
                self._hidden_frames[limb] += 1
                n = self._hidden_frames[limb]
                estimating = arm is not None and elbow is not None and arm.mode != "freeze"
                if estimating and n <= self._hand_max_frames:
                    target = arm.place(self._frozen[limb], shoulder, elbow)
                    # Through the same smoothing as a hand in view, and no faster than
                    # `hand_estimate_max_speed`: elbow jitter (and the switch from kept to
                    # following) mustn't make the hand jump, and when the hand reappears the
                    # filter has already caught up with it.
                    limbs[limb] = self._filters[limb](t, self._step_towards(arm, limb, t, target))
                elif n <= self._freeze_max_frames:
                    # Keep the frozen point through the configured short visibility dip, then
                    # stop exposing it as a usable contact point while the pose remains hidden.
                    # A hand that was following its elbow stays where it was last shown while
                    # the elbow drops out of view (rather than jumping back to the frozen point).
                    following = arm is not None and arm.following and limb in self._shown
                    limbs[limb] = self._shown[limb][1] if following else self._frozen[limb]
                # else: leave the limb out of `limbs` (the contract has no "missing" point)
            else:
                limbs[limb] = self._filters[limb](t, point)
                if vis >= self._min_visibility:  # never remember a low-visibility guess as reliable
                    recent.append(limbs[limb])
                if arm is not None and elbow is not None and vis >= self._resume_visibility:
                    arm.learn(shoulder, elbow, point)  # clearly in view: learn the arm's shape
            if limb in limbs:
                self._shown[limb] = (t, limbs[limb])
        return limbs

    def _step_towards(self, arm: ArmModel, limb: Limb, t: float, target: Point) -> Point:
        """Move the shown hand towards `target` no faster than `hand_estimate_max_speed` (frame
        heights per second, in the real aspect)."""
        shown = self._shown.get(limb)
        if shown is None:
            return target
        t0, p0 = shown
        a, b = arm._iso(p0), arm._iso(target)
        step = np.linalg.norm(b - a)
        reach = self._hand_max_speed * max(t - t0, 0.0)
        if step <= reach or step < 1e-9:
            return target
        return arm._norm(a + (b - a) * (reach / step))

    def _arm(self, limb: Limb, kp: dict) -> tuple[ArmModel | None, Point | None, Point | None]:
        """A hand's ArmModel, with its shoulder and elbow when both are in view (else None)."""
        side = HAND_SIDES.get(limb)
        if side is None:
            return None, None, None
        shoulder, elbow = kp[f"{side}_shoulder"], kp[f"{side}_elbow"]
        if min(shoulder[2], elbow[2]) < self._min_visibility:
            return self._arms[limb], None, None
        return self._arms[limb], shoulder[:2], elbow[:2]

    def _plausible(self, arm: ArmModel | None, point: Point, elbow: Point | None) -> bool:
        """A returning hand must be within reach of the elbow (when that can be checked)."""
        if arm is None or elbow is None or not arm.known:
            return True
        reach = np.linalg.norm(arm._iso(point) - arm._iso(elbow))
        return bool(reach <= arm.forearm * self._resume_reach)

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
    transcript_group = parser.add_mutually_exclusive_group()
    transcript_group.add_argument("--transcript", dest="transcript", action="store_true")
    transcript_group.add_argument("--no-transcript", dest="transcript", action="store_false")
    parser.set_defaults(transcript=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from coach.perception.holds import load_holdmap_with_shapes

    holdmap, shapes = load_holdmap_with_shapes(args.problem) if args.problem else (None, {})
    transcript_enabled = args.transcript if args.transcript is not None else bool(args.problem)
    engine = None
    engine_cost_ms = 0.0
    show_transcript = True
    max_lines = 25
    if transcript_enabled and holdmap is not None:
        from coach.perception.transcript import TranscriptEngine, draw_panel, format_line

        engine = TranscriptEngine(load_config(), holdmap)
    elif transcript_enabled:
        log.info("Transcript needs --problem; skipping it")
    tracker = PoseTracker(load_perception_config(), holdmap=holdmap, shapes=shapes)
    source = parse_source(args.source)
    fps = getattr(source, "fps", None)  # file sources know their fps
    if not args.no_window:
        cv2.namedWindow("pose", cv2.WINDOW_NORMAL)  # resizable
        window_sized = False
    t_first = time.monotonic()
    n, found, t_start = 0, 0, t_first
    try:
        for t, idx, frame in source:
            state = tracker.process(t, idx, frame)
            started = time.perf_counter()
            entries = engine.update(state) if engine else []
            if engine:
                for entry in entries:
                    log.info("%s", format_line(entry, engine.t0 or 0.0))
                engine_cost_ms += (time.perf_counter() - started) * 1000
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
                image = draw_overlay(frame, state, video_t, holdmap, shapes)
                if engine and show_transcript:
                    panel = draw_panel(
                        engine.entries,
                        state.t,
                        image.shape[0],
                        max_lines=max_lines,
                        t0=engine.t0 or 0.0,
                    )
                    image = np.hstack((image, panel))
                if not window_sized:  # once: fit the window to the screen, keeping proportions
                    fit = min(1.0, 800 / image.shape[0], 1400 / image.shape[1])
                    cv2.resizeWindow(
                        "pose", round(image.shape[1] * fit), round(image.shape[0] * fit)
                    )
                    window_sized = True
                cv2.imshow("pose", image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("t") and engine:
                    show_transcript = not show_transcript
                if key in (27, ord("q")):
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
    if engine:
        log.info("transcript engine: %.3f ms/frame average", engine_cost_ms / max(n, 1))


if __name__ == "__main__":
    main()
