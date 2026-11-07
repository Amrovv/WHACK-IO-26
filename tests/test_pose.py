import json

import numpy as np
import pytest

from coach.contracts import Limb
from coach.perception.one_euro import OneEuroFilter, OneEuroPoint
from coach.perception.pose import (
    LANDMARK_NAMES,
    SWAP_GROUPS,
    LeftRightStabilizer,
    PoseTracker,
    largest_pose,
    load_perception_config,
)
from coach.perception.serialize import state_from_line, state_to_line

CFG = {
    "num_poses": 2,
    "lr_swap_frames": 5,
    "lr_min_x_gap": 0.03,
    "min_visibility": 0.5,
    "resume_visibility": 0.7,
    "resume_frames": 2,
    "hold_lookback_frames": 4,
    "lost_memory_frames": 5,
    "max_width": 640,
    "one_euro": {"min_cutoff": 1.0, "beta": 0.05, "d_cutoff": 1.0},
}
FRAME = np.zeros((48, 64, 3), np.uint8)


def make_pose(flipped=False, scale=1.0):
    """33 landmarks; every 'left_*' at x=0.4 and 'right_*' at x=0.6 (swapped if flipped)."""
    lms = []
    for i, name in enumerate(LANDMARK_NAMES):
        x = 0.5
        if name.startswith("left_"):
            x = 0.6 if flipped else 0.4
        elif name.startswith("right_"):
            x = 0.4 if flipped else 0.6
        y = 0.1 + 0.8 * i / 32
        lms.append((0.5 + (x - 0.5) * scale, 0.5 + (y - 0.5) * scale, 0.9))
    return lms


def test_config_has_pose_keys():
    cfg = load_perception_config()
    assert cfg["num_poses"] == 2 and cfg["lr_swap_frames"] == 5 and cfg["lr_min_x_gap"] > 0
    assert set(cfg["one_euro"]) == {"min_cutoff", "beta", "d_cutoff"}


def test_one_euro_first_sample_passthrough_and_smooths_jitter():
    f = OneEuroFilter(1.0, 0.0, 1.0)
    assert f(0.0, 5.0) == 5.0
    rng = np.random.default_rng(0)
    raw = 0.5 + rng.normal(0, 0.02, 200)
    out = [f(i / 30, v) for i, v in enumerate(raw)]
    assert np.std(out[30:]) < np.std(raw[30:]) / 2


def test_one_euro_ignores_non_increasing_time():
    f = OneEuroPoint(1.0, 0.05, 1.0)
    first = f(1.0, (0.2, 0.2))
    assert f(1.0, (0.9, 0.9)) == first


def hands_pose(left, right):
    """A pose whose left/right hand landmarks sit at the given points (others at defaults)."""
    lms = dict(zip(LANDMARK_NAMES, make_pose(), strict=True))
    for part in SWAP_GROUPS["hand"]:
        lms[f"left_{part}"] = (*left, 0.9)
        lms[f"right_{part}"] = (*right, 0.9)
    return lms


def stab():
    return LeftRightStabilizer(swap_frames=5, min_gap=0.03)


def test_stabilizer_swaps_only_after_n_stable_frames():
    s = stab()
    flipped = hands_pose((0.7, 0.4), (0.3, 0.4))  # raw "left" is far to the right
    for _ in range(4):
        out = s.apply(flipped)
        assert out["left_wrist"][0] > out["right_wrist"][0]  # not committed yet
    out = s.apply(flipped)  # 5th frame commits the swap
    assert out["left_wrist"][0] < out["right_wrist"][0]


def test_stabilizer_ignores_short_glitch_with_correct_raw_labels():
    # Like 5.7 s in the clip: raw labels are right (left hand on the left) except for a couple
    # of glitch frames where the hands momentarily look crossed. Labels must not flip.
    s = stab()
    normal = hands_pose((0.66, 0.4), (0.82, 0.45))
    glitch = hands_pose((0.80, 0.43), (0.62, 0.50))
    for i in range(40):
        out = s.apply(glitch if i in (10, 11) else normal)
        assert out["left_wrist"][0] < out["right_wrist"][0] or i in (10, 11)
    assert out["left_wrist"][0] == pytest.approx(0.66)


def test_stabilizer_ignores_x_order_when_hands_are_stacked():
    # Like 14 s in the clip: one hand straight above the other, x differs by ~0.01.
    s = stab()
    s.apply(hands_pose((0.40, 0.4), (0.60, 0.4)))
    for i in range(30):
        wobble = 0.01 if i % 2 else -0.01
        out = s.apply(hands_pose((0.655 + wobble, 0.37), (0.655 - wobble, 0.14)))
        assert out["left_wrist"][1] == pytest.approx(0.37)  # raw labels kept, never swapped
        assert out["right_wrist"][1] == pytest.approx(0.14)


def test_stabilizer_reset_reinitialises():
    s = stab()
    flipped = hands_pose((0.7, 0.4), (0.3, 0.4))
    for _ in range(5):
        s.apply(flipped)
    s.reset()
    out = s.apply(flipped)
    assert out["left_wrist"][0] > out["right_wrist"][0]  # back to raw until 5 frames


def test_largest_pose_picks_biggest_body():
    small, big = make_pose(scale=0.3), make_pose(scale=1.0)
    assert largest_pose([small, big]) is big
    assert largest_pose([]) is None


def test_tracker_builds_state_with_limb_points():
    tracker = PoseTracker(CFG, detector=lambda frame, ts: [make_pose()])
    s = tracker.process(0.0, 0, FRAME)
    assert s.person_found
    names = dict(zip(LANDMARK_NAMES, make_pose(), strict=True))
    lw, li = names["left_wrist"], names["left_index"]
    assert s.limbs[Limb.LH] == pytest.approx(((lw[0] + li[0]) / 2, (lw[1] + li[1]) / 2))
    lh, rh = names["left_hip"], names["right_hip"]
    assert s.hip == pytest.approx(((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2))
    assert all(v is None for v in s.contacts.values()) and len(s.contacts) == 4


def test_tracker_no_person_and_timestamps_strictly_increase():
    stamps = []

    def detector(frame, ts):
        stamps.append(ts)
        return []

    tracker = PoseTracker(CFG, detector=detector)
    for i in range(3):
        s = tracker.process(5.0, i, FRAME)  # identical capture times
    assert not s.person_found and s.limbs == {}
    assert stamps == sorted(set(stamps)) and len(stamps) == 3


def test_state_json_roundtrip_uses_string_limb_keys():
    tracker = PoseTracker(CFG, detector=lambda frame, ts: [make_pose()])
    s = tracker.process(1.5, 7, FRAME)
    s.contacts[Limb.RH] = 4
    line = state_to_line(s)
    d = json.loads(line)
    assert set(d["limbs"]) == {"left_hand", "right_hand", "left_foot", "right_foot"}
    assert d["contacts"]["right_hand"] == 4 and d["contacts"]["left_hand"] is None
    back = state_from_line(line)
    assert back == s


def test_a_hidden_limb_is_held_at_its_last_reliable_position_then_follows_again():
    # Like 16 s in the clip: the right hand goes behind the head, visibility collapses and
    # MediaPipe's guess for it is wrong. The dot must stay where it was last reliable.
    def pose(right_wrist_xy, vis):
        lms = make_pose()
        for name in ("right_wrist", "right_index"):
            lms[LANDMARK_NAMES.index(name)] = (*right_wrist_xy, vis)
        return lms

    frames = [((0.60, 0.14), 0.9)] * 5 + [((0.64, 0.25), 0.1)] * 5 + [((0.70, 0.20), 0.9)] * 15
    it = iter(frames)
    tracker = PoseTracker(CFG, detector=lambda f, ts: [pose(*next(it))])
    states = [tracker.process(i / 30, i, FRAME) for i in range(len(frames))]
    good = states[4].limbs[Limb.RH]
    assert all(s.limbs[Limb.RH] == good for s in states[5:10])  # frozen, ignores the bad guess
    assert states[-1].limbs[Limb.RH] == pytest.approx((0.70, 0.20), abs=0.02)  # follows again


def test_a_limb_hidden_from_the_start_still_gets_a_point():
    lms = make_pose()
    for name in ("right_wrist", "right_index"):
        lms[LANDMARK_NAMES.index(name)] = (0.6, 0.2, 0.1)
    tracker = PoseTracker(CFG, detector=lambda f, ts: [lms])
    assert tracker.process(0.0, 0, FRAME).limbs[Limb.RH] == pytest.approx((0.6, 0.2))


def _rh_pose(xy, vis):
    lms = make_pose()
    for name in ("right_wrist", "right_index"):
        lms[LANDMARK_NAMES.index(name)] = (*xy, vis)
    return lms


def test_frozen_limb_ignores_flickering_visibility_and_resumes_only_when_clearly_back():
    frames = (
        [((0.60, 0.14), 0.9)] * 6
        + [((0.60, 0.30), 0.2)] * 2  # hidden: the guess is wrong
        + [((0.60, 0.30), 0.6)] * 3  # flickers above min_visibility (0.5) but below resume (0.7)
        + [((0.65, 0.16), 0.9)] * 20  # clearly visible again
    )
    it = iter(frames)
    tracker = PoseTracker(CFG, detector=lambda f, ts: [_rh_pose(*next(it))])
    states = [tracker.process(i / 30, i, FRAME) for i in range(len(frames))]
    frozen = states[6].limbs[Limb.RH]
    assert all(s.limbs[Limb.RH] == frozen for s in states[6:11])  # stays frozen through the flicker
    assert states[-1].limbs[Limb.RH] == pytest.approx((0.65, 0.16), abs=0.02)


def test_limb_freezes_at_the_position_from_a_few_frames_before_it_went_hidden():
    # The point drifts (0.14 -> 0.25) while visibility is still fine, then visibility drops.
    # It should freeze at the early, correct position, not the drifted one.
    ys = [0.14, 0.14, 0.14, 0.14, 0.18, 0.25]
    frames = [((0.60, y), 0.9) for y in ys] + [((0.60, 0.30), 0.1)] * 3
    it = iter(frames)
    tracker = PoseTracker(CFG, detector=lambda f, ts: [_rh_pose(*next(it))])  # lookback = 4
    states = [tracker.process(i / 30, i, FRAME) for i in range(len(frames))]
    assert states[-1].limbs[Limb.RH][1] < 0.17  # near 0.14, not 0.25


def test_a_brief_person_loss_does_not_make_a_hidden_limb_follow_a_bad_guess():
    # Like f642-f678 in the clip: the hand is last seen well, the person is lost for a moment,
    # and when they are found again the hand is hidden and MediaPipe guesses badly.
    good = ((0.60, 0.14), 0.9)
    bad = ((0.58, 0.30), 0.1)
    frames = [good] * 5 + [None] * 2 + [bad] * 6
    it = iter(frames)

    def detector(f, ts):
        item = next(it)
        return [] if item is None else [_rh_pose(*item)]

    tracker = PoseTracker(CFG, detector=detector)
    states = [tracker.process(i / 30, i, FRAME) for i in range(len(frames))]
    assert not states[5].person_found
    for s in states[7:]:
        assert s.limbs[Limb.RH] == pytest.approx((0.60, 0.14), abs=0.01)  # held, not the bad guess


def test_a_long_person_loss_forgets_the_frozen_point():
    good = ((0.60, 0.14), 0.9)
    bad = ((0.58, 0.30), 0.1)
    frames = [good] * 5 + [None] * 8 + [bad] * 3  # gap of 8 > lost_memory_frames (5)
    it = iter(frames)

    def detector(f, ts):
        item = next(it)
        return [] if item is None else [_rh_pose(*item)]

    tracker = PoseTracker(CFG, detector=detector)
    states = [tracker.process(i / 30, i, FRAME) for i in range(len(frames))]
    assert states[-1].limbs[Limb.RH] == pytest.approx((0.58, 0.30), abs=0.01)  # no memory left


def test_after_resuming_the_lookback_uses_fresh_points_not_the_old_frozen_one():
    # hidden at an old position (0.10), resumes where the hand really is (0.60), then hides again:
    # it must freeze near 0.60 and not fall back to a transition value near the old 0.10.
    far, near = ((0.50, 0.10), 0.9), ((0.60, 0.40), 0.9)
    frames = (
        [far] * 4
        + [((0.50, 0.50), 0.1)] * 2  # hidden
        + [near] * 20  # resumes and settles at the new position (the smoothing needs time)
        + [((0.58, 0.90), 0.1)] * 3  # hidden again, bad guess
    )
    it = iter(frames)
    tracker = PoseTracker(CFG, detector=lambda f, ts: [_rh_pose(*next(it))])  # lookback = 4
    states = [tracker.process(i / 30, i, FRAME) for i in range(len(frames))]
    assert states[-1].limbs[Limb.RH][1] == pytest.approx(0.40, abs=0.1)  # not near 0.10 or 0.90
