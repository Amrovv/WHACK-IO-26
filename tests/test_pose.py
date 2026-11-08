import json

import numpy as np
import pytest

from coach.contracts import Limb
from coach.perception.one_euro import OneEuroFilter, OneEuroPoint
from coach.perception.pose import (
    LANDMARK_NAMES,
    SWAP_GROUPS,
    ArmModel,
    LeftRightStabilizer,
    PoseTracker,
    load_perception_config,
    pick_climber,
)
from coach.perception.serialize import state_from_line, state_to_line

CFG = {
    **load_perception_config(),  # the real settings (so new keys are there), this test's on top
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


def test_pick_climber_prefers_the_bigger_of_two_fully_visible_people():
    small, big = make_pose(scale=0.3), make_pose(scale=1.0)
    assert pick_climber([small, big]) is big
    assert pick_climber([]) is None


def cropped_bystander():
    """A big person at the frame edge: a bigger box than the climber, but half of them is outside
    the frame and the detector is unsure about the rest (the bystander in Climb 2)."""
    lms = make_pose(scale=2.0)
    return [(0.9 + 0.5 * (x - 0.5), 0.9 + 0.8 * (y - 0.5), 0.45) for x, y, _ in lms]


def test_pick_climber_ignores_a_larger_but_cropped_low_visibility_bystander():
    climber = make_pose(scale=0.4)  # smaller box, fully in view, confident
    bystander = cropped_bystander()
    xs, ys = [q[0] for q in bystander], [q[1] for q in bystander]
    cx = [q[0] for q in climber]
    cy = [q[1] for q in climber]
    assert (max(xs) - min(xs)) * (max(ys) - min(ys)) > (max(cx) - min(cx)) * (max(cy) - min(cy))
    assert any(not 0 <= x <= 1 or not 0 <= y <= 1 for x, y, _ in bystander)  # really cropped
    assert pick_climber([bystander, climber]) is climber
    assert pick_climber([climber, bystander]) is climber  # order does not matter


def test_pick_climber_keeps_a_single_cropped_person():
    # If the climber is the only person detected (even partly cropped at the top), use them.
    only = cropped_bystander()
    assert pick_climber([only]) is only


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


def test_frozen_limb_point_is_kept_up_to_the_cap_then_withheld_and_returns_on_resume():
    # The freeze cap (hands placed from the elbow have their own: see the ArmModel tests).
    cfg = {**CFG, "freeze_max_frames": 3, "hidden_hand": "freeze"}
    tracker = PoseTracker(cfg, detector=lambda _frame, _ts: [])
    good = hands_pose((0.4, 0.5), (0.6, 0.3))
    for i in range(6):
        tracker._smooth_limbs(i / 30, good)

    hidden = dict(good)
    for part in SWAP_GROUPS["hand"]:
        hidden[f"right_{part}"] = (0.9, 0.9, 0.3)
    early = [tracker._smooth_limbs((6 + i) / 30, hidden).get(Limb.RH) for i in range(3)]
    assert all(point is not None for point in early)
    assert Limb.RH not in tracker._smooth_limbs(
        9 / 30, hidden
    )  # over the cap: left out, never None
    assert Limb.RH not in tracker._smooth_limbs(10 / 30, hidden)

    visible = dict(good)
    for part in SWAP_GROUPS["hand"]:
        visible[f"right_{part}"] = (0.7, 0.4, 0.9)
    assert Limb.RH not in tracker._smooth_limbs(11 / 30, visible)
    resumed = tracker._smooth_limbs(12 / 30, visible)[Limb.RH]
    assert resumed is not None


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


def test_a_hidden_limb_keeps_its_point_for_the_default_three_seconds():
    """A hand hidden by the body while resting on a hold must still count (top-out detection)."""
    cfg = {**CFG, "freeze_max_frames": 90, "hidden_hand": "freeze"}
    tracker = PoseTracker(cfg, detector=lambda _frame, _ts: [])
    good = hands_pose((0.4, 0.5), (0.6, 0.3))
    for i in range(6):
        tracker._smooth_limbs(i / 30, good)
    lost, murky = dict(good), dict(good)
    for part in SWAP_GROUPS["hand"]:
        lost[f"right_{part}"] = (0.9, 0.9, 0.3)  # visibility under 0.4: the limb freezes
        murky[f"right_{part}"] = (0.9, 0.9, 0.5)  # 0.4-0.7: still frozen, but counts as visible
    assert Limb.RH in tracker._smooth_limbs(6 / 30, lost)
    for i in range(1, 90):
        assert Limb.RH in tracker._smooth_limbs((6 + i) / 30, murky)
    assert Limb.RH not in tracker._smooth_limbs(96 / 30, murky)  # frame 91 frozen: over the cap


# ---- Hidden hands placed from the elbow (ArmModel) ------------------------------------------


def test_arm_model_keeps_a_reachable_hand_and_follows_the_elbow_once_it_is_not():
    arm = ArmModel(CFG)  # aspect 1: the normalised shape is the real one
    for _ in range(5):  # upper arm straight down, forearm out to the right: bent 90 degrees
        arm.learn((0.5, 0.2), (0.5, 0.35), (0.65, 0.35))
    assert arm.forearm == pytest.approx(0.15)

    gripping = (0.65, 0.35)
    assert arm.place(gripping, (0.5, 0.2), (0.5, 0.36)) == gripping  # still in reach: kept

    # The body swings 0.2 left: the old point is out of reach, so the hand follows the elbow at
    # the learnt 90 degree bend (forearm out to the right of the upper arm again).
    x, y = arm.place(gripping, (0.3, 0.2), (0.3, 0.35))
    assert (x, y) == pytest.approx((0.45, 0.35), abs=1e-6)
    assert arm.following
    assert arm.place(gripping, (0.3, 0.2), (0.3, 0.35)) == pytest.approx((0.45, 0.35))

    arm.learn((0.3, 0.2), (0.3, 0.35), (0.45, 0.35))  # seen again: back to keeping
    assert not arm.following


def test_freeze_mode_and_an_unknown_arm_keep_the_frozen_point():
    assert ArmModel({**CFG, "hidden_hand": "freeze"}).place((0.1, 0.1), (0.5, 0.2), (0.5, 0.3)) == (
        0.1,
        0.1,
    )
    assert ArmModel(CFG).place((0.1, 0.1), (0.5, 0.2), (0.5, 0.3)) == (0.1, 0.1)  # nothing learnt


def arm_pose(shoulder, elbow, hand, hand_vis=0.9):
    """A pose with the right arm at these points (left arm and the rest at defaults)."""
    lms = dict(zip(LANDMARK_NAMES, make_pose(), strict=True))
    lms["right_shoulder"], lms["right_elbow"] = (*shoulder, 0.9), (*elbow, 0.9)
    for part in SWAP_GROUPS["hand"]:
        lms[f"right_{part}"] = (*hand, hand_vis)
    return lms


def test_a_hand_hidden_behind_the_body_follows_its_elbow_without_jumping():
    tracker = PoseTracker(CFG, detector=lambda _frame, _ts: [])
    t = 0.0
    for _ in range(10):  # in view: the arm's shape is learnt
        tracker._smooth_limbs(t, arm_pose((0.6, 0.2), (0.6, 0.35), (0.75, 0.35)))
        t += 1 / 30
    hidden_at = tracker._smooth_limbs(t, arm_pose((0.6, 0.2), (0.6, 0.35), (0.5, 0.5), 0.1))
    assert hidden_at[Limb.RH] == pytest.approx((0.75, 0.35), abs=0.01)  # not the bad guess

    # The climber moves 0.25 left with the hand still hidden: the hand goes with the elbow,
    # gliding (never more than the speed limit per frame), to forearm length right of it.
    points = []
    for _ in range(40):
        t += 1 / 30
        out = tracker._smooth_limbs(t, arm_pose((0.35, 0.2), (0.35, 0.35), (0.5, 0.5), 0.1))
        points.append(np.array(out[Limb.RH]))
    steps = [np.linalg.norm(b - a) for a, b in zip(points, points[1:], strict=False)]
    assert max(steps) <= CFG["hand_estimate_max_speed"] / 30 + 1e-6
    assert points[-1] == pytest.approx((0.50, 0.35), abs=0.02)


def test_a_reappearing_hand_out_of_reach_of_its_elbow_is_not_believed():
    tracker = PoseTracker(CFG, detector=lambda _frame, _ts: [])
    t = 0.0
    for _ in range(10):
        tracker._smooth_limbs(t, arm_pose((0.6, 0.2), (0.6, 0.35), (0.75, 0.35)))
        t += 1 / 30
    tracker._smooth_limbs(t, arm_pose((0.6, 0.2), (0.6, 0.35), (0.5, 0.5), 0.1))
    for _ in range(5):  # "clearly visible" again, but 0.4 from an elbow with a 0.15 forearm
        t += 1 / 30
        out = tracker._smooth_limbs(t, arm_pose((0.6, 0.2), (0.6, 0.35), (0.2, 0.35)))
        assert tracker._hidden[Limb.RH] and out[Limb.RH][0] > 0.6
    for _ in range(3):  # back where a hand can be: it's believed
        t += 1 / 30
        tracker._smooth_limbs(t, arm_pose((0.6, 0.2), (0.6, 0.35), (0.72, 0.3)))
    assert not tracker._hidden[Limb.RH]


def test_a_hand_that_stays_in_view_is_believed_even_if_the_arm_model_disagrees():
    """The arm model can be wrong (a forearm learnt foreshortened): a hand MediaPipe keeps seeing
    clearly must come back after resume_frames + hand_resume_patience_frames, not stay lost."""
    tracker = PoseTracker(CFG, detector=lambda _frame, _ts: [])
    t = 0.0
    for _ in range(10):
        tracker._smooth_limbs(t, arm_pose((0.6, 0.2), (0.6, 0.35), (0.75, 0.35)))
        t += 1 / 30
    tracker._smooth_limbs(t, arm_pose((0.6, 0.2), (0.6, 0.35), (0.5, 0.5), 0.1))
    frames = 0
    while tracker._hidden[Limb.RH] and frames < 30:
        t += 1 / 30
        frames += 1
        out = tracker._smooth_limbs(t, arm_pose((0.6, 0.2), (0.6, 0.35), (0.2, 0.35)))
    assert frames == CFG["resume_frames"] + CFG["hand_resume_patience_frames"]
    # ...and it's shown where it's seen right away, not dragged across by the smoothing.
    assert out[Limb.RH] == pytest.approx((0.2, 0.35), abs=0.01)
