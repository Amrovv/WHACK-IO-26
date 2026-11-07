import numpy as np
import pytest

from coach.contracts import ClimberState, Hold, HoldMap, Limb
from coach.perception.contacts import ContactTracker
from coach.perception.pose import LANDMARK_NAMES, PoseTracker

CFG = {
    "contact_margin": 0.15,
    "contact_settle_ms": 150,
    "contact_release_ratio": 1.3,
    "contact_still_speed": 0.2,
    "min_visibility": 0.5,
    "resume_visibility": 0.7,
    "resume_frames": 2,
    "hold_lookback_frames": 4,
    "lost_memory_frames": 5,
    "arm_length_frames": 60,
    "arm_length_default": 0.25,
    "num_poses": 2,
    "lr_swap_frames": 5,
    "lr_min_x_gap": 0.03,
    "max_width": 640,
    "one_euro": {"min_cutoff": 1.0, "beta": 0.05, "d_cutoff": 1.0},
}
DT = 1 / 30
# arm = shoulder->elbow 0.15 + elbow->wrist 0.15 = 0.30, so margin = 0.15 * 0.30 = 0.045
# and the release margin = 0.045 * 1.3 = 0.0585
H1 = Hold(1, (0.45, 0.45), (0.40, 0.40, 0.50, 0.50))
H2 = Hold(2, (0.57, 0.45), (0.52, 0.40, 0.62, 0.50))
HOLDMAP = HoldMap("test", "yellow", [H1, H2], (100, 100), "manual")


def make_state(t, limbs=None, vis=1.0, found=True):
    limbs = limbs or {}
    kp = {name: (0.5, 0.5, vis) for name in LANDMARK_NAMES}
    for side in ("left", "right"):
        kp[f"{side}_shoulder"] = (0.5, 0.40, vis)
        kp[f"{side}_elbow"] = (0.5, 0.55, vis)
        kp[f"{side}_wrist"] = (0.5, 0.70, vis)
    return ClimberState(
        t=t,
        frame_idx=round(t / DT),
        person_found=found,
        keypoints=kp if found else {},
        limbs=limbs if found else {},
        contacts={limb: None for limb in Limb},
        hip=(0.5, 0.5),
    )


def hold_still(tracker, point, seconds, t0=0.0, limb=Limb.RH, vis=1.0):
    """Feed a still limb for `seconds`; returns [(t, contact)]."""
    out = []
    for i in range(int(seconds / DT) + 1):
        t = t0 + i * DT
        out.append((t, tracker.update(make_state(t, {limb: point}, vis))[limb]))
    return out


def test_contact_appears_only_after_settle_time():
    tr = ContactTracker(CFG, HOLDMAP)
    for t, contact in hold_still(tr, (0.45, 0.45), 0.4):
        assert contact == (1 if t >= 0.15 else None), f"t={t:.3f}"


def test_fast_moving_limb_inside_a_hold_gets_no_contact_until_it_stops():
    tr = ContactTracker(CFG, HOLDMAP)
    for i in range(10):  # sweeps across the hold at ~0.6 units/s
        x = 0.41 + 0.01 * i
        assert tr.update(make_state(i * DT, {Limb.RH: (x, 0.45)}))[Limb.RH] is None
    results = hold_still(tr, (0.49, 0.45), 0.3, t0=10 * DT)
    assert results[-1][1] == 1


def test_a_limb_outside_every_hold_has_no_contact():
    tr = ContactTracker(CFG, HOLDMAP)
    assert all(c is None for _, c in hold_still(tr, (0.2, 0.9), 0.5))


def test_hysteresis_keeps_contact_in_the_release_zone_but_not_beyond():
    tr = ContactTracker(CFG, HOLDMAP)
    hold_still(tr, (0.45, 0.45), 0.3)
    # x=0.55: outside the enter box (0.50+0.045) but inside the release box (0.50+0.0585)
    assert tr.update(make_state(0.31, {Limb.RH: (0.54, 0.45)}))[Limb.RH] == 1
    assert tr.update(make_state(0.34, {Limb.RH: (0.5575, 0.45)}))[Limb.RH] == 1
    assert tr.update(make_state(0.37, {Limb.RH: (0.45, 0.70)}))[Limb.RH] is None  # clearly away


def test_a_fresh_limb_in_the_release_zone_does_not_make_contact():
    tr = ContactTracker(CFG, HOLDMAP)
    assert all(c is None for _, c in hold_still(tr, (0.45, 0.5525), 0.4))


def test_two_qualifying_holds_nearest_centre_wins():
    # x=0.51 is within the margin of both holds (H1 right edge .50, H2 left edge .52)
    for x, expected in ((0.50, 1), (0.53, 2)):
        tr = ContactTracker(CFG, HOLDMAP)
        assert hold_still(tr, (x, 0.45), 0.4)[-1][1] == expected


def test_low_visibility_reports_none_but_a_brief_dip_keeps_the_hold_memory():
    tr = ContactTracker(CFG, HOLDMAP)
    assert hold_still(tr, (0.45, 0.45), 0.3)[-1][1] == 1
    assert tr.update(make_state(0.31, {Limb.RH: (0.45, 0.45)}, vis=0.3))[Limb.RH] is None
    assert tr.update(make_state(0.34, {Limb.RH: (0.45, 0.45)}))[Limb.RH] == 1  # straight back


def test_a_long_low_visibility_stretch_clears_the_hold_and_needs_a_new_settle():
    tr = ContactTracker(CFG, HOLDMAP)
    assert hold_still(tr, (0.45, 0.45), 0.3)[-1][1] == 1
    for i in range(8):  # 8 low-vis frames > lost_memory_frames (5)
        assert (
            tr.update(make_state(0.31 + i * DT, {Limb.RH: (0.45, 0.45)}, vis=0.3))[Limb.RH] is None
        )
    after = hold_still(tr, (0.45, 0.45), 0.4, t0=0.6)
    assert after[0][1] is None and after[-1][1] == 1


def test_contacts_are_none_while_the_person_is_lost():
    tr = ContactTracker(CFG, HOLDMAP)
    hold_still(tr, (0.45, 0.45), 0.3)
    assert all(v is None for v in tr.update(make_state(0.4, found=False)).values())


def test_a_brief_gap_keeps_the_hold_so_the_limb_needs_no_new_settle_time():
    tr = ContactTracker(CFG, HOLDMAP)
    hold_still(tr, (0.45, 0.45), 0.3)
    for i in range(3):  # 3 lost frames <= lost_memory_frames (5)
        tr.update(make_state(0.33 + i * DT, found=False))
    assert tr.update(make_state(0.43, {Limb.RH: (0.45, 0.45)}))[Limb.RH] == 1  # straight back on


def test_a_long_gap_clears_the_hold_and_the_limb_must_settle_again():
    tr = ContactTracker(CFG, HOLDMAP)
    hold_still(tr, (0.45, 0.45), 0.3)
    for i in range(8):  # 8 lost frames > lost_memory_frames (5)
        tr.update(make_state(0.33 + i * DT, found=False))
    assert tr.update(make_state(0.6, {Limb.RH: (0.45, 0.45)}))[Limb.RH] is None


def test_each_limb_is_tracked_independently():
    tr = ContactTracker(CFG, HOLDMAP)
    contacts = {}
    for i in range(12):
        contacts = tr.update(make_state(i * DT, {Limb.LH: (0.45, 0.45), Limb.RH: (0.57, 0.45)}))
    assert contacts[Limb.LH] == 1 and contacts[Limb.RH] == 2
    assert contacts[Limb.LF] is None and contacts[Limb.RF] is None


def test_arm_length_is_the_median_and_defaults_until_seen():
    tr = ContactTracker(CFG, HOLDMAP)
    assert tr.arm_length == CFG["arm_length_default"]
    for i in range(5):
        tr.update(make_state(i * DT, {}))
    assert tr.arm_length == pytest.approx(0.30)


def test_pose_tracker_fills_contacts_when_given_a_holdmap():
    def landmarks():
        lms = {name: (0.5, 0.5, 0.9) for name in LANDMARK_NAMES}
        for side in ("left", "right"):
            lms[f"{side}_shoulder"] = (0.5, 0.40, 0.9)
            lms[f"{side}_elbow"] = (0.5, 0.55, 0.9)
        # right hand sits on hold 2; everything else far from any hold
        lms["right_wrist"] = lms["right_index"] = (0.57, 0.45, 0.9)
        lms["left_wrist"] = lms["left_index"] = (0.30, 0.45, 0.9)
        lms["left_heel"] = lms["left_foot_index"] = (0.35, 0.95, 0.9)
        lms["right_heel"] = lms["right_foot_index"] = (0.65, 0.95, 0.9)
        return [(x, y, v) for x, y, v in (lms[n] for n in LANDMARK_NAMES)]

    frame = np.zeros((48, 64, 3), np.uint8)
    with_map = PoseTracker(CFG, detector=lambda f, ts: [landmarks()], holdmap=HOLDMAP)
    without = PoseTracker(CFG, detector=lambda f, ts: [landmarks()])
    for i in range(15):
        a = with_map.process(i * DT, i, frame)
        b = without.process(i * DT, i, frame)
    assert a.contacts[Limb.RH] == 2 and a.contacts[Limb.LH] is None
    assert all(v is None for v in b.contacts.values())
