import pytest

from coach.brain.features import FeatureTracker
from coach.contracts import ClimberState, Limb


def state(
    t=0.0,
    *,
    x=0.2,
    lh_y=0.4,
    hip_x=0.3,
    hip_y=0.6,
    contacts=None,
    keypoints=None,
    person_found=True,
):
    points = {
        "left_shoulder": (0.2, 0.2, 1),
        "left_elbow": (0.2, 0.3, 1),
        "left_wrist": (0.2, 0.4, 1),
        "right_shoulder": (0.4, 0.2, 1),
        "right_elbow": (0.5, 0.2, 1),
        "right_wrist": (0.5, 0.3, 1),
        "left_hip": (0.25, 0.6, 1),
        "left_knee": (0.25, 0.75, 1),
        "left_ankle": (0.25, 0.9, 1),
        "right_hip": (0.35, 0.6, 1),
        "right_knee": (0.35, 0.75, 1),
        "right_ankle": (0.35, 0.9, 1),
    }
    points.update(keypoints or {})
    return ClimberState(
        t,
        int(t * 10),
        person_found,
        points,
        {Limb.LH: (x, lh_y), Limb.RH: (0.4, 0.4), Limb.LF: (0.25, 0.9), Limb.RF: (0.35, 0.9)},
        contacts or {limb: None for limb in Limb},
        (hip_x, hip_y),
    )


def test_speeds_angles_hip_and_static():
    tracker = FeatureTracker(0.02)
    first = tracker.update(state())
    second = tracker.update(state(2, x=0.4, hip_y=0.8))
    assert second.limb_speed[Limb.LH] == 0.1
    assert second.elbow_angle["right"] == 90
    assert second.hip_velocity_y > 0
    assert second.hip_static_s == 0
    still = tracker.update(state(3, x=0.4, hip_y=0.8))
    assert still.hip_static_s == 1
    assert first.highest_hold_id is None


def test_low_visibility_keypoint_is_excluded_from_elbow_angle():
    tracker = FeatureTracker(0.02)

    features = tracker.update(state(keypoints={"left_elbow": (0.2, 0.3, 0.3)}))

    assert features.elbow_angle["left"] is None


def test_calibration_highest_hold_and_hand_search():
    tracker = FeatureTracker(0.02)
    a = tracker.update(state())
    a.add_calibration_sample(state())
    on_hold_3 = {Limb.LH: 3, Limb.RH: None, Limb.LF: None, Limb.RF: None}
    off = {limb: None for limb in Limb}
    tracker.update(state(1, contacts=on_hold_3))
    held = tracker.update(state(1.5, contacts=on_hold_3))  # held past contact_stable_s: steady
    assert held.stable_contacts[Limb.LH] == 3
    # Let go, hand up above the shoulder (y 0.2): searching once that's steady, not before.
    let_go = tracker.update(state(2, x=0.4, lh_y=0.1, contacts=off))
    assert let_go.highest_hold_id == 3
    assert let_go.hand_moving_no_contact_s[Limb.LH] == 0  # one frame off: maybe a flicker
    searching = tracker.update(state(3, x=0.4, lh_y=0.1, contacts=off))
    assert searching.hand_moving_no_contact_s[Limb.LH] == 1
    # Not reset by the hand pausing as it pats around (same point again).
    paused = tracker.update(state(3.5, x=0.4, lh_y=0.1, contacts=off))
    assert paused.hand_moving_no_contact_s[Limb.LH] == 1.5
    # A hand down by the side (shaking out, chalking) isn't searching.
    down = tracker.update(state(4, x=0.4, lh_y=0.5, contacts=off))
    assert down.hand_moving_no_contact_s[Limb.LH] == 0
    assert a.arm_length > 0 and a.leg_length > 0


def test_a_still_hip_stays_still_through_jitter_and_a_stall_adds_up():
    """A hip jittering a little (landmark noise) is still: hip_static_s keeps counting. The old
    frame-to-frame speed test reset on every jiggle, so a 12 s stall never added up."""
    tracker = FeatureTracker(0.02)
    jitter = (0.0, 0.004, -0.003, 0.005, -0.004, 0.002)
    last = None
    for i in range(61):  # 2 s at 30 fps
        last = tracker.update(state(i / 30, hip_x=0.3 + jitter[i % len(jitter)]))
    assert last.hip_static_s == pytest.approx(2.0, abs=0.04)
    moved = tracker.update(state(61 / 30, hip_x=0.4))  # a real move (0.1, over 0.12 x leg)
    assert moved.hip_static_s == 0


def test_a_contact_flicker_does_not_change_the_stable_contacts():
    tracker = FeatureTracker(0.02, contact_stable_s=0.25)
    on = {Limb.LH: None, Limb.RH: None, Limb.LF: 2, Limb.RF: 1}
    flicker = {**on, Limb.LF: None}
    tracker.update(state(0.0, contacts=on))
    blip = tracker.update(state(1 / 30, contacts=flicker))  # one frame off
    assert blip.stable_contacts[Limb.LF] == 2 and blip.contacts(state()) == blip.stable_contacts
    back = tracker.update(state(2 / 30, contacts=on))
    assert back.stable_contacts[Limb.LF] == 2
    for i in range(3, 15):  # really off for 0.4 s: now it counts
        gone = tracker.update(state(i / 30, contacts=flicker))
    assert gone.stable_contacts[Limb.LF] is None


def test_highest_hold_resets_between_attempts():
    tracker = FeatureTracker(0.02)
    tracker.update(state(0, contacts={Limb.LH: 5}))
    assert tracker.update(state(1, contacts={Limb.LH: 5})).highest_hold_id == 5

    tracker.reset_highest_hold()
    assert tracker.update(state(2, contacts={Limb.LH: 2})).highest_hold_id == 2


def test_sideways_hip_motion_is_not_static():
    tracker = FeatureTracker(0.02)
    tracker.update(state(0, hip_x=0.3))
    moved = tracker.update(state(0.2, hip_x=0.4))

    assert moved.hip_speed > 0
    assert moved.hip_static_s == 0


def test_person_reappearing_after_dropout_is_not_a_velocity_spike():
    # pose.py freezes the hip at its last known point while person_found is False, so
    # the next real detection can land far away (a different person, or the same one
    # having moved during the gap). That jump must not register as hip velocity/speed
    # -- it previously got mistaken for a fall.
    tracker = FeatureTracker(0.02)
    tracker.update(state(0, hip_y=0.3))
    tracker.update(state(0.5, hip_y=0.3, person_found=False))
    reappeared = tracker.update(state(0.6, hip_y=0.9))
    assert reappeared.hip_velocity_y == 0
    assert reappeared.hip_speed == 0


def test_duplicate_timestamps_no_zero_division():
    # Two frames at the same t (dt == 0) must not raise ZeroDivisionError.
    tracker = FeatureTracker(0.02)
    s0 = state(1.0)
    tracker.update(s0)
    # Second call with the same t: dt == 0 → all speed paths guarded by dt > 0.
    s1 = state(1.0, x=0.5)
    result = tracker.update(s1)
    # Speeds must be 0.0 (no division), and no exception must be raised.
    assert result.limb_speed[Limb.LH] == 0.0
    assert result.hip_speed == 0.0


def test_calibration_suppressed_when_keypoint_low_visibility():
    # A chain_length calculation during add_calibration_sample must be skipped
    # (not raise) when one of the three keypoints in the chain has low visibility.
    tracker = FeatureTracker(0.02)
    # Build a state where the left elbow has visibility < 0.5 so the left-arm
    # chain fails; the right-arm chain should succeed as a fallback, and the
    # leg chain (all visible) should also succeed.
    low_vis_state = state(
        0.0,
        keypoints={
            "left_shoulder": (0.2, 0.2, 1.0),
            "left_elbow": (0.2, 0.3, 0.1),  # low visibility
            "left_wrist": (0.2, 0.4, 1.0),
        },
    )
    features = tracker.update(low_vis_state)
    # add_calibration_sample must not raise even with the broken left-arm chain.
    features.add_calibration_sample(low_vis_state)
    # The fallback right-arm chain is fully visible in the default state fixture,
    # so arm_length must be a positive number after the sample.
    assert tracker._arm_samples  # at least one sample recorded via the right chain
    assert all(v > 0 for v in tracker._arm_samples)


def test_missing_limb_yields_zero_speed():
    # A state.limbs dict that is missing a Limb entry entirely must produce
    # 0.0 speed for that limb in FeatureTracker.update(), not raise.
    tracker = FeatureTracker(0.02)
    # First frame: all four limbs present so the tracker records positions.
    tracker.update(state(0.0))
    # Second frame: RF is absent from the limbs dict.
    partial_limbs = {Limb.LH: (0.3, 0.4), Limb.RH: (0.4, 0.4), Limb.LF: (0.25, 0.9)}
    s_partial = ClimberState(
        1.0,
        10,
        True,
        state(0.0).keypoints,
        partial_limbs,
        {limb: None for limb in Limb},
        (0.3, 0.6),
    )
    result = tracker.update(s_partial)
    assert result.limb_speed[Limb.RF] == 0.0
