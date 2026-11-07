from coach.brain.features import FeatureTracker
from coach.contracts import ClimberState, Limb


def state(t=0.0, *, x=0.2, hip_x=0.3, hip_y=0.6, contacts=None, keypoints=None):
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
        True,
        points,
        {Limb.LH: (x, 0.4), Limb.RH: (0.4, 0.4), Limb.LF: (0.25, 0.9), Limb.RF: (0.35, 0.9)},
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
    tracker.update(state(1, contacts={Limb.LH: 3, Limb.RH: None, Limb.LF: None, Limb.RF: None}))
    moving = tracker.update(state(2, x=0.4, contacts={limb: None for limb in Limb}))
    assert moving.highest_hold_id == 3
    assert moving.hand_moving_no_contact_s[Limb.LH] == 1
    reset = tracker.update(
        state(3, x=0.4, contacts={Limb.LH: 3, Limb.RH: None, Limb.LF: None, Limb.RF: None})
    )
    assert reset.hand_moving_no_contact_s[Limb.LH] == 0
    assert a.arm_length > 0 and a.leg_length > 0


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
