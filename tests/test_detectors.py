from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from coach.brain.detectors import (
    ArmsBent,
    DetectorContext,
    FeetLow,
    HandSearching,
    Stalling,
    mvp_detectors,
)
from coach.brain.features import Features
from coach.contracts import ClimberState, Hold, HoldMap, Limb

BRAIN = yaml.safe_load((Path(__file__).parents[1] / "config" / "default.yaml").read_text())["brain"]
CONFIG = {
    "brain": {
        **BRAIN,  # the real settings (so new keys are there), this test's on top
        "feet_low_s": 0.4,  # these tests time feet_low by the debounce
        "search_near_ratio": 2.0,  # candidate scoring tests: any hold in reach is near enough
        "debounce_ms": 400,
        "stall_s": [4, 7, 10],
        "arms_bent_deg": 140,
        "feet_low_ratio": 0.9,
        "search_ms": 400,
        "move_speed_eps": 0.02,
        "cue_mute_after_fall_s": 1.0,
        "cue_mute_hip_speed": 0.6,
        "next_hold_distance_penalty": 0.5,
        "priorities": {"stalling": 1, "arms_bent": 1, "feet_low": 1, "hand_searching": 1},
    },
    "arbiter": {"ttl_s": {"p1": 1.2}},
}


def state(t: float, *, contacts=None, limbs=None, keypoints=None, hip=(0.5, 0.5)):
    return ClimberState(
        t,
        int(t * 10),
        True,
        keypoints or {},
        limbs or {},
        contacts or {limb: None for limb in Limb},
        hip,
    )


def ctx(**kwargs):
    return DetectorContext(
        CONFIG, kwargs.get("holdmap"), kwargs.get("on_wall", True), kwargs.get("last_fall_t")
    )


def feature(**kwargs):
    values = dict(
        hip_speed=0,
        hip_static_s=0,
        limb_speed={limb: 0 for limb in Limb},
        elbow_angle={},
        hand_moving_no_contact_s={},
        arm_length=0.25,
        leg_length=0.3,
    )
    values.update(kwargs)
    return Features(**values)


def test_stalling_escalates_and_resets():
    detector = Stalling()
    events = []
    for t, seconds in ((0, 3.9), (0.4, 4), (4.4, 7), (7.4, 10), (7.8, 10)):
        events += detector.update(
            feature(hip_static_s=seconds),
            state(t, contacts={Limb.LH: 3, Limb.RH: 5}),
            ctx(),
        )
    assert [(event.data["stage"]) for event in events] == [0, 1, 2]
    assert all(event.data["hold_id"] == 5 for event in events)
    assert detector.still_valid(events[-1], state(8))
    detector.update(feature(hip_static_s=0, hip_speed=0.1), state(8), ctx())
    assert not detector.still_valid(events[-1], state(8))


HANGING = {Limb.LH: 1, Limb.RH: 2, Limb.LF: None, Limb.RF: None}  # both hands on holds


def hanging(**kwargs):
    """Features of a climber hanging still off both hands."""
    return feature(**{"hip_static_s": 1.0, "stable_contacts": HANGING, **kwargs})


def test_arms_bent_requires_both_and_hysteresis():
    detector = ArmsBent()
    f = hanging(elbow_angle={"left": 100, "right": 100})
    assert detector.update(f, state(0), ctx()) == []
    assert detector.update(f, state(0.4), ctx())[0].kind == "arms_bent"
    detector.update(hanging(elbow_angle={"left": 142, "right": 100}), state(1), ctx())
    assert detector.update(f, state(1.4), ctx()) == []
    assert detector.update(hanging(elbow_angle={"left": 150, "right": 100}), state(2), ctx()) == []


def test_arms_bent_needs_a_settled_hip_and_both_hands_on_holds():
    bent = {"left": 100, "right": 100}
    moving = ArmsBent()  # pulling through a move: the hip hasn't settled
    for t in (0, 0.4, 0.8):
        assert moving.update(hanging(elbow_angle=bent, hip_static_s=0.1), state(t), ctx()) == []
    one_hand = ArmsBent()  # a hand off: reaching, not hanging
    reaching = feature(
        elbow_angle=bent, hip_static_s=1.0, stable_contacts={**HANGING, Limb.RH: None}
    )
    for t in (0, 0.4, 0.8):
        assert one_hand.update(reaching, state(t), ctx()) == []


ON_HOLDS = {Limb.LH: 4, Limb.RH: 4, Limb.LF: None, Limb.RF: None}  # hanging off both hands
FEET_LOW_HOLDMAP = HoldMap("p", "red", [Hold(4, (0.5, 0.2), (0, 0, 0, 0))], (100, 100), "manual")


def test_feet_low_requires_high_hands():
    kp = {
        "left_shoulder": (0.4, 0.4, 1),
        "right_shoulder": (0.6, 0.4, 1),
        "left_wrist": (0.4, 0.2, 1),
        "right_wrist": (0.6, 0.2, 1),
    }
    limbs = {Limb.LF: (0.4, 0.9), Limb.RF: (0.6, 0.8)}
    detector = FeetLow()
    f = feature(leg_length=0.3, stable_contacts=ON_HOLDS)
    holdmap_ctx = ctx(holdmap=FEET_LOW_HOLDMAP)
    assert detector.update(f, state(0, limbs=limbs, keypoints=kp), holdmap_ctx) == []
    event = detector.update(
        f, state(0.4, limbs=limbs, keypoints=kp, contacts={Limb.LH: 4}), holdmap_ctx
    )[0]
    assert event.kind == "feet_low"
    assert event.data == {"hold_id": 4, "limb": Limb.LF.value}
    low_kp = {**kp, "left_wrist": (0.4, 0.5), "right_wrist": (0.6, 0.5)}
    assert detector.update(f, state(1, limbs=limbs, keypoints=low_kp), holdmap_ctx) == []


def test_feet_low_requires_a_hold_map():
    # Without a hold map there is no route-relative meaning to "low"; feet dangling
    # with no labelled holds must not fire feet_low at all (it used to fire
    # repeatedly for the whole climb).
    kp = {
        "left_shoulder": (0.4, 0.4, 1),
        "right_shoulder": (0.6, 0.4, 1),
        "left_wrist": (0.4, 0.2, 1),
        "right_wrist": (0.6, 0.2, 1),
    }
    limbs = {Limb.LF: (0.4, 0.9), Limb.RF: (0.6, 0.8)}
    detector = FeetLow()
    f = feature(leg_length=0.3)
    assert detector.update(f, state(0, limbs=limbs, keypoints=kp), ctx()) == []
    assert (
        detector.update(f, state(0.4, limbs=limbs, keypoints=kp, contacts={Limb.LH: 4}), ctx())
        == []
    )


def test_hand_searching_scores_higher_candidate():
    holds = [Hold(4, (0.55, 0.2), (0, 0, 0, 0)), Hold(7, (0.7, 0.3), (0, 0, 0, 0))]
    holdmap = HoldMap("p", "red", holds, (100, 100), "manual")
    limbs = {Limb.LH: (0.5, 0.5), Limb.RH: (0.8, 0.8), Limb.LF: (0.4, 0.9), Limb.RF: (0.6, 0.9)}
    kp = {"left_shoulder": (0.5, 0.5, 1), "right_shoulder": (0.8, 0.7, 1)}
    detector = mvp_detectors(CONFIG)[-1]
    f = feature(hand_moving_no_contact_s={Limb.LH: 0.4}, arm_length=0.31)
    event = detector.update(f, state(0, limbs=limbs, keypoints=kp), ctx(holdmap=holdmap))[0]
    assert event.data["hold_id"] == 4
    assert event.data["dir"] == "up"
    assert (
        detector.still_valid(event, state(0.5, contacts={**state(0).contacts, Limb.LH: 4})) is False
    )


@pytest.mark.parametrize("context", [ctx(on_wall=False), ctx(last_fall_t=0), ctx()])
def test_gate(context):
    detector = Stalling()
    hip_speed = 0.7 if context.on_wall else 0
    assert detector.update(feature(hip_static_s=10, hip_speed=hip_speed), state(0.1), context) == []


def test_missing_optional_integration_data_is_skipped():
    pytest.skip("optional labelled footage is not present in this checkout")


def test_feet_low_release_band_suppresses_refiring():
    # After FeetLow fires, hovering the ratio just below the trigger threshold
    # but above the 95% release band should NOT re-arm the detector.
    # feet_low_ratio = 0.9; theoretical release = 0.9 * 0.95 = 0.855.
    kp_high_hands = {
        "left_shoulder": (0.4, 0.4, 1),
        "right_shoulder": (0.6, 0.4, 1),
        "left_wrist": (0.4, 0.2, 1),
        "right_wrist": (0.6, 0.2, 1),
    }
    # leg_length = 0.3; ratio = 0.9; threshold = 0.27.
    # Foot at y=0.9, hip at y=0.5 → height_above(0.9, 0.5)/0.3 = 1.33 > 0.9 → fires.
    # At 94% → 0.9*0.3*0.94 = 0.254; height 0.254 / 0.3 = 0.847 < 0.9 (below trigger)
    # but > 0.855 (above release), so _active should stay True and NOT re-fire.
    detector = FeetLow()
    f = feature(leg_length=0.3, stable_contacts=ON_HOLDS)
    holdmap_ctx = ctx(holdmap=FEET_LOW_HOLDMAP)
    limbs_fire = {Limb.LF: (0.4, 0.9), Limb.RF: (0.6, 0.9)}  # low
    # Debounce fires after 0.4 s.
    assert (
        detector.update(f, state(0, limbs=limbs_fire, keypoints=kp_high_hands), holdmap_ctx) == []
    )
    fired = detector.update(
        f, state(0.4, limbs=limbs_fire, keypoints=kp_high_hands, contacts={Limb.LH: 4}), holdmap_ctx
    )
    assert len(fired) == 1 and fired[0].kind == "feet_low"
    # Hover at 94% of trigger threshold (between release and trigger): must NOT re-fire.
    limbs_hover = {Limb.LF: (0.4, 0.75), Limb.RF: (0.6, 0.75)}  # above 95% band, below trigger
    assert (
        detector.update(f, state(0.8, limbs=limbs_hover, keypoints=kp_high_hands), holdmap_ctx)
        == []
    )
    # Drop back past the release threshold: must re-arm and eventually re-fire.
    limbs_low_again = {Limb.LF: (0.4, 0.95), Limb.RF: (0.6, 0.95)}
    assert (
        detector.update(f, state(1.2, limbs=limbs_low_again, keypoints=kp_high_hands), holdmap_ctx)
        == []
    )
    fired2 = detector.update(
        f,
        state(1.6, limbs=limbs_low_again, keypoints=kp_high_hands, contacts={Limb.LH: 4}),
        holdmap_ctx,
    )
    assert len(fired2) == 1 and fired2[0].kind == "feet_low"


def test_hand_searching_discards_holds_beyond_arm_length():
    # HandSearching must skip any candidate hold whose shoulder→hold distance
    # exceeds arm_length, even if that hold would otherwise score higher.
    #
    # Setup: shoulder at (0.5, 0.4). arm_length = 0.3.
    # Hold A at (0.55, 0.2): reach = hypot(0.05, 0.2) ≈ 0.206 < 0.3  → in reach.
    # Hold B at (0.9, 0.2): reach = hypot(0.4, 0.2) ≈ 0.447 > 0.3  → out of reach.
    # Hold B is higher (y=0.2) so it would score more by height; it must be excluded.

    holds = [
        Hold(3, (0.55, 0.2), (0, 0, 0, 0)),  # in reach
        Hold(9, (0.90, 0.2), (0, 0, 0, 0)),  # beyond arm_length; higher score if reachable
    ]
    holdmap = HoldMap("p", "red", holds, (100, 100), "manual")
    # LH at (0.5, 0.5), shoulder at (0.5, 0.4).
    limbs = {Limb.LH: (0.5, 0.5), Limb.RH: (0.8, 0.8), Limb.LF: (0.4, 0.9), Limb.RF: (0.6, 0.9)}
    kp = {"left_shoulder": (0.5, 0.4, 1.0), "right_shoulder": (0.8, 0.7, 1.0)}
    detector = HandSearching()
    f = feature(hand_moving_no_contact_s={Limb.LH: 0.4}, arm_length=0.3)
    events = detector.update(f, state(0, limbs=limbs, keypoints=kp), ctx(holdmap=holdmap))
    assert len(events) == 1
    assert events[0].data["hold_id"] == 3, "expected in-reach hold 3, but got out-of-reach hold 9"
