from __future__ import annotations

import pytest

from coach.brain.detectors import (
    CruxWarning,
    DetectorContext,
    FeetCut,
    QuietFeet,
    RoutePreview,
    WeightShift,
    good_tier_detectors,
)
from coach.brain.features import Features
from coach.contracts import ClimberState, Hold, HoldMap, Limb

CONFIG = {
    "brain": {
        "debounce_ms": 400,
        "move_speed_eps": 0.02,
        "cue_mute_after_fall_s": 1.0,
        "cue_mute_hip_speed": 0.6,
        "route_preview_ratio": 0.8,
        "weight_shift_ratio": 0.25,
        "quiet_feet_window_s": 1.0,
        "feet_cut_ms": 300,
        "priorities": {
            "crux_warning": 1,
            "route_preview": 2,
            "quiet_feet": 1,
            "feet_cut": 1,
            "weight_shift": 1,
        },
    },
    "arbiter": {
        "ttl_s": {
            "p0": 2.0,
            "p1": 1.2,
            "p2": 4.0,
        }
    },
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
        CONFIG,
        kwargs.get("holdmap"),
        kwargs.get("on_wall", True),
        kwargs.get("last_fall_t"),
        kwargs.get("crux_hold"),
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


def test_crux_warning_fires_once_and_handles_none_and_reset():
    detector = CruxWarning()
    context = ctx(crux_hold=7)

    # 1. Contact below crux_hold - 1 (target is 6) -> no event
    c_low = {Limb.LH: 4, Limb.RH: 5, Limb.LF: 1, Limb.RF: 2}
    assert detector.update(feature(), state(0.0, contacts=c_low), context) == []

    # 2. Reaches crux_hold - 1 (hold 6) -> fires once
    c_target = {Limb.LH: 4, Limb.RH: 6, Limb.LF: 1, Limb.RF: 2}
    events = detector.update(feature(), state(0.5, contacts=c_target), context)
    assert len(events) == 1
    assert events[0].kind == "crux_warning"
    assert events[0].data.get("crux_hold") == 7 or events[0].data.get("hold_id") == 7

    # 3. Next frame still on hold 6 or progressing to 7 -> does not fire again
    c_crux = {Limb.LH: 6, Limb.RH: 7, Limb.LF: 2, Limb.RF: 3}
    assert detector.update(feature(), state(1.0, contacts=c_crux), context) == []

    # 4. Climber downclimbs below target (max hold 5) -> latch resets internally
    assert detector.update(feature(), state(2.0, contacts=c_low), context) == []

    # 5. Climber approaches again and reaches hold 6 -> fires again
    re_events = detector.update(feature(), state(3.0, contacts=c_target), context)
    assert len(re_events) == 1
    assert re_events[0].kind == "crux_warning"

    # 6. If ctx.crux_hold is None, does nothing even if on hold 6
    det_none = CruxWarning()
    assert det_none.update(feature(), state(4.0, contacts=c_target), ctx(crux_hold=None)) == []


def test_route_preview_fires_on_ground_with_far_hold():
    detector = RoutePreview()
    # Holds: 1 -> 2 gap is 0.2; 2 -> 3 gap is 0.5
    holds = [
        Hold(1, (0.5, 0.8), (0, 0, 0, 0)),
        Hold(2, (0.5, 0.6), (0, 0, 0, 0)),
        Hold(3, (0.5, 0.1), (0, 0, 0, 0)),
    ]
    holdmap = HoldMap("p1", "red", holds, (100, 100), "manual")
    f = feature(arm_length=0.3)  # threshold = 0.8 * 0.3 = 0.24
    c_ground = ctx(holdmap=holdmap, on_wall=False)

    # Fires once on ground; biggest gap is 2->3 (distance 0.5), far hold is 3
    events = detector.update(f, state(0.0), c_ground)
    assert len(events) == 1
    assert events[0].kind == "route_preview"
    assert events[0].data["hold_id"] == 3
    assert events[0].data["gap"] == pytest.approx(0.5)

    # Does not spam on subsequent frames on ground
    assert detector.update(f, state(0.5), c_ground) == []

    # Small gaps do not fire
    small_holds = [
        Hold(1, (0.5, 0.8), (0, 0, 0, 0)),
        Hold(2, (0.5, 0.7), (0, 0, 0, 0)),
        Hold(3, (0.5, 0.6), (0, 0, 0, 0)),
    ]
    small_map = HoldMap("p2", "blue", small_holds, (100, 100), "manual")
    det_small = RoutePreview()
    assert det_small.update(f, state(0.0), ctx(holdmap=small_map, on_wall=False)) == []

    # Holdmap None or arm_length <= 0 do nothing
    det_none = RoutePreview()
    assert det_none.update(f, state(0.0), ctx(holdmap=None, on_wall=False)) == []
    assert det_none.update(feature(arm_length=0), state(0.0), c_ground) == []

    # Does not fire while on_wall=True
    det_wall = RoutePreview()
    assert det_wall.update(f, state(0.0), ctx(holdmap=holdmap, on_wall=True)) == []


def test_quiet_feet_detects_fidget():
    detector = QuietFeet()
    context = ctx()

    # t=0.0: foot LF touches hold 4
    c0 = {Limb.LF: 4, Limb.RF: 2, Limb.LH: 5, Limb.RH: 6}
    assert detector.update(feature(), state(0.0, contacts=c0), context) == []

    # t=0.2: foot LF leaves hold 4
    c1 = {Limb.LF: None, Limb.RF: 2, Limb.LH: 5, Limb.RH: 6}
    assert detector.update(feature(), state(0.2, contacts=c1), context) == []

    # t=0.5: foot LF re-touches hold 4 within quiet_feet_window_s (1.0s) -> fires!
    events = detector.update(feature(), state(0.5, contacts=c0), context)
    assert len(events) == 1
    assert events[0].kind == "quiet_feet"
    assert events[0].data == {"limb": "left_foot", "hold_id": 4}
    assert detector.still_valid(events[0], state(0.5, contacts=c0))

    # t=0.6: still on hold 4 -> does not fire again
    assert detector.update(feature(), state(0.6, contacts=c0), context) == []

    # t=1.0: foot LF leaves hold 4
    assert detector.update(feature(), state(1.0, contacts=c1), context) == []

    # t=1.2: foot LF touches a DIFFERENT hold (hold 8) -> no event
    c_diff = {Limb.LF: 8, Limb.RF: 2, Limb.LH: 5, Limb.RH: 6}
    assert detector.update(feature(), state(1.2, contacts=c_diff), context) == []

    # t=1.5: foot LF leaves hold 8
    assert detector.update(feature(), state(1.5, contacts=c1), context) == []

    # t=3.0: foot LF re-touches hold 8 after 1.5s (> quiet_feet_window_s 1.0s) -> no event
    assert detector.update(feature(), state(3.0, contacts=c_diff), context) == []


def test_feet_cut_timing_and_reset():
    detector = FeetCut()
    context = ctx()

    c_normal = {Limb.LH: 5, Limb.RH: 6, Limb.LF: 1, Limb.RF: 2}
    c_cut = {Limb.LH: 5, Limb.RH: 6, Limb.LF: None, Limb.RF: None}
    c_fell = {Limb.LH: None, Limb.RH: None, Limb.LF: None, Limb.RF: None}

    # t=0.0: normal climbing
    assert detector.update(feature(), state(0.0, contacts=c_normal), context) == []

    # t=0.1: feet cut (feet_cut_ms is 300) -> not fired yet
    assert detector.update(feature(), state(0.1, contacts=c_cut), context) == []

    # t=0.3: 200 ms elapsed -> not fired yet
    assert detector.update(feature(), state(0.3, contacts=c_cut), context) == []

    # t=0.41: 310 ms elapsed (> 300 ms) -> fires!
    events = detector.update(feature(), state(0.41, contacts=c_cut), context)
    assert len(events) == 1
    assert events[0].kind == "feet_cut"
    assert events[0].data["hold_id"] == 6
    assert detector.still_valid(events[0], state(0.41, contacts=c_cut))

    # t=0.5: still in the same cut episode -> does not re-fire
    assert detector.update(feature(), state(0.5, contacts=c_cut), context) == []

    # t=0.6: foot regains contact -> resets episode
    assert detector.update(feature(), state(0.6, contacts=c_normal), context) == []
    assert not detector.still_valid(events[0], state(0.6, contacts=c_normal))

    # If hands come off before 300 ms, continuous episode is interrupted
    assert detector.update(feature(), state(1.0, contacts=c_cut), context) == []
    assert detector.update(feature(), state(1.1, contacts=c_fell), context) == []
    assert detector.update(feature(), state(1.35, contacts=c_cut), context) == []


def test_weight_shift_conditions_and_debouncing():
    detector = WeightShift()
    context = ctx()

    # weight_shift_ratio = 0.25, leg_length = 0.4 -> threshold = 0.1
    f_static = feature(leg_length=0.4, hip_speed=0.01)
    f_moving = feature(leg_length=0.4, hip_speed=0.05)

    # LF is supporting foot at x=0.2. Hip is at x=0.4 -> dx = 0.2 > 0.1
    limbs = {Limb.LF: (0.2, 0.8), Limb.RF: (0.7, 0.8)}
    c_one_foot = {Limb.LF: 1, Limb.RF: None, Limb.LH: 4, Limb.RH: 5}

    # t=0.0: static, 1 foot on, hip far -> debounce starts (debounce_ms=400)
    assert (
        detector.update(
            f_static, state(0.0, contacts=c_one_foot, limbs=limbs, hip=(0.4, 0.5)), context
        )
        == []
    )

    # t=0.4: debounced time elapsed -> fires!
    events = detector.update(
        f_static, state(0.4, contacts=c_one_foot, limbs=limbs, hip=(0.4, 0.5)), context
    )
    assert len(events) == 1
    assert events[0].kind == "weight_shift"
    assert events[0].data["foot"] == "left_foot"
    assert events[0].data["hold_id"] == 1

    # Hip directly over foot (dx <= threshold) -> does not trigger
    det_centered = WeightShift()
    assert (
        det_centered.update(
            f_static, state(0.0, contacts=c_one_foot, limbs=limbs, hip=(0.22, 0.5)), context
        )
        == []
    )

    # Two feet on hold -> does not trigger
    c_two_feet = {Limb.LF: 1, Limb.RF: 2, Limb.LH: 4, Limb.RH: 5}
    det_two = WeightShift()
    assert (
        det_two.update(
            f_static, state(0.0, contacts=c_two_feet, limbs=limbs, hip=(0.4, 0.5)), context
        )
        == []
    )

    # Moving hip (not static) -> does not trigger
    det_moving = WeightShift()
    assert (
        det_moving.update(
            f_moving, state(0.0, contacts=c_one_foot, limbs=limbs, hip=(0.4, 0.5)), context
        )
        == []
    )


def test_gates_ground_vs_on_wall():
    # RoutePreview requires on_wall=False
    route_det = RoutePreview()
    holds = [Hold(1, (0.5, 0.8), (0, 0, 0, 0)), Hold(2, (0.5, 0.1), (0, 0, 0, 0))]
    holdmap = HoldMap("p1", "red", holds, (100, 100), "manual")
    f = feature(arm_length=0.3)

    assert route_det.update(f, state(0.0), ctx(holdmap=holdmap, on_wall=True)) == []
    assert len(route_det.update(f, state(0.0), ctx(holdmap=holdmap, on_wall=False))) == 1

    # On-wall detectors do not fire while on_wall=False
    c_cut = {Limb.LH: 5, Limb.RH: 6, Limb.LF: None, Limb.RF: None}
    assert (
        CruxWarning().update(
            feature(), state(0.0, contacts={Limb.LH: 6}), ctx(crux_hold=7, on_wall=False)
        )
        == []
    )
    assert FeetCut().update(feature(), state(0.5, contacts=c_cut), ctx(on_wall=False)) == []
    assert (
        QuietFeet().update(feature(), state(0.5, contacts={Limb.LF: 4}), ctx(on_wall=False)) == []
    )
    assert (
        WeightShift().update(
            f, state(0.5, contacts={Limb.LF: 1, Limb.RF: None}), ctx(on_wall=False)
        )
        == []
    )


def test_good_tier_detectors_factory():
    detectors = good_tier_detectors(CONFIG)
    assert len(detectors) == 5
    assert [type(d).__name__ for d in detectors] == [
        "CruxWarning",
        "RoutePreview",
        "QuietFeet",
        "FeetCut",
        "WeightShift",
    ]


def test_quiet_feet_does_not_fire_on_alternating_holds():
    # A foot alternating between hold A and hold B (not returning to the SAME
    # hold) must NOT trigger QuietFeet.  Only same-hold re-contact within the
    # window is a fidget; moving to a different hold is normal footwork.
    detector = QuietFeet()
    context = ctx()

    # Phase 1: LF touches hold 4, then hold 5 (never re-contacts hold 4).
    c_a = {Limb.LF: 4, Limb.RF: 2, Limb.LH: 6, Limb.RH: 7}
    c_none = {Limb.LF: None, Limb.RF: 2, Limb.LH: 6, Limb.RH: 7}
    c_b = {Limb.LF: 5, Limb.RF: 2, Limb.LH: 6, Limb.RH: 7}

    # t=0.0: LF on hold 4.
    assert detector.update(feature(), state(0.0, contacts=c_a), context) == []
    # t=0.1: LF leaves hold 4.
    assert detector.update(feature(), state(0.1, contacts=c_none), context) == []
    # t=0.3: LF lands on hold 5 (not hold 4) → no quiet_feet.
    assert detector.update(feature(), state(0.3, contacts=c_b), context) == []

    # Phase 2: LF leaves hold 5, then returns to hold 4 → still not same hold
    # as the last left hold (hold 5), so no event.
    # t=0.5: LF leaves hold 5.
    assert detector.update(feature(), state(0.5, contacts=c_none), context) == []
    # t=0.7: LF returns to hold 4, but _last_hold[LF] is 5, not 4 → no event.
    assert detector.update(feature(), state(0.7, contacts=c_a), context) == []

    # Phase 3: Confirm it still fires for a true same-hold fidget.
    # LF leaves hold 4, then immediately re-contacts hold 4 within window.
    assert detector.update(feature(), state(0.9, contacts=c_none), context) == []
    events = detector.update(feature(), state(1.1, contacts=c_a), context)
    assert len(events) == 1
    assert events[0].kind == "quiet_feet"
    assert events[0].data["hold_id"] == 4
