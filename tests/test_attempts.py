from pathlib import Path

import yaml

from coach.brain.attempts import AttemptTracker
from coach.brain.features import FeatureTracker
from coach.contracts import ClimberState, Hold, HoldMap, Limb

# Mirror the real config/default.yaml keys. "downclimb" is deliberately absent from
# priorities: it is an AttemptSummary outcome, not a spoken event kind.
CONFIG = {
    "brain": {
        "on_wall_margin": 0.15,
        "move_speed_eps": 0.02,
        "on_wall_settle_s": 0.5,
        "start_hold_settle_s": 0.0,
        "top_hold_s": 1,
        "fall_speed": 0.6,
        "ground_settle_s": 1,
        "priorities": {"attempt_start": 2, "high_point": 1, "top": 0, "fall": 0},
    },
    "arbiter": {"ttl_s": {"p0": 2, "p1": 1.2, "p2": 4}},
}


def real_config():
    return yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))


def s(t, hip=0.7, contacts=None):
    points = {
        name: (0.3, y, 1)
        for name, y in {
            "left_shoulder": 0.2,
            "left_elbow": 0.3,
            "left_wrist": 0.4,
            "right_shoulder": 0.2,
            "right_elbow": 0.3,
            "right_wrist": 0.4,
            "left_hip": 0.7,
            "left_knee": 0.8,
            "left_ankle": 0.9,
            "right_hip": 0.7,
            "right_knee": 0.8,
            "right_ankle": 0.9,
        }.items()
    }
    return ClimberState(
        t,
        int(t * 10),
        True,
        points,
        {limb: (0.2, hip) for limb in Limb},
        contacts or {limb: None for limb in Limb},
        (0.3, hip),
    )


def run(tracker, feature, state):
    return tracker.update(state, feature.update(state))


def make_tracker():
    fmap = HoldMap(
        "p",
        "red",
        [Hold(1, (0.2, 0.8), (0, 0, 0, 0)), Hold(5, (0.2, 0.2), (0, 0, 0, 0), "top")],
        (100, 100),
        "manual",
    )
    feature = FeatureTracker()
    return AttemptTracker(fmap, CONFIG), feature


def test_both_feet_start_and_top_summary():
    attempt, feature = make_tracker()
    ground = s(0)
    run(attempt, feature, ground)
    start = run(
        attempt, feature, s(1, contacts={Limb.LF: 1, Limb.RF: 1, Limb.LH: None, Limb.RH: None})
    )
    assert [e.kind for e in start] == ["attempt_start"]
    run(attempt, feature, s(1.5, contacts={Limb.LF: 1, Limb.RF: 1, Limb.LH: 5, Limb.RH: 5}))
    events = run(
        attempt, feature, s(2.5, contacts={Limb.LF: 1, Limb.RF: 1, Limb.LH: 5, Limb.RH: 5})
    )
    assert "top" in [e.kind for e in events]
    assert attempt.finished_attempts[-1].outcome == "top"
    assert attempt.finished_attempts[-1].duration_s == 1.5


def test_start_requires_both_feet_on_start_holds_for_settle_time():
    config = {**CONFIG, "brain": {**CONFIG["brain"], "start_hold_settle_s": 0.5}}
    holds = [
        Hold(1, (0.2, 0.8), (0, 0, 0, 0), "start"),
        Hold(2, (0.4, 0.8), (0, 0, 0, 0), "start"),
        Hold(3, (0.6, 0.6), (0, 0, 0, 0), "mid"),
    ]
    attempt = AttemptTracker(HoldMap("p", "red", holds, (100, 100), "manual"), config)
    feature = FeatureTracker()
    run(attempt, feature, s(0))
    assert run(attempt, feature, s(1, contacts={Limb.LF: 1})) == []
    assert run(attempt, feature, s(2, contacts={Limb.LF: 3, Limb.RF: 3})) == []
    assert run(attempt, feature, s(2.1, contacts={Limb.LF: 1, Limb.RF: 2})) == []
    assert run(attempt, feature, s(2.5, contacts={Limb.LF: 1, Limb.RF: 2})) == []
    events = run(attempt, feature, s(2.6, contacts={Limb.LF: 1, Limb.RF: 2}))
    assert [event.kind for event in events] == ["attempt_start"]


def test_hip_settle_highpoint_and_downclimb():
    attempt, feature = make_tracker()
    run(attempt, feature, s(0))
    assert not run(attempt, feature, s(0.2, hip=0.45))
    events = run(attempt, feature, s(0.8, hip=0.45, contacts={Limb.LF: 1, Limb.RF: 1}))
    assert "attempt_start" in [e.kind for e in events]
    events = run(
        attempt,
        feature,
        s(1, hip=0.45, contacts={Limb.LH: 3, Limb.RH: None, Limb.LF: None, Limb.RF: None}),
    )
    # No finished attempt yet, so there is no prior best to beat: high_point must
    # stay quiet on attempt 1 (it would otherwise fire on almost every new hold).
    assert "high_point" not in [e.kind for e in events]
    run(attempt, feature, s(1.5, hip=0.7))
    events = run(attempt, feature, s(2.6, hip=0.7))
    # downclimb emits no CoachEvent; it only shows up as the summary outcome.
    assert "downclimb" not in [e.kind for e in events]
    assert attempt.finished_attempts[-1].outcome == "downclimb"
    assert attempt.finished_attempts[-1].high_hold_id == 3


def test_high_point_fires_once_a_prior_attempt_exists():
    # Attempt 1 reaches hold 3 and downclimbs (no high_point spoken, see above). On
    # attempt 2, beating that hold 3 best is a real record and must fire high_point.
    attempt, feature = make_tracker()
    run(attempt, feature, s(0))
    run(attempt, feature, s(0.2, hip=0.45))
    run(attempt, feature, s(0.8, hip=0.45, contacts={Limb.LF: 1, Limb.RF: 1}))
    run(
        attempt,
        feature,
        s(1, hip=0.45, contacts={Limb.LH: 3, Limb.RH: None, Limb.LF: None, Limb.RF: None}),
    )
    run(attempt, feature, s(1.5, hip=0.7))
    run(attempt, feature, s(2.6, hip=0.7))
    assert attempt.finished_attempts[-1].outcome == "downclimb"

    run(attempt, feature, s(3, hip=0.45))
    events = run(attempt, feature, s(3.6, hip=0.45, contacts={Limb.LF: 1, Limb.RF: 1}))
    assert "attempt_start" in [e.kind for e in events]
    events = run(
        attempt,
        feature,
        s(3.8, hip=0.45, contacts={Limb.LH: 4, Limb.RH: None, Limb.LF: None, Limb.RF: None}),
    )
    assert "high_point" in [e.kind for e in events]


def test_dynamic_move_mid_climb_does_not_poison_later_downclimb():
    # A fast hip move made while still clearly on the wall (e.g. a dynamic bump) must
    # not be confused with a fall: _fall_seen should only reflect velocity during the
    # actual trip to the ground, not any fast moment earlier in the attempt.
    attempt, feature = make_tracker()
    run(attempt, feature, s(0))
    run(attempt, feature, s(0.2, hip=0.45))
    run(attempt, feature, s(0.8, hip=0.45, contacts={Limb.LF: 1, Limb.RF: 1}))
    run(
        attempt,
        feature,
        s(1, hip=0.45, contacts={Limb.LH: 1, Limb.RH: None, Limb.LF: None, Limb.RF: None}),
    )
    assert attempt.phase == "ON_WALL"

    # Dynamic move: hip drops fast but the climber is still well off the ground
    # (height above standing far exceeds on_wall_margin), so this is not a fall.
    run(attempt, feature, s(1.05, hip=0.5))  # velocity = 0.05/0.05 = 1.0 > fall_speed
    assert not attempt._fall_seen
    assert attempt.phase == "ON_WALL"

    # Climber then downclimbs slowly and controlled all the way to the ground.
    run(attempt, feature, s(1.6, hip=0.6))
    run(attempt, feature, s(2.2, hip=0.7))
    events = run(attempt, feature, s(3.3, hip=0.7))
    assert "fall" not in [e.kind for e in events]
    assert attempt.finished_attempts[-1].outcome == "downclimb"


def test_frozen_top_contact_during_a_fall_does_not_count_as_send():
    # A frozen/hidden-limb contact (coach/perception/pose.py) can keep both hands
    # reporting the top hold for a while after the climber has actually let go and
    # started falling. The top-hold timer must not accumulate while hip velocity
    # shows the climber is mid-fall, even if _is_top() still reads true.
    attempt, feature = make_tracker()
    run(attempt, feature, s(0))
    run(attempt, feature, s(0.2, hip=0.3))
    events = run(attempt, feature, s(0.8, hip=0.3, contacts={Limb.LF: 1, Limb.RF: 1}))
    assert "attempt_start" in [e.kind for e in events]
    run(
        attempt,
        feature,
        s(1, hip=0.3, contacts={Limb.LH: 5, Limb.RH: 5, Limb.LF: 1, Limb.RF: 1}),
    )

    # Stale top-hold contact (frozen point) while the hip is actually plunging fast.
    events = run(
        attempt,
        feature,
        s(1.3, hip=0.75, contacts={Limb.LH: 5, Limb.RH: 5, Limb.LF: None, Limb.RF: None}),
    )
    assert "top" not in [e.kind for e in events]
    assert attempt._top_since is None

    events = run(
        attempt,
        feature,
        s(2.4, hip=0.7, contacts={Limb.LH: None, Limb.RH: None, Limb.LF: None, Limb.RF: None}),
    )
    assert "top" not in [e.kind for e in events]
    assert "fall" in [e.kind for e in events]
    assert attempt.finished_attempts[-1].outcome == "fall"


def test_downclimb_against_real_config_does_not_crash():
    # Regression: real config/default.yaml has no "downclimb" priority. Emitting a
    # downclimb CoachEvent would KeyError. A downclimb must close quietly instead.
    config = real_config()
    feature = FeatureTracker(float(config["brain"]["move_speed_eps"]))
    attempt = AttemptTracker(
        HoldMap("p", "red", [Hold(1, (0.2, 0.8), (0, 0, 0, 0), "start")], (100, 100), "manual"),
        config,
    )
    run(attempt, feature, s(0))
    run(attempt, feature, s(0.8, hip=0.45, contacts={Limb.LF: 1, Limb.RF: 1}))
    run(
        attempt,
        feature,
        s(1.4, hip=0.45, contacts={Limb.LH: 3, Limb.RH: None, Limb.LF: 1, Limb.RF: 1}),
    )  # raised for >= on_wall_settle_s -> ON_WALL
    # Descend gradually so hip velocity stays under fall_speed (no fall is registered).
    run(attempt, feature, s(2.0, hip=0.55))
    run(attempt, feature, s(2.6, hip=0.7))  # back on the ground; ground_since starts
    events = run(attempt, feature, s(3.7, hip=0.7))  # grounded >= ground_settle_s
    assert "downclimb" not in [e.kind for e in events]
    assert "fall" not in [e.kind for e in events]
    assert attempt.finished_attempts[-1].outcome == "downclimb"


def test_fall_summary():
    attempt, feature = make_tracker()
    run(attempt, feature, s(0))
    run(attempt, feature, s(1, contacts={Limb.LF: 1, Limb.RF: 1}))
    run(
        attempt,
        feature,
        s(1.5, hip=0.45, contacts={Limb.LH: 1, Limb.RH: None, Limb.LF: None, Limb.RF: None}),
    )
    run(attempt, feature, s(1.8, hip=0.8, contacts={limb: None for limb in Limb}))
    events = run(attempt, feature, s(3, hip=0.7, contacts={limb: None for limb in Limb}))
    assert "fall" in [e.kind for e in events]
    assert next(e for e in events if e.kind == "fall").data["hold_id"] == 1
    assert attempt.finished_attempts[-1].outcome == "fall"


def test_brief_person_not_found_does_not_close_attempt():
    # 1-2 frames of person_found=False in the middle of an ON_WALL climb must
    # NOT close the attempt.  AttemptTracker.update() returns early with no
    # events when person_found is False, leaving phase and state intact.
    attempt, feature = make_tracker()
    # Ground → start
    run(attempt, feature, s(0))
    run(attempt, feature, s(1, contacts={Limb.LF: 1, Limb.RF: 1, Limb.LH: None, Limb.RH: None}))
    assert attempt.phase == "ON_WALL"

    # Build feature state manually for the dropout frames so run() can still
    # forward a Features object; person_found=False short-circuits before any
    # contact/phase logic.
    def s_no_person(t):
        st = s(t)
        return ClimberState(st.t, st.frame_idx, False, st.keypoints, st.limbs, st.contacts, st.hip)

    feat_tracker = FeatureTracker()
    dropout1 = s_no_person(1.1)
    events1 = attempt.update(dropout1, feat_tracker.update(dropout1))
    assert events1 == []
    assert attempt.phase == "ON_WALL"
    assert attempt.finished_attempts == []

    dropout2 = s_no_person(1.2)
    events2 = attempt.update(dropout2, feat_tracker.update(dropout2))
    assert events2 == []
    assert attempt.phase == "ON_WALL"
    assert attempt.finished_attempts == []

    # Resumes normally: person_found=True again, climbing continues.
    run(attempt, feature, s(1.3, contacts={Limb.LF: 1, Limb.RF: 1}))
    assert attempt.phase == "ON_WALL"
    assert attempt.finished_attempts == []


def test_feet_cut_no_hip_velocity_does_not_set_fall_seen():
    # A feet-cut (both feet lose contact) without downward hip velocity must NOT
    # set the internal _fall_seen flag.  The attempt must therefore close as
    # "downclimb", not "fall", when the climber comes back down slowly.
    attempt, feature = make_tracker()
    # Establish ON_WALL phase with hip raised.
    run(attempt, feature, s(0))
    run(attempt, feature, s(0.8, hip=0.45, contacts={Limb.LF: 1, Limb.RF: 1}))
    run(
        attempt,
        feature,
        s(1.4, hip=0.45, contacts={Limb.LH: 1, Limb.RH: None, Limb.LF: None, Limb.RF: None}),
    )
    assert attempt.phase == "ON_WALL"

    # Feet cut frame: both feet leave contact but hip is NOT moving fast downward.
    # hip_velocity_y is controlled by the hip y-coordinate change.  Keep hip at
    # 0.45 (stationary in y) → hip_velocity_y ≈ 0, well below fall_speed=0.6.
    run(
        attempt,
        feature,
        s(1.5, hip=0.45, contacts={Limb.LH: 1, Limb.RH: None, Limb.LF: None, Limb.RF: None}),
    )
    assert not attempt._fall_seen

    # Climber descends slowly back to ground (hip_velocity_y < fall_speed each step).
    run(attempt, feature, s(2.0, hip=0.55))
    run(attempt, feature, s(2.6, hip=0.70))  # grounded; ground_since starts
    events = run(attempt, feature, s(3.7, hip=0.70))
    assert attempt.finished_attempts[-1].outcome == "downclimb"
    assert "fall" not in [e.kind for e in events]


def test_one_hand_on_top_hold_does_not_trigger_top():
    # Only one hand on a top hold must not cause a "top" event.
    # _is_top() requires BOTH hands to be on a top-role hold.
    attempt, feature = make_tracker()
    # make_tracker() has hold 5 as "top" and hold 1 as "mid".
    run(attempt, feature, s(0))
    run(attempt, feature, s(1, contacts={Limb.LF: 1, Limb.RF: 1, Limb.LH: None, Limb.RH: None}))
    assert attempt.phase == "ON_WALL"

    # LH on top hold (5), RH on a mid hold (1) → not a top.
    for t in (1.5, 2.0, 2.6):
        events = run(
            attempt,
            feature,
            s(t, contacts={Limb.LH: 5, Limb.RH: 1, Limb.LF: 1, Limb.RF: 1}),
        )
        assert "top" not in [e.kind for e in events], f"unexpected top event at t={t}"
        assert attempt.finished_attempts == [], f"attempt unexpectedly closed at t={t}"


def test_holding_the_top_does_not_start_new_attempts():
    # Regression (C's audit): after a top, _finish() sets phase="GROUND" while the
    # climber is still physically on the wall. The very next frame still shows both
    # feet/hands on holds, so the GROUND branch used to call _enter_wall() again,
    # firing attempt_start/top once a second for as long as they stayed up, and
    # scoring the eventual climb-down as a fall. One real top-out must be one attempt.
    cfg = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))
    holds = [
        Hold(1, (0.5, 0.9), (0.45, 0.85, 0.55, 0.95), "start"),
        Hold(2, (0.5, 0.5), (0.45, 0.45, 0.55, 0.55)),
        Hold(3, (0.5, 0.1), (0.45, 0.05, 0.55, 0.15), "top"),
    ]
    tracker = AttemptTracker(HoldMap("x", "pink", holds, (100, 100), "manual"), cfg)
    features = FeatureTracker(cfg["brain"]["move_speed_eps"])
    parts = (
        ("shoulder", 0.3),
        ("elbow", 0.4),
        ("wrist", 0.5),
        ("hip", 0.6),
        ("knee", 0.75),
        ("ankle", 0.9),
    )
    kp = {f"{side}_{part}": (0.5, y, 1.0) for side in ("left", "right") for part, y in parts}
    none = {limb: None for limb in Limb}
    on_top = {Limb.LH: 3, Limb.RH: 3, Limb.LF: 1, Limb.RF: 1}
    kinds, t = [], 0.0
    # 30 fps: stand 1 s, climb and hold the top 4 s, back on the floor 2 s, top again
    for frames, hip_y, contacts in (
        (30, 0.6, none),
        (120, 0.3, on_top),
        (60, 0.6, none),
        (60, 0.3, on_top),
    ):
        for _ in range(frames):
            state = ClimberState(t, 0, True, kp, {}, contacts, (0.5, hip_y))
            kinds += [e.kind for e in tracker.update(state, features.update(state))]
            t += 1 / 30
    # No "high_point" here: attempt 1 has no finished attempt to beat yet (PR #25).
    assert kinds == ["attempt_start", "top", "attempt_start", "top"]
    assert [a.outcome for a in tracker.finished_attempts] == ["top", "top"]
