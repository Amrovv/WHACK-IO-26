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


def test_hip_settle_highpoint_and_downclimb():
    attempt, feature = make_tracker()
    run(attempt, feature, s(0))
    assert not run(attempt, feature, s(0.2, hip=0.45))
    events = run(attempt, feature, s(0.8, hip=0.45))
    assert "attempt_start" in [e.kind for e in events]
    events = run(
        attempt,
        feature,
        s(1, hip=0.45, contacts={Limb.LH: 3, Limb.RH: None, Limb.LF: None, Limb.RF: None}),
    )
    assert "high_point" in [e.kind for e in events]
    run(attempt, feature, s(1.5, hip=0.7))
    events = run(attempt, feature, s(2.6, hip=0.7))
    # downclimb emits no CoachEvent; it only shows up as the summary outcome.
    assert "downclimb" not in [e.kind for e in events]
    assert attempt.finished_attempts[-1].outcome == "downclimb"
    assert attempt.finished_attempts[-1].high_hold_id == 3


def test_downclimb_against_real_config_does_not_crash():
    # Regression: real config/default.yaml has no "downclimb" priority. Emitting a
    # downclimb CoachEvent would KeyError. A downclimb must close quietly instead.
    config = real_config()
    feature = FeatureTracker(float(config["brain"]["move_speed_eps"]))
    attempt = AttemptTracker(None, config)
    run(attempt, feature, s(0))
    run(attempt, feature, s(0.8, hip=0.45))  # hip raised; raised_since starts
    run(
        attempt,
        feature,
        s(1.4, hip=0.45, contacts={Limb.LH: 3, Limb.RH: None, Limb.LF: None, Limb.RF: None}),
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
    run(attempt, feature, s(1.5, hip=0.45, contacts={limb: None for limb in Limb}))
    run(attempt, feature, s(1.8, hip=0.8, contacts={limb: None for limb in Limb}))
    events = run(attempt, feature, s(3, hip=0.7, contacts={limb: None for limb in Limb}))
    assert "fall" in [e.kind for e in events]
    assert attempt.finished_attempts[-1].outcome == "fall"
