"""End-to-end brain wiring test.

Runs a synthetic climb through the WHOLE coach brain in the order the orchestrator
(D's run.py) must use: features -> attempts -> detectors -> memory -> arbiter. The
per-layer tests prove each piece; this proves they fit together and that the
integration contract in design-stages-exp/INTEGRATION-contract.md is correct.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from coach.brain.arbiter import Arbiter
from coach.brain.attempts import AttemptTracker
from coach.brain.detectors import DetectorContext, good_tier_detectors, mvp_detectors
from coach.brain.features import FeatureTracker
from coach.brain.memory import Memory
from coach.contracts import ClimberState, Hold, HoldMap, Limb, SpokenLine


def config() -> dict:
    return yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))


def holdmap() -> HoldMap:
    holds = [
        Hold(1, (0.5, 0.85), (0, 0, 0, 0), "start"),
        Hold(3, (0.5, 0.55), (0, 0, 0, 0), "mid"),
        Hold(5, (0.5, 0.2), (0, 0, 0, 0), "top"),
    ]
    return HoldMap("demo", "pink", holds, (1920, 1080), "manual")


def frame(t: float, *, hip_y: float, contacts: dict) -> ClimberState:
    # Full keypoint set so elbow angles / feet-low / hand-searching can compute.
    kp = {
        "left_shoulder": (0.45, hip_y - 0.25, 1.0),
        "right_shoulder": (0.55, hip_y - 0.25, 1.0),
        "left_elbow": (0.42, hip_y - 0.15, 1.0),
        "right_elbow": (0.58, hip_y - 0.15, 1.0),
        "left_wrist": (0.46, hip_y - 0.30, 1.0),
        "right_wrist": (0.54, hip_y - 0.30, 1.0),
        "left_hip": (0.47, hip_y, 1.0),
        "right_hip": (0.53, hip_y, 1.0),
        "left_knee": (0.47, hip_y + 0.12, 1.0),
        "right_knee": (0.53, hip_y + 0.12, 1.0),
        "left_ankle": (0.47, hip_y + 0.24, 1.0),
        "right_ankle": (0.53, hip_y + 0.24, 1.0),
    }
    limbs = {
        Limb.LH: (0.46, hip_y - 0.30),
        Limb.RH: (0.54, hip_y - 0.30),
        Limb.LF: (0.47, hip_y + 0.24),
        Limb.RF: (0.53, hip_y + 0.24),
    }
    full = {limb: None for limb in Limb}
    full.update(contacts)
    return ClimberState(t, int(t * 10), True, kp, limbs, full, (0.5, hip_y))


class FakeVoice:
    def __init__(self) -> None:
        self.spoken: list = []
        self.debriefs: list = []

    def say(self, req) -> bool:
        self.spoken.append(req)
        return True

    def is_speaking(self) -> bool:
        return False

    def stop(self) -> None:
        pass

    def debrief(self, history, habits, crux_hold) -> None:
        self.debriefs.append((list(history), habits, crux_hold))


def _synthetic_climb() -> list[ClimberState]:
    states: list[ClimberState] = []
    # GROUND: stand and calibrate.
    for t in (0.0, 0.5, 1.0):
        states.append(frame(t, hip_y=0.85, contacts={}))
    # Enter wall: both feet on the start hold.
    states.append(frame(1.5, hip_y=0.85, contacts={Limb.LF: 1, Limb.RF: 1}))
    # Pull on, hands reach the mid hold, hip rises then holds still -> stalling.
    on_wall = {Limb.LF: 1, Limb.RF: 1, Limb.LH: 3, Limb.RH: 3}
    states.append(frame(2.0, hip_y=0.55, contacts=on_wall))
    for t in (2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0):
        states.append(frame(t, hip_y=0.55, contacts=on_wall))  # hip fixed -> static
    # Reach the top hold with both hands and hold for > top_hold_s.
    top = {Limb.LF: 3, Limb.RF: 3, Limb.LH: 5, Limb.RH: 5}
    for t in (7.5, 8.0, 8.5, 9.0):
        states.append(frame(t, hip_y=0.25, contacts=top))
    return states


def test_full_brain_pipeline_on_a_synthetic_top():
    cfg = config()
    hmap = holdmap()
    voice = FakeVoice()
    ft = FeatureTracker(float(cfg["brain"]["move_speed_eps"]))
    at = AttemptTracker(hmap, cfg)
    detectors = mvp_detectors(cfg) + good_tier_detectors(cfg)
    mem = Memory(voice, cfg)
    clock_now = [0.0]
    arbiter = Arbiter(voice, cfg, clock=lambda: clock_now[0])

    emitted_kinds: set[str] = set()
    spoken_kinds: set[str] = set()
    last_fall_t: float | None = None
    seen_summaries = 0

    for state in _synthetic_climb():
        features = ft.update(state)
        events = list(at.update(state, features))

        # Close any finished attempts into memory (contract: before querying memory).
        while seen_summaries < len(at.finished_attempts):
            mem.on_attempt_end(at.finished_attempts[seen_summaries])
            seen_summaries += 1

        on_wall = at.phase == "ON_WALL"
        ctx = DetectorContext(cfg, hmap, on_wall, last_fall_t, mem.crux_hold())
        for det in detectors:
            events += det.update(features, state, ctx)
        events += mem.observe(state, features)

        for ev in events:
            emitted_kinds.add(ev.kind)
            mem.on_event(ev)
            arbiter.offer(ev, mem.level_for(ev.kind), mem.mood())

        clock_now[0] = state.t
        spoken = arbiter.tick()
        if spoken is not None:
            spoken_kinds.add(spoken.kind)
            if spoken.kind == "fall":
                last_fall_t = spoken.t
            # Simulate C publishing the spoken line back to the brain.
            mem.on_spoken(
                SpokenLine(t_event=spoken.t, t_audio_start=state.t, kind=spoken.kind, text="")
            )

    # The chain produced the key events...
    assert "attempt_start" in emitted_kinds
    assert "stalling" in emitted_kinds
    assert "top" in emitted_kinds
    # ...the arbiter actually spoke, and a P0 big moment (top) always gets through.
    assert spoken_kinds
    assert "top" in spoken_kinds
    # ...the attempt was recorded with the right outcome and high point...
    assert len(mem.history) == 1
    assert mem.history[0].outcome == "top"
    assert mem.history[0].high_hold_id == 5
    # ...habit memory updated so the next attempt would be more specific.
    assert mem.level_for("stalling") >= 1
    # ...and the debrief fired exactly once for the finished attempt.
    assert len(voice.debriefs) == 1
