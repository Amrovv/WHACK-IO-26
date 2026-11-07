from __future__ import annotations

import asyncio

from coach.brain.features import Features
from coach.brain.memory import Memory
from coach.contracts import AttemptSummary, ClimberState, CoachEvent, Limb, SpokenLine

CONFIG = {
    "brain": {
        "move_speed_eps": 0.02,
        "arms_bent_deg": 140,
        "arms_bent_comply_margin_deg": 10,
        "feet_low_comply_rise_ratio": 0.3,
        "compliance_window_s": 3.0,
        "compliance_progress_s": 5.0,
        "mood_impressed_rate": 0.6,
        "mood_exasperated_attempts": 3,
        "priorities": {"feet_low": 1, "arms_bent": 1, "stalling": 1, "told_you_so": 2},
    },
    "arbiter": {"ttl_s": {"p1": 1.2, "p2": 4.0}},
}


class FakeVoice:
    def __init__(self) -> None:
        self.debriefs: list[tuple] = []

    def say(self, req):  # unused here
        return True

    def is_speaking(self) -> bool:
        return False

    def stop(self) -> None:
        pass

    def debrief(self, history, habits, crux_hold):
        self.debriefs.append((list(history), habits, crux_hold))


def feat(**kw) -> Features:
    base = dict(
        limb_speed={limb: 0.0 for limb in Limb},
        elbow_angle={},
        hip_velocity_y=0.0,
        hip_speed=0.0,
        hip_static_s=0.0,
        arm_length=0.25,
        leg_length=0.3,
        highest_hold_id=None,
        hand_moving_no_contact_s={},
    )
    base.update(kw)
    return Features(**base)


def st(t, *, contacts=None, foot_y=0.9, hip=(0.5, 0.5)) -> ClimberState:
    limbs = {
        Limb.LH: (0.5, 0.4),
        Limb.RH: (0.6, 0.4),
        Limb.LF: (0.4, foot_y),
        Limb.RF: (0.6, foot_y),
    }
    return ClimberState(
        t, int(t * 10), True, {}, limbs, contacts or {limb: None for limb in Limb}, hip
    )


def summary(no, outcome="downclimb", high=None) -> AttemptSummary:
    return AttemptSummary(no, 5.0, high, outcome, [], [], [])


def test_level_for_caps_at_two():
    mem = Memory(FakeVoice(), CONFIG)
    assert mem.level_for("feet_low") == 0
    for n in range(1, 5):
        mem.on_event(CoachEvent(t=float(n), kind="feet_low", priority=1, ttl_s=1.2))
        mem.on_attempt_end(summary(n))
    assert mem.level_for("feet_low") == 2  # capped, not 4


def test_crux_hold_needs_two_attempts_and_scores_stall_plus_falls():
    mem = Memory(FakeVoice(), CONFIG)
    assert mem.crux_hold() is None
    # Attempt 1: linger (hip static) on hold 7 for a few seconds.
    mem.observe(st(0.0, contacts={Limb.LH: 7}), feat(hip_static_s=0.0))
    mem.observe(st(1.0, contacts={Limb.LH: 7}), feat(hip_static_s=1.0))
    mem.observe(st(6.0, contacts={Limb.LH: 7}), feat(hip_static_s=6.0))  # +5s at hold 7
    mem.on_attempt_end(summary(1))
    assert mem.crux_hold() is None  # only one attempt so far
    # Attempt 2: a single fall off hold 5 (worth 5 crux points).
    mem.on_attempt_end(summary(2, outcome="fall", high=5))
    # hold 7: ~5 stall-seconds; hold 5: 5*1 fall points = 5. Hold 7 edges it out.
    assert mem.crux_hold() == 7


def _drive_compliance(mem, kind, comply_frame, *, event_data=None):
    """Speak a cue, then feed a complying frame, then a higher-hold frame."""
    t0 = 10.0
    if event_data is not None:
        mem.on_event(CoachEvent(t=t0, kind=kind, priority=1, ttl_s=1.2, data=event_data))
    mem.observe(st(t0 - 0.1), feat())  # establish last_state baseline
    mem.on_spoken(SpokenLine(t_event=t0, t_audio_start=t0, kind=kind, text="x"))
    followed_events = mem.observe(comply_frame[0], comply_frame[1])
    # progress: a new higher hold within the window
    progressed = mem.observe(st(t0 + 1.0, contacts={Limb.LH: 9}), feat(highest_hold_id=9))
    return followed_events + progressed


def test_compliance_stalling_hip_moves_then_told_you_so():
    mem = Memory(FakeVoice(), CONFIG)
    events = _drive_compliance(mem, "stalling", (st(10.2), feat(hip_speed=0.1, highest_hold_id=3)))
    assert any(e.kind == "told_you_so" for e in events)


def test_compliance_arms_bent_elbows_open():
    mem = Memory(FakeVoice(), CONFIG)
    events = _drive_compliance(
        mem,
        "arms_bent",
        (st(10.2), feat(elbow_angle={"left": 160, "right": 160}, highest_hold_id=3)),
    )
    assert any(e.kind == "told_you_so" for e in events)


def test_compliance_feet_low_foot_rises():
    mem = Memory(FakeVoice(), CONFIG)
    # baseline foot_y 0.9; raise to 0.75 (rise 0.15 >= feet_low_comply_rise_ratio*leg = 0.3*0.3)
    events = _drive_compliance(mem, "feet_low", (st(10.2, foot_y=0.75), feat(highest_hold_id=3)))
    assert any(e.kind == "told_you_so" for e in events)


def test_compliance_hand_searching_touches_suggested_hold():
    mem = Memory(FakeVoice(), CONFIG)
    frame = (st(10.2, contacts={Limb.LH: 4}), feat(highest_hold_id=4))
    events = _drive_compliance(
        mem, "hand_searching", frame, event_data={"limb": "left_hand", "hold_id": 4}
    )
    # CONFIG has no hand_searching priority; it is a WATCHED kind for compliance only,
    # and told_you_so is what gets emitted here.
    told_you_so = next(e for e in events if e.kind == "told_you_so")
    assert told_you_so.data == {"cue": "hand_searching", "hold_id": 9, "limb": "left_hand"}


def test_uncomplied_cue_expires_without_told_you_so():
    mem = Memory(FakeVoice(), CONFIG)
    mem.observe(st(9.9), feat())
    mem.on_spoken(SpokenLine(t_event=10.0, t_audio_start=10.0, kind="stalling", text="move"))
    # hip never moves; past the compliance window -> dropped, no told_you_so
    events = mem.observe(st(13.5), feat(hip_speed=0.0))
    assert events == []


def test_mood_impressed_when_followed():
    mem = Memory(FakeVoice(), CONFIG)
    mem.observe(st(9.9), feat())
    mem.on_spoken(SpokenLine(t_event=10.0, t_audio_start=10.0, kind="feet_low", text="feet"))
    mem.observe(st(10.2, foot_y=0.7), feat(highest_hold_id=1))  # complied
    mem.on_attempt_end(summary(1))
    assert mem.mood() == "impressed"


def test_mood_exasperated_when_habit_repeats():
    mem = Memory(FakeVoice(), CONFIG)
    for n in range(1, 4):
        mem.on_event(CoachEvent(t=float(n), kind="arms_bent", priority=1, ttl_s=1.2))
        mem.on_attempt_end(summary(n))  # no compliance recorded
    assert mem.mood() == "exasperated"


def test_mood_neutral_by_default():
    mem = Memory(FakeVoice(), CONFIG)
    mem.on_event(CoachEvent(t=1.0, kind="feet_low", priority=1, ttl_s=1.2))
    mem.on_attempt_end(summary(1))
    assert mem.mood() == "neutral"


def test_debrief_fires_once_per_attempt_with_history_habits_crux():
    voice = FakeVoice()
    mem = Memory(voice, CONFIG)
    mem.on_event(CoachEvent(t=1.0, kind="feet_low", priority=1, ttl_s=1.2))
    mem.on_attempt_end(summary(1))
    assert len(voice.debriefs) == 1
    history, habits, crux = voice.debriefs[0]
    assert len(history) == 1
    assert habits["feet_low"]["attempts"] == 1
    assert habits["mood"] == "neutral"
    assert crux is None  # only one attempt
    # A second, distinct attempt debriefs again.
    mem.on_attempt_end(summary(2))
    assert len(voice.debriefs) == 2


class AsyncFakeVoice(FakeVoice):
    async def debrief(self, history, habits, crux_hold):
        await asyncio.sleep(0)
        self.debriefs.append((list(history), habits, crux_hold))


async def test_debrief_task_is_retained_until_complete():
    voice = AsyncFakeVoice()
    mem = Memory(voice, CONFIG)

    mem.on_attempt_end(summary(1))

    assert len(mem._debrief_tasks) == 1
    await asyncio.gather(*mem._debrief_tasks)
    assert mem._debrief_tasks == []


def test_summary_records_given_and_followed_cues_per_attempt():
    mem = Memory(FakeVoice(), CONFIG)
    mem.on_event(CoachEvent(t=10.0, kind="stalling", priority=1, ttl_s=1.2))
    mem.observe(st(9.9), feat())
    mem.on_spoken(SpokenLine(t_event=10.0, t_audio_start=10.0, kind="stalling", text="move"))
    mem.observe(st(10.2), feat(hip_speed=0.1, highest_hold_id=3))

    result = summary(1)
    mem.on_attempt_end(result)

    assert result.cues_given == ["stalling"]
    assert result.cues_followed == ["stalling"]


def test_summary_records_stall_seconds_per_attempt():
    mem = Memory(FakeVoice(), CONFIG)
    mem.observe(st(0.0, contacts={Limb.LH: 7}), feat(hip_static_s=0.0))
    mem.observe(st(1.5, contacts={Limb.LH: 7}), feat(hip_static_s=1.0))

    result = summary(1)
    mem.on_attempt_end(result)

    assert result.stalls == [(7, 1.5)]


def test_crux_hold_uses_cumulative_stalls_after_summary_bookkeeping():
    mem = Memory(FakeVoice(), CONFIG)
    mem.observe(st(0.0, contacts={Limb.LH: 7}), feat(hip_static_s=0.0))
    mem.observe(st(6.0, contacts={Limb.LH: 7}), feat(hip_static_s=1.0))
    first = summary(1)
    mem.on_attempt_end(first)

    mem.on_attempt_end(summary(2, outcome="fall", high=5))

    assert first.stalls == [(7, 6.0)]
    assert mem.crux_hold() == 7


def test_crux_hold_tie_is_deterministic():
    # When two holds have exactly equal scores (stall_seconds + 5*falls),
    # crux_hold() must return the same hold on every call.  We don't prescribe
    # WHICH hold wins (Python set iteration order over equal-score keys is
    # implementation-defined), only that the result is consistent.
    mem = Memory(FakeVoice(), CONFIG)

    # Arrange: hold 3 and hold 7 each accumulate exactly 5 stall-seconds and
    # 0 falls → both have score 5.0.  We use two attempts to satisfy the
    # "at least two attempts" guard.
    mem.observe(st(0.0, contacts={Limb.LH: 3}), feat(hip_static_s=0.0))
    mem.observe(st(5.0, contacts={Limb.LH: 3}), feat(hip_static_s=1.0))  # ~5 s at hold 3
    mem.on_attempt_end(summary(1))

    mem.observe(st(10.0, contacts={Limb.LH: 7}), feat(hip_static_s=0.0))
    mem.observe(st(15.0, contacts={Limb.LH: 7}), feat(hip_static_s=1.0))  # ~5 s at hold 7
    mem.on_attempt_end(summary(2))

    # Both holds must have been recorded.
    assert 3 in mem._stall_seconds and 7 in mem._stall_seconds

    # The result must be the same hold on every call.
    first_result = mem.crux_hold()
    assert first_result is not None
    for _ in range(10):
        assert mem.crux_hold() == first_result, "crux_hold() is not deterministic for tied scores"
