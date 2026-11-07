from dataclasses import dataclass

from coach.brain.arbiter import Arbiter
from coach.contracts import CoachEvent, SpeakRequest, Suppressed


@dataclass
class FakeClock:
    now: float = 10.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeVoice:
    def __init__(self, speaking: bool = False, say_result: bool = True) -> None:
        self.speaking = speaking
        self.say_result = say_result
        self.requests: list[SpeakRequest] = []
        self.stop_calls = 0

    def say(self, request: SpeakRequest) -> bool:
        self.requests.append(request)
        if self.say_result:
            self.speaking = True
        return self.say_result

    def is_speaking(self) -> bool:
        return self.speaking

    def stop(self) -> None:
        self.stop_calls += 1
        self.speaking = False


def config(**overrides: float) -> dict[str, float]:
    values = {
        "global_gap_s": 0.0,
        "cooldown_s": 0.0,
        "budget_lines": 10.0,
        "budget_window_s": 10.0,
    }
    values.update(overrides)
    return values


def event(
    kind: str = "feet_low", priority: int = 1, t: float = 10.0, ttl: float = 2.0
) -> CoachEvent:
    return CoachEvent(t=t, kind=kind, priority=priority, ttl_s=ttl)


def make_arbiter(clock: FakeClock, voice: FakeVoice, published: list[object]) -> Arbiter:
    return Arbiter(
        voice,
        config(),
        clock=clock,
        publish=lambda topic, payload: published.append((topic, payload)),
    )


def reasons(published: list[object]) -> list[str]:
    return [payload.reason for _, payload in published]


def test_priority_one_event_is_spoken_within_ttl() -> None:
    clock, voice, published = FakeClock(), FakeVoice(), []
    arbiter = make_arbiter(clock, voice, published)
    candidate = event()
    arbiter.offer(candidate, level=1, mood="neutral")

    assert arbiter.tick() is candidate
    assert voice.requests[0].event is candidate
    assert published == []


def test_expired_event_is_stale() -> None:
    clock, voice, published = FakeClock(12.1), FakeVoice(), []
    arbiter = make_arbiter(clock, voice, published)
    candidate = event(t=10.0, ttl=2.0)
    arbiter.offer(candidate, 1, "neutral")

    assert arbiter.tick() is None
    assert reasons(published) == ["stale"]


def test_still_valid_false_is_stale() -> None:
    clock, voice, published = FakeClock(), FakeVoice(), []
    arbiter = Arbiter(
        voice,
        config(),
        clock=clock,
        publish=lambda t, p: published.append((t, p)),
        still_valid=lambda _: False,
    )
    candidate = event()
    arbiter.offer(candidate, 1, "neutral")

    assert arbiter.tick() is None
    assert reasons(published) == ["stale"]


def test_busy_priority_one_is_suppressed_and_priority_zero_interrupts() -> None:
    clock, voice, published = FakeClock(), FakeVoice(speaking=True), []
    arbiter = make_arbiter(clock, voice, published)
    ordinary = event("feet_low", 1)
    urgent = event("fall", 0)
    arbiter.offer(ordinary, 1, "neutral")
    assert arbiter.tick() is None
    assert reasons(published) == ["busy"]

    arbiter.offer(urgent, 1, "impressed")
    assert arbiter.tick() is urgent
    assert voice.stop_calls == 1


def test_priority_zero_bypasses_gap_cooldown_and_budget() -> None:
    clock, voice, published = FakeClock(), FakeVoice(), []
    arbiter = Arbiter(
        voice,
        config(global_gap_s=10.0, cooldown_s=10.0, budget_lines=1.0, budget_window_s=30.0),
        clock=clock,
        publish=lambda t, p: published.append((t, p)),
    )
    # Spend the whole budget and trip every cooldown with a normal cue.
    first = event("feet_low", 1)
    arbiter.offer(first, 1, "neutral")
    assert arbiter.tick() is first
    voice.speaking = False

    # A priority-0 big moment still fires immediately despite gap/cooldown/budget.
    big_moment = event("top", 0, t=10.1)
    arbiter.offer(big_moment, 2, "impressed")
    assert arbiter.tick() is big_moment
    assert published == []


def test_global_gap_blocks_then_allows_next_speak() -> None:
    clock, voice, published = FakeClock(), FakeVoice(), []
    arbiter = Arbiter(
        voice,
        config(global_gap_s=1.0),
        clock=clock,
        publish=lambda t, p: published.append((t, p)),
    )
    first, second = event("feet_low"), event("arms_bent", t=10.5)
    arbiter.offer(first, 1, "neutral")
    assert arbiter.tick() is first
    voice.speaking = False
    arbiter.offer(second, 1, "neutral")
    assert arbiter.tick() is None
    assert reasons(published) == ["cooldown"]
    clock.advance(1.0)
    arbiter.offer(second, 1, "neutral")
    assert arbiter.tick() is second


def test_per_kind_cooldown_does_not_affect_other_kind() -> None:
    clock, voice, published = FakeClock(), FakeVoice(), []
    arbiter = Arbiter(
        voice,
        config(cooldown_s=2.0),
        clock=clock,
        publish=lambda t, p: published.append((t, p)),
    )
    first = event("feet_low")
    arbiter.offer(first, 1, "neutral")
    assert arbiter.tick() is first
    voice.speaking = False
    same_kind = event("feet_low", t=10.1)
    arbiter.offer(same_kind, 1, "neutral")
    assert arbiter.tick() is None
    other_kind = event("arms_bent", t=10.1)
    arbiter.offer(other_kind, 1, "neutral")
    assert arbiter.tick() is other_kind
    assert reasons(published) == ["cooldown"]


def test_budget_blocks_until_window_rolls_forward() -> None:
    clock, voice, published = FakeClock(), FakeVoice(), []
    arbiter = Arbiter(
        voice,
        config(budget_lines=2.0, budget_window_s=5.0),
        clock=clock,
        publish=lambda t, p: published.append((t, p)),
    )
    for index in range(2):
        candidate = event(f"kind-{index}")
        arbiter.offer(candidate, 1, "neutral")
        assert arbiter.tick() is candidate
        voice.speaking = False
        clock.advance(1.0)
    blocked = event("blocked", t=12.0, ttl=10.0)
    arbiter.offer(blocked, 1, "neutral")
    assert arbiter.tick() is None
    assert reasons(published) == ["budget"]
    clock.advance(3.1)
    arbiter.offer(blocked, 1, "neutral")
    assert arbiter.tick() is blocked


def test_highest_priority_wins_and_other_eligible_candidate_is_lower_priority() -> None:
    clock, voice, published = FakeClock(), FakeVoice(), []
    arbiter = make_arbiter(clock, voice, published)
    ordinary, urgent = event("feet_low", 1), event("fall", 0, t=10.1)
    arbiter.offer(ordinary, 1, "neutral")
    arbiter.offer(urgent, 1, "impressed")

    assert arbiter.tick() is urgent
    assert reasons(published) == ["lower_priority"]


def test_voice_rejecting_request_is_late() -> None:
    clock, voice, published = FakeClock(), FakeVoice(say_result=False), []
    arbiter = make_arbiter(clock, voice, published)
    candidate = event()
    arbiter.offer(candidate, 1, "neutral")

    assert arbiter.tick() is None
    assert reasons(published) == ["late"]


def test_every_suppressed_candidate_is_published() -> None:
    clock, voice, published = FakeClock(12.1), FakeVoice(), []
    arbiter = make_arbiter(clock, voice, published)
    candidates = [event("one", t=10.0, ttl=2.0), event("two", t=10.0, ttl=2.0)]
    for candidate in candidates:
        arbiter.offer(candidate, 1, "neutral")

    arbiter.tick()
    assert len(published) == len(candidates)
    assert all(topic == "suppressed" for topic, _ in published)
    assert all(isinstance(payload, Suppressed) for _, payload in published)


def test_budget_unblocks_when_only_oldest_exits_window() -> None:
    # The budget must unblock as soon as the OLDEST spoken timestamp exits the
    # rolling window, without waiting for all entries to expire.
    clock, voice, published = FakeClock(now=10.0), FakeVoice(), []
    arbiter = Arbiter(
        voice,
        config(budget_lines=3.0, budget_window_s=10.0),
        clock=clock,
        publish=lambda topic, payload: published.append((topic, payload)),
    )

    # Fill the budget: speak 3 events at t=10, t=11, t=12.
    for index in range(3):
        candidate = event(f"kind-fill-{index}", t=clock.now, ttl=60.0)
        arbiter.offer(candidate, 1, "neutral")
        assert arbiter.tick() is candidate, f"fill event {index} should have been spoken"
        voice.speaking = False
        clock.advance(1.0)
    # clock is now at 13.0; spoken_times = [10, 11, 12]; budget full.

    blocked = event("blocked", t=clock.now, ttl=60.0)
    arbiter.offer(blocked, 1, "neutral")
    assert arbiter.tick() is None
    assert reasons(published)[-1] == "budget"

    # Advance to just past when the OLDEST entry (t=10) exits the 10-second window.
    # At now=20.1: cutoff = 20.1 - 10 = 10.1 > 10 → oldest trimmed.
    # spoken_times = [11, 12]; len=2 < 3 → budget available.
    # The two remaining entries (11, 12) are still inside the window (11+10=21 > 20.1).
    clock.advance(7.1)  # 13.0 + 7.1 = 20.1
    blocked2 = event("unblocked", t=clock.now, ttl=60.0)
    arbiter.offer(blocked2, 1, "neutral")
    assert arbiter.tick() is blocked2, (
        "budget should unblock once the oldest entry exits the window, "
        "even if other entries are still inside it"
    )
