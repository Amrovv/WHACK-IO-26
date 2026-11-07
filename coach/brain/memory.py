"""Cross-attempt memory, cue compliance, mood and the debrief trigger (B5).

The arbiter and detectors decide what to say *now*; this module remembers what
happened across attempts so the coach can get more specific (``level_for``),
change tone (``mood``), warn before the hard move (``crux_hold``), and fire a
told-you-so when advice is followed and works. It also triggers the between-
attempt debrief exactly once per attempt.

All timing is driven by ``state.t`` (capture time); nothing here calls the wall
clock, so replays and tests stay deterministic.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from coach.brain.attempts import _value, make_event
from coach.brain.features import Features
from coach.contracts import AttemptSummary, ClimberState, CoachEvent, Limb, SpokenLine, Voice

logger = logging.getLogger(__name__)

# Cue kinds whose compliance we can check from pose alone.
WATCHED_KINDS = ("arms_bent", "feet_low", "hand_searching", "stalling")


def _brain(config: Any, name: str) -> Any:
    return _value(config, "brain", name)


@dataclass
class HabitCounters:
    """How often each event kind has fired, by attempts and in total."""

    attempts_with_kind: dict[str, int] = field(default_factory=dict)
    total_by_kind: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, dict[str, int]]:
        kinds = set(self.attempts_with_kind) | set(self.total_by_kind)
        return {
            kind: {
                "attempts": self.attempts_with_kind.get(kind, 0),
                "total": self.total_by_kind.get(kind, 0),
            }
            for kind in sorted(kinds)
        }


@dataclass
class _Watch:
    """A spoken cue we are watching to see if the climber complies."""

    kind: str
    start_t: float
    data: dict
    baseline_foot_y: float | None
    followed: bool = False
    followed_t: float | None = None
    progress_base_hold: int = 0


def _lowest_foot_y(state: ClimberState) -> float | None:
    feet = [state.limbs.get(Limb.LF), state.limbs.get(Limb.RF)]
    ys = [point[1] for point in feet if point is not None]
    return max(ys) if ys else None


def _highest_contact(state: ClimberState) -> int | None:
    held = [hold for hold in state.contacts.values() if hold is not None]
    return max(held) if held else None


class Memory:
    """Remember attempts, track compliance, pick mood, and trigger debriefs."""

    def __init__(self, voice: Voice, config: Any) -> None:
        self.voice = voice
        self.config = config
        self.history: list[AttemptSummary] = []
        self._habit_attempts: dict[str, int] = defaultdict(int)
        self._habit_total: dict[str, int] = defaultdict(int)
        self._kinds_this_attempt: set[str] = set()
        self._stall_seconds: dict[int, float] = defaultdict(float)
        self._falls_at: dict[int, int] = defaultdict(int)
        self._cues_given = 0
        self._cues_followed = 0
        self._attempt_compliance: list[tuple[int, int]] = []
        self._watches: list[_Watch] = []
        self._recent_events: list[CoachEvent] = []
        self._last_state: ClimberState | None = None
        self._last_t: float | None = None
        self._debriefed_attempt = -1

    # ---- feeds (called by the orchestrator each tick / on each bus message) ----

    def on_event(self, event: CoachEvent) -> None:
        """Record an emitted cue for habit counting and spoken-line matching."""
        self._recent_events.append(event)
        self._kinds_this_attempt.add(event.kind)
        self._habit_total[event.kind] += 1

    def on_spoken(self, spoken: SpokenLine) -> None:
        """A line was actually spoken; start a compliance watch for watched cues."""
        if spoken.kind not in WATCHED_KINDS:
            return
        self._cues_given += 1
        data: dict = {}
        for event in reversed(self._recent_events):
            if event.kind == spoken.kind and abs(event.t - spoken.t_event) <= 1e-6:
                data = event.data
                break
        baseline_foot = _lowest_foot_y(self._last_state) if self._last_state else None
        self._watches.append(
            _Watch(
                kind=spoken.kind, start_t=spoken.t_event, data=data, baseline_foot_y=baseline_foot
            )
        )

    def observe(self, state: ClimberState, features: Features) -> list[CoachEvent]:
        """Advance the live view: stall bookkeeping, compliance, told-you-so."""
        events: list[CoachEvent] = []
        dt = state.t - self._last_t if self._last_t is not None else 0.0

        if dt > 0 and features.hip_static_s > 0:
            ref = _highest_contact(state)
            if ref is not None:
                self._stall_seconds[ref] += dt

        window = float(_brain(self.config, "compliance_window_s"))
        progress = float(_brain(self.config, "compliance_progress_s"))
        for watch in list(self._watches):
            if not watch.followed:
                if state.t - watch.start_t > window:
                    self._watches.remove(watch)
                    continue
                if self._complied(watch, state, features):
                    watch.followed = True
                    watch.followed_t = state.t
                    watch.progress_base_hold = features.highest_hold_id or 0
                    self._cues_followed += 1
            else:
                if watch.followed_t is not None and state.t - watch.followed_t <= progress:
                    high = features.highest_hold_id
                    if high is not None and high > watch.progress_base_hold:
                        events.append(
                            make_event("told_you_so", state.t, {"cue": watch.kind}, self.config)
                        )
                        self._watches.remove(watch)
                else:
                    self._watches.remove(watch)

        # Keep only recent events (matching + memory stay small).
        self._recent_events = [e for e in self._recent_events if state.t - e.t <= window + 2.0]
        self._last_state = state
        self._last_t = state.t
        return events

    def on_attempt_end(self, summary: AttemptSummary) -> None:
        """Finalise habit counters and compliance, then trigger the debrief once."""
        self.history.append(summary)
        for kind in self._kinds_this_attempt:
            self._habit_attempts[kind] += 1
        self._kinds_this_attempt = set()
        if summary.outcome == "fall" and summary.high_hold_id is not None:
            self._falls_at[summary.high_hold_id] += 1
        self._attempt_compliance.append((self._cues_given, self._cues_followed))
        self._cues_given = 0
        self._cues_followed = 0
        self._watches = []
        self._maybe_debrief(summary)

    def _maybe_debrief(self, summary: AttemptSummary) -> None:
        if self._debriefed_attempt == summary.attempt_no:
            return
        self._debriefed_attempt = summary.attempt_no
        result = self.voice.debrief(list(self.history), self.habits().as_dict(), self.crux_hold())
        if inspect.isawaitable(result):
            try:
                asyncio.get_running_loop().create_task(result)
            except RuntimeError:
                # No running loop (e.g. a sync test harness); the orchestrator owns
                # the loop in production, so just drop the un-awaited coroutine.
                result.close()

    # ---- compliance rules ----

    def _complied(self, watch: _Watch, state: ClimberState, features: Features) -> bool:
        if watch.kind == "stalling":
            return features.hip_speed > float(_brain(self.config, "move_speed_eps"))
        if watch.kind == "arms_bent":
            limit = float(_brain(self.config, "arms_bent_deg")) + float(
                _brain(self.config, "arms_bent_comply_margin_deg")
            )
            angles = features.elbow_angle
            return all(
                angles.get(side) is not None and angles[side] > limit for side in ("left", "right")
            )
        if watch.kind == "feet_low":
            if watch.baseline_foot_y is None or features.leg_length <= 0:
                return False
            lowest = _lowest_foot_y(state)
            if lowest is None:
                return False
            rise = watch.baseline_foot_y - lowest  # y is down: foot rising = y shrinks
            return (
                rise
                >= float(_brain(self.config, "feet_low_comply_rise_ratio")) * features.leg_length
            )
        if watch.kind == "hand_searching":
            limb = watch.data.get("limb")
            hold = watch.data.get("hold_id")
            if limb is None or hold is None:
                return False
            return state.contacts.get(Limb(limb)) == hold
        return False

    # ---- queries ----

    def level_for(self, kind: str) -> int:
        """Specificity 0-2: capped count of earlier attempts in which ``kind`` fired."""
        return min(2, self._habit_attempts.get(kind, 0))

    def crux_hold(self) -> int | None:
        """Hold with the most stall-seconds + 5*falls; None until two attempts."""
        if len(self.history) < 2:
            return None
        holds = set(self._stall_seconds) | set(self._falls_at)
        if not holds:
            return None
        return max(
            holds,
            key=lambda h: self._stall_seconds.get(h, 0.0) + 5 * self._falls_at.get(h, 0),
        )

    def habits(self) -> HabitCounters:
        return HabitCounters(dict(self._habit_attempts), dict(self._habit_total))

    def mood(self) -> str:
        last2 = self._attempt_compliance[-2:]
        given = sum(g for g, _ in last2)
        followed = sum(f for _, f in last2)
        if given > 0 and followed / given > float(_brain(self.config, "mood_impressed_rate")):
            return "impressed"
        exasperated_at = int(_brain(self.config, "mood_exasperated_attempts"))
        if any(count >= exasperated_at for count in self._habit_attempts.values()):
            return "exasperated"
        return "neutral"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Memory ready: feed on_event / on_spoken / observe / on_attempt_end.")
