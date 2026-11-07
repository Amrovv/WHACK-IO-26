"""Choose which coaching cue, if any, should be spoken."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Mapping
from typing import Any

from coach.contracts import CoachEvent, SpeakRequest, Suppressed, Voice

# Voice.say must be able to start audio within this interval.
SPEECH_START_MARGIN_S = 0.2


class Arbiter:
    """Apply freshness, interruption, cooldown, and budget rules to cues."""

    def __init__(
        self,
        voice: Voice,
        config: Mapping[str, Any] | object,
        clock: Callable[[], float] = time.monotonic,
        publish: Callable[[str, object], None] | None = None,
        still_valid: Callable[[CoachEvent], bool] | None = None,
    ) -> None:
        self.voice = voice
        self._config = config
        self.clock = clock
        self.publish = publish
        self.still_valid = still_valid
        self._candidates: list[tuple[CoachEvent, int, str]] = []
        self._last_spoken: float | None = None
        self._last_spoken_kind: dict[str, float] = {}
        self._spoken_times: deque[float] = deque()

    def offer(self, event: CoachEvent, level: int, mood: str) -> None:
        """Queue a candidate for consideration by the next :meth:`tick`."""
        self._candidates.append((event, level, mood))

    def tick(self) -> CoachEvent | None:
        """Run one decision pass and return the event spoken, if any."""
        now = self.clock()
        candidates = self._candidates
        self._candidates = []
        eligible: list[tuple[CoachEvent, int, str]] = []

        for event, level, mood in candidates:
            if now > event.t + event.ttl_s:
                self._suppress(event, "stale")
                continue
            if self.still_valid is not None and not self.still_valid(event):
                self._suppress(event, "stale")
                continue
            if self.voice.is_speaking():
                if event.priority == 0:
                    self.voice.stop()
                else:
                    self._suppress(event, "busy")
                    continue
            # Priority 0 (top, fall) are the big moments: they interrupt and always
            # speak, so they skip the gap, per-kind cooldown, and budget limits.
            if event.priority != 0:
                if self._last_spoken is not None and now - self._last_spoken < self._value(
                    "global_gap_s"
                ):
                    self._suppress(event, "cooldown")
                    continue
                last_kind = self._last_spoken_kind.get(event.kind)
                if last_kind is not None and now - last_kind < self._value("cooldown_s"):
                    self._suppress(event, "cooldown")
                    continue
                self._trim_budget(now)
                if len(self._spoken_times) >= self._value("budget_lines"):
                    self._suppress(event, "budget")
                    continue
            eligible.append((event, level, mood))

        if not eligible:
            return None

        chosen = min(eligible, key=lambda candidate: (candidate[0].priority, -candidate[0].t))
        chosen_event, level, mood = chosen
        for event, _, _ in eligible:
            if event is not chosen_event:
                self._suppress(event, "lower_priority")

        request = SpeakRequest(
            event=chosen_event,
            level=level,
            mood=mood,
            deadline=now + SPEECH_START_MARGIN_S,
        )
        if not self.voice.say(request):
            self._suppress(chosen_event, "late")
            return None

        self._last_spoken = now
        self._last_spoken_kind[chosen_event.kind] = now
        self._spoken_times.append(now)
        return chosen_event

    def _value(self, name: str) -> float:
        """Read an arbiter setting from a mapping or an attribute-based config."""
        config: Any = self._config
        if isinstance(config, Mapping) and "arbiter" in config:
            config = config["arbiter"]
        elif hasattr(config, "arbiter"):
            config = config.arbiter
        if isinstance(config, Mapping):
            value = config[name]
        else:
            value = getattr(config, name)
        return float(value)

    def _trim_budget(self, now: float) -> None:
        cutoff = now - self._value("budget_window_s")
        while self._spoken_times and self._spoken_times[0] <= cutoff:
            self._spoken_times.popleft()

    def _suppress(self, event: CoachEvent, reason: str) -> None:
        if self.publish is not None:
            self.publish("suppressed", Suppressed(event=event, reason=reason))


if __name__ == "__main__":

    class PrintVoice:
        def say(self, request: SpeakRequest) -> bool:
            print(f"saying {request.event.kind} by {request.deadline:.1f}")
            return True

        def is_speaking(self) -> bool:
            return False

        def stop(self) -> None:
            print("stopped")

    current_time = 10.0
    arbiter = Arbiter(
        PrintVoice(),
        {
            "global_gap_s": 0.0,
            "cooldown_s": 0.0,
            "budget_lines": 3,
            "budget_window_s": 10.0,
        },
        clock=lambda: current_time,
    )
    arbiter.offer(CoachEvent(10.0, "feet_low", 1, 1.0), level=1, mood="neutral")
    arbiter.offer(CoachEvent(10.0, "top", 0, 1.0), level=2, mood="impressed")
    print(f"selected: {arbiter.tick()}")
