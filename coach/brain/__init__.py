"""B: features, attempts, detectors, arbiter, memory."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from coach.brain.arbiter import Arbiter
from coach.brain.attempts import AttemptTracker
from coach.brain.detectors import DetectorContext, good_tier_detectors, mvp_detectors
from coach.brain.features import FeatureTracker
from coach.brain.memory import Memory
from coach.contracts import ClimberState, CoachEvent, HoldMap, SpokenLine, Voice

logger = logging.getLogger(__name__)

# This is orchestration plumbing, rather than a cue threshold.  A missing map
# must not prevent colour-only and pose-replay runs from starting.
HOLDMAP_GRACE_S = 1.0


class Brain:
    """Assemble and run the local coach pipeline over a stream of states."""

    def __init__(self, config: dict, bus: Any, voice: Voice) -> None:
        self.config = config
        self.bus = bus
        self.voice = voice
        self._holdmap_queue = bus.subscribe("holdmap")
        self._spoken_queue = bus.subscribe("spoken")

        self.features = FeatureTracker(float(config["brain"]["move_speed_eps"]))
        self.attempts = AttemptTracker(None, config)
        self.detectors = mvp_detectors(config) + good_tier_detectors(config)
        self.memory = Memory(voice, config)
        self._clock_now = [0.0]
        self.arbiter = Arbiter(
            voice,
            config,
            clock=lambda: self._clock_now[0],
            publish=bus.publish,
            still_valid=None,
        )
        self._last_fall_t: float | None = None
        self._seen_summaries = 0

    async def _install_holdmap(self) -> None:
        try:
            holdmap = await asyncio.wait_for(self._holdmap_queue.get(), timeout=HOLDMAP_GRACE_S)
        except TimeoutError:
            logger.debug("No hold map received during startup grace period")
            return
        if not isinstance(holdmap, HoldMap):
            logger.warning("Ignoring invalid holdmap payload: %r", holdmap)
            return
        self.attempts.holdmap = holdmap

    async def _spoken_loop(self) -> None:
        while True:
            spoken = await self._spoken_queue.get()
            if isinstance(spoken, SpokenLine):
                self.memory.on_spoken(spoken)

    def _publish_event(self, event: CoachEvent) -> None:
        self.bus.publish("event", event)

    def _drain_finished_attempts(self) -> None:
        while self._seen_summaries < len(self.attempts.finished_attempts):
            summary = self.attempts.finished_attempts[self._seen_summaries]
            self.memory.on_attempt_end(summary)
            self.bus.publish("attempt", summary)
            self._seen_summaries += 1

    def _process_state(self, state: ClimberState) -> None:
        features = self.features.update(state)
        events = list(self.attempts.update(state, features))
        for event in events:
            self._publish_event(event)
            if event.kind == "attempt_start":
                self.features.reset_highest_hold()

        self._drain_finished_attempts()

        context = DetectorContext(
            self.config,
            self.attempts.holdmap,
            self.attempts.phase == "ON_WALL",
            self._last_fall_t,
            self.memory.crux_hold(),
        )
        for detector in self.detectors:
            detected = detector.update(features, state, context)
            events.extend(detected)
            for event in detected:
                self._publish_event(event)
        observed = self.memory.observe(state, features)
        events.extend(observed)
        for event in observed:
            self._publish_event(event)

        for event in events:
            self.memory.on_event(event)
            self.arbiter.offer(event, self.memory.level_for(event.kind), self.memory.mood())

        self._clock_now[0] = state.t
        spoken = self.arbiter.tick()
        if spoken is not None and spoken.kind == "fall":
            self._last_fall_t = spoken.t

    async def run(self, states: Any) -> None:
        """Consume states forever, processing each state in capture-time order."""
        spoken_task = asyncio.create_task(self._spoken_loop())
        try:
            await self._install_holdmap()
            while True:
                state = await states.get()
                self._process_state(state)
                # Queue-backed producers can make ``get`` complete without
                # suspending; give the spoken-line consumer a turn per frame.
                await asyncio.sleep(0)
        finally:
            spoken_task.cancel()
            await asyncio.gather(spoken_task, return_exceptions=True)


__all__ = ["Brain"]
