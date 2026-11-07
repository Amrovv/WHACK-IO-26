"""Voice that prints instead of speaking; counts as speaking for a time per word."""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable

from coach.contracts import AttemptSummary, CoachEvent, SpeakRequest

log = logging.getLogger(__name__)

SECONDS_PER_WORD = 0.35


class PrintVoice:
    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        seconds_per_word: float = SECONDS_PER_WORD,
    ) -> None:
        self._clock = clock
        self._spw = seconds_per_word
        self._busy_until = float("-inf")

    @staticmethod
    def _text(event: CoachEvent) -> str:
        return str(event.data.get("text") or event.kind.replace("_", " "))

    def say(self, req: SpeakRequest) -> bool:
        now = self._clock()
        if now > req.deadline:
            return False
        text = self._text(req.event)
        print(f"[SPEAK] {req.event.kind} L{req.level} {req.mood}: {text}", flush=True)
        self._busy_until = now + len(text.split()) * self._spw
        return True

    def is_speaking(self) -> bool:
        return self._clock() < self._busy_until

    def stop(self) -> None:
        self._busy_until = float("-inf")

    async def debrief(
        self, history: list[AttemptSummary], habits: dict, crux_hold: int | None
    ) -> None:
        text = f"Debrief after {len(history)} attempts, crux hold {crux_hold}."
        print(f"[DEBRIEF] {text}", flush=True)
        self._busy_until = self._clock() + len(text.split()) * self._spw


def main() -> None:
    voice = PrintVoice()
    now = time.monotonic()
    ev = CoachEvent(
        t=now, kind="feet_low", priority=1, ttl_s=1.2, data={"text": "Feet up, you're dangling"}
    )
    voice.say(SpeakRequest(ev, 1, "neutral", now + 1))
    while voice.is_speaking():
        time.sleep(0.1)
    log.info("done")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    argparse.ArgumentParser().parse_args()
    main()
