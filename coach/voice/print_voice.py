"""Voice that prints instead of speaking; counts as speaking for a time per word.

With a bus it publishes "spoken" and "debrief_text" like the real Voice, so B's compliance and
D's dashboard work in print mode. With a phrase bank it prints the persona line the real Voice
would pick (same slots, fallbacks and direction preference), so a print run reads like the demo.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import time
from collections.abc import Callable
from pathlib import Path

from coach.bus import Bus
from coach.contracts import AttemptSummary, CoachEvent, HoldMap, SpeakRequest, SpokenLine
from coach.voice.debrief import debrief_payload, template_sentences
from coach.voice.phrasebank import LinePicker, slot_values, stitch

log = logging.getLogger(__name__)

SECONDS_PER_WORD = 0.35
BANK_PATH = Path("data/phrasebank.json")


class PrintVoice:
    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        seconds_per_word: float = SECONDS_PER_WORD,
        bus: Bus | None = None,
        bank: dict | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._clock = clock
        self._spw = seconds_per_word
        self._busy_until = float("-inf")
        self._bus = bus
        self._lines = LinePicker(bank, rng) if bank else None
        self._holdmaps: asyncio.Queue[HoldMap] | None = bus.subscribe("holdmap") if bus else None
        self._route_colour: str | None = None

    def _text(self, req: SpeakRequest) -> str:
        event = req.event
        if self._lines is not None:
            values = slot_values(event.data, self._route_colour)
            line = self._lines.pick(event.kind, req.level, req.mood, values)
            if line is not None:
                return stitch(line, values)[1]
            log.warning("no usable line for %s L%d %s", event.kind, req.level, req.mood)
        return str(event.data.get("text") or event.kind.replace("_", " "))

    def say(self, req: SpeakRequest) -> bool:
        now = self._clock()
        if now > req.deadline:
            return False
        while self._holdmaps is not None and not self._holdmaps.empty():
            self._route_colour = self._holdmaps.get_nowait().colour
        text = self._text(req)
        print(f"[SPEAK] {req.event.kind} L{req.level} {req.mood}: {text}", flush=True)
        self._busy_until = now + len(text.split()) * self._spw
        if self._bus is not None:
            self._bus.publish("spoken", SpokenLine(req.event.t, now, req.event.kind, text))
        return True

    def is_speaking(self) -> bool:
        return self._clock() < self._busy_until

    def stop(self) -> None:
        self._busy_until = float("-inf")

    def close(self) -> None:
        self.stop()

    async def debrief(
        self, history: list[AttemptSummary], habits: dict, crux_hold: int | None
    ) -> None:
        """Print the template debrief (no LLM), one sentence at a time."""
        sentences = template_sentences(debrief_payload(history, habits, crux_hold))
        for sentence in sentences:
            print(f"[DEBRIEF] {sentence}", flush=True)
            if self._bus is not None:
                self._bus.publish("debrief_text", sentence)
        words = sum(len(sentence.split()) for sentence in sentences)
        self._busy_until = self._clock() + words * self._spw


def make_print_voice(cfg: dict, bus: Bus, no_llm: bool = True) -> PrintVoice:
    """Build a PrintVoice the way run.py builds voices: same signature as make_voice.

    Uses data/phrasebank.json when it exists, else prints event text. Never calls an LLM.
    """
    bank = json.loads(BANK_PATH.read_text(encoding="utf-8")) if BANK_PATH.exists() else None
    if bank is None:
        log.warning("%s not found; printing event names instead of persona lines", BANK_PATH)
    return PrintVoice(bus=bus, bank=bank)


def main() -> None:
    bus = Bus()
    bank = json.loads(BANK_PATH.read_text(encoding="utf-8")) if BANK_PATH.exists() else None
    voice = PrintVoice(bus=bus, bank=bank)
    spoken = bus.subscribe("spoken")
    now = time.monotonic()
    data = {"limb": "LH", "hold_id": 7, "dir": "up and left"}
    ev = CoachEvent(t=now, kind="hand_searching", priority=1, ttl_s=1.2, data=data)
    voice.say(SpeakRequest(ev, 2, "neutral", now + 1))
    log.info("published: %s", spoken.get_nowait())
    while voice.is_speaking():
        time.sleep(0.1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    argparse.ArgumentParser().parse_args()
    main()
