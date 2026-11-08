"""Debrief content: the compact attempt JSON, the Gemini prompt and stream, the template fallback.

Everything here is about what gets said. Voice owns when it is said: filler, playback and cuts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import AsyncGenerator, Callable, Iterable, Sequence
from contextlib import aclosing
from dataclasses import dataclass

from coach.bus import to_jsonable
from coach.contracts import AttemptSummary
from coach.voice.phrasebank import has_banned_phrase, persona_instruction, word_count

log = logging.getLogger(__name__)

HISTORY_ATTEMPTS = 3  # attempts sent to the LLM
MAX_SENTENCES = 3
DEFAULT_MOOD = "neutral"
SENTENCE_END_RE = re.compile(r"[.!?]+[\"'”’)\]]*\s+|\n+")

NO_ATTEMPTS_LINE = "No attempts yet, so get on the wall."
DEFAULT_TIP = "Next go, place your feet before you reach."
HABIT_TIPS = {
    "feet_low": "Next go, bring your feet up before you reach.",
    "arms_bent": "Next go, hang on straight arms between moves.",
    "hand_searching": "Next go, pick your next hold before you let go.",
    "stalling": "Next go, plan the sequence on the ground and keep moving.",
    "feet_cut": "Next go, keep your feet pressed on through the big moves.",
}

LLMStream = Callable[[str], AsyncGenerator[str]]  # prompt -> text chunks


@dataclass(frozen=True)
class DebriefLLM:
    stream: LLMStream
    first_sentence_timeout_s: float
    max_words: int  # per sentence
    banned: Sequence[str]


def debrief_payload(
    history: list[AttemptSummary],
    habits: dict,
    crux_hold: int | None,
    places: dict[int, str] | None = None,
) -> dict:
    """The JSON both the LLM and the template work from.

    `places` (phrasebank.hold_places) names holds by position; without it holds are numbers.
    """
    counters = {key: value for key, value in habits.items() if key != "mood"}
    return {
        "attempts": to_jsonable(history[-HISTORY_ATTEMPTS:]),
        "habits": to_jsonable(counters),
        "crux_hold": crux_hold,
        "mood": habits.get("mood", DEFAULT_MOOD),
        "places": {int(hold): name for hold, name in (places or {}).items()},
    }


def _llm_view(payload: dict) -> dict:
    """The payload as the LLM sees it: with places known, hold numbers become their names, so
    the model has no number to say."""
    places = payload.get("places") or {}
    view = {key: value for key, value in payload.items() if key != "places"}
    if not places:
        return view

    def name(hold: int | None) -> str | None:
        return None if hold is None else places.get(hold, "a hold")

    view["attempts"] = [
        {
            **{k: v for k, v in attempt.items() if k not in ("high_hold_id", "stalls")},
            "high_point": name(attempt.get("high_hold_id")),
            "stalls": [[name(hold), seconds] for hold, seconds in attempt.get("stalls", [])],
        }
        for attempt in payload["attempts"]
    ]
    view.pop("crux_hold", None)
    view["crux"] = name(payload.get("crux_hold"))
    return view


def debrief_prompt(payload: dict, max_words: int) -> str:
    has_earlier = len(payload["attempts"]) > 1
    by_place = bool(payload.get("places"))
    earlier = "Refer to at least one earlier attempt by its number or by a hold."
    holds = (
        "Use only facts from the data. Name holds only as the data does, e.g. 'the high hold'; "
        "never say a hold number."
        if by_place
        else "Use only facts from the data. Hold numbers count up from the bottom of the wall."
    )
    rules = [
        f"At most {MAX_SENTENCES} sentences, each at most {max_words} words.",
        *([earlier] * has_earlier),
        "Give exactly one actionable tip.",
        "Stay in character.",
        "No greetings.",
        "Plain spoken text: no lists, no markdown.",
        holds,
    ]
    data = json.dumps(_llm_view(payload), separators=(",", ":"), default=str)
    bullets = "\n".join(f"- {rule}" for rule in rules)
    return f"Debrief the climber on their latest attempt, the last one listed.\n{bullets}\n{data}"


def reject_reason(sentence: str, max_words: int, banned: Iterable[str]) -> str | None:
    """Why a debrief sentence can't be spoken, or None if it can.

    Not phrasebank.reject_reason: debriefs need the numbers and sides that it refuses.
    """
    words = word_count(sentence)
    if words == 0:
        return "empty"
    if words > max_words:
        return "too long"
    if has_banned_phrase(sentence, banned):
        return "banned phrase"
    return None


class SentenceSplitter:
    """Cuts streamed text into sentences. An ending counts once whitespace follows it, so "3.5"
    survives and a chunk boundary can't split a sentence."""

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, chunk: str) -> list[str]:
        self._buffer += chunk
        sentences = []
        start = 0
        for match in SENTENCE_END_RE.finditer(self._buffer):
            sentences.append(self._buffer[start : match.end()].strip())
            start = match.end()
        self._buffer = self._buffer[start:]
        return [sentence for sentence in sentences if sentence]

    def flush(self) -> list[str]:
        rest, self._buffer = self._buffer.strip(), ""
        return [rest] if rest else []


async def stream_sentences(chunks: AsyncGenerator[str]) -> AsyncGenerator[str]:
    splitter = SentenceSplitter()
    async with aclosing(chunks):
        async for chunk in chunks:
            for sentence in splitter.feed(chunk):
                yield sentence
    for sentence in splitter.flush():
        yield sentence


async def stream_debrief(llm: DebriefLLM, payload: dict, emit: Callable[[str], None]) -> int:
    """Emit usable sentences as they arrive and return how many; 0 means use the template.

    The timeout restarts after each usable sentence, so a stream that hangs later can't hold the
    debrief open. An error after the first sentence keeps what was said: the template would only
    repeat it.
    """
    timeout_s = llm.first_sentence_timeout_s
    loop = asyncio.get_running_loop()
    emitted = 0
    try:
        chunks = llm.stream(debrief_prompt(payload, llm.max_words))
        async with (
            asyncio.timeout(timeout_s) as limit,
            aclosing(stream_sentences(chunks)) as sentences,
        ):
            async for sentence in sentences:
                if reason := reject_reason(sentence, llm.max_words, llm.banned):
                    log.info("dropped debrief sentence (%s): %r", reason, sentence)
                    continue
                emit(sentence)
                emitted += 1
                if emitted == MAX_SENTENCES:
                    break
                limit.reschedule(loop.time() + timeout_s)
    except Exception as exc:  # includes TimeoutError; cancellation still gets through
        log.warning("LLM debrief ended after %d sentences: %r", emitted, exc)
    return emitted


def _hold(payload: dict, hold: int) -> str:
    """A hold as spoken: "the high hold" when places are known, else "hold 4"."""
    name = (payload.get("places") or {}).get(hold)
    return f"the {name}" if name else f"hold {hold}"


def _result(payload: dict, attempt: dict) -> str:
    number, hold = attempt["attempt_no"], attempt["high_hold_id"]
    if attempt["outcome"] == "top":
        return f"Attempt {number}, you topped it."
    if hold is None:
        return f"Attempt {number}, you never got established."
    verb = "came down from" if attempt["outcome"] == "downclimb" else "fell at"
    return f"Attempt {number}, you {verb} {_hold(payload, hold)}."


def _comparison(payload: dict, latest: dict, earlier: dict) -> str:
    number, then, now = earlier["attempt_no"], earlier["high_hold_id"], latest["high_hold_id"]
    by_place = bool(payload.get("places"))
    if then is None:
        return f"Attempt {number} never reached a hold."
    if now is None:
        return f"Attempt {number} got to {_hold(payload, then)}."
    if now > then:
        if by_place:  # "beats the high hold" is unclear when both holds share a name
            return f"That's higher than attempt {number}."
        return f"That beats hold {then} on attempt {number}."
    if now == then:
        return f"Same high point as attempt {number}."
    return f"Attempt {number} got to {_hold(payload, then)}, so that's a step back."


def _worst_habit(habits: dict) -> str | None:
    """The tippable habit seen on the most attempts, then the most often overall.

    Counters look like {"feet_low": {"attempts": 2, "total": 5}}, from Memory.habits().as_dict().
    Anything else is skipped: the template is the last resort, so it must not raise.
    """
    ranked = {
        habit: (counts.get("attempts", 0), counts.get("total", 0))
        for habit, counts in habits.items()
        if habit in HABIT_TIPS and isinstance(counts, dict)
    }
    ranked = {habit: rank for habit, rank in ranked.items() if any(rank)}
    return max(ranked, key=ranked.__getitem__) if ranked else None


def _tip(payload: dict) -> str:
    if payload["crux_hold"] is not None:
        return (
            f"Next go, set your feet before you reach for {_hold(payload, payload['crux_hold'])}."
        )
    stalls = payload["attempts"][-1]["stalls"]
    if stalls:
        hold = max(stalls, key=lambda stall: stall[1])[0]
        return f"Next go, plan your move before you reach {_hold(payload, hold)}."
    return HABIT_TIPS.get(_worst_habit(payload["habits"]), DEFAULT_TIP)


def template_sentences(payload: dict) -> list[str]:
    """A debrief without the LLM: the result, a comparison if there is one, and one tip."""
    attempts = payload["attempts"]
    if not attempts:
        return [NO_ATTEMPTS_LINE]
    sentences = [_result(payload, attempts[-1])]
    if len(attempts) > 1:
        sentences.append(_comparison(payload, attempts[-1], attempts[-2]))
    return [*sentences, _tip(payload)]


def make_gemini_stream(api_key: str, model: str, system_instruction: str) -> LLMStream:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)  # the closure keeps it alive; dropping it closes aio
    config = types.GenerateContentConfig(system_instruction=system_instruction)

    async def stream(prompt: str) -> AsyncGenerator[str]:
        response = await client.aio.models.generate_content_stream(
            model=model, contents=prompt, config=config
        )
        async with aclosing(response):
            async for chunk in response:
                if chunk.text:
                    yield chunk.text

    return stream


def make_debrief_llm(voice_cfg: dict, persona: dict) -> DebriefLLM | None:
    """The configured LLM, or None (template debriefs) if GEMINI_API_KEY isn't set."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        log.warning("GEMINI_API_KEY is not set: debriefs will use the template")
        return None
    return DebriefLLM(
        stream=make_gemini_stream(api_key, voice_cfg["llm_model"], persona_instruction(persona)),
        first_sentence_timeout_s=voice_cfg["debrief_first_sentence_timeout_s"],
        max_words=persona["max_words"]["debrief_sentence"],
        banned=persona["banned_phrases"],
    )


HABITS_DEMO = {"feet_low": {"attempts": 2, "total": 3}}


def main() -> None:
    history = [
        AttemptSummary(1, 21.0, 4, "fall", [(4, 6.5)], ["feet_low"], []),
        AttemptSummary(2, 34.0, 6, "fall", [(6, 8.0)], ["feet_low"], ["feet_low"]),
    ]
    payload = debrief_payload(history, {**HABITS_DEMO, "mood": "exasperated"}, 6)
    log.info("prompt:\n%s", debrief_prompt(payload, 20))
    log.info("template: %s", " ".join(template_sentences(payload)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
