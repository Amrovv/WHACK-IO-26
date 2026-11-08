import asyncio
import json
import logging
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from test_voice import FakeSpeaker, FakeStream, FakeTTS, Rig, bank_of

import coach.voice.voice as voice_module
from coach.brain.arbiter import Arbiter
from coach.bus import Bus
from coach.contracts import AttemptSummary, CoachEvent, Limb
from coach.voice import debrief as debrief_module
from coach.voice.debrief import (
    DEFAULT_TIP,
    HABIT_TIPS,
    MAX_SENTENCES,
    NO_ATTEMPTS_LINE,
    DebriefLLM,
    SentenceSplitter,
    debrief_payload,
    debrief_prompt,
    make_gemini_stream,
    reject_reason,
    template_sentences,
)
from coach.voice.phrasebank import reject_reason as phrase_reject_reason
from coach.voice.phrasebank import word_count
from coach.voice.tts import TTS, KokoroTTS, TTSParams
from coach.voice.voice import Voice, make_voice

FILLER = "Right. Listen."
EARLIER_RULE = "earlier attempt"
TIMEOUT_S = 0.05
GAP_S = 0.2  # a pause in the stream, long enough for the speaker to go quiet
WAIT_S = 2.0
LATEST = "Attempt 2, you fell at hold 6."
ARBITER = {"global_gap_s": 0, "cooldown_s": 0, "budget_lines": 9, "budget_window_s": 60}


def counts(attempts: int, total: int) -> dict:
    """One habit, as Memory.habits().as_dict() sends it."""
    return {"attempts": attempts, "total": total}


def attempt(no=1, high=4, outcome="fall", stalls=()) -> AttemptSummary:
    return AttemptSummary(no, 20.0, high, outcome, list(stalls), ["feet_low"], [])


TWO_ATTEMPTS = [attempt(1, 4), attempt(2, 6)]


class TurnTakingSpeaker(FakeSpeaker):
    """Stays busy until finish(), or for one poll if not `held`, like a clip playing out."""

    def __init__(self, clock) -> None:
        super().__init__(clock)
        self.held = False
        self.overlaps = 0

    def play(self, samples):
        self.overlaps += self.speaking
        self.speaking = True
        return super().play(samples)

    def is_speaking(self) -> bool:
        speaking = self.speaking
        if not self.held:
            self.speaking = False
        return speaking

    def finish(self) -> None:
        self.speaking = False

    def stop(self) -> None:
        super().stop()
        self.speaking = False


class FakeLLM:
    """Plays a script of text chunks, float pauses, and exceptions to raise."""

    def __init__(self, *script, timeout_s=TIMEOUT_S, max_words=20, banned=()) -> None:
        self.script = script
        self.prompts: list[str] = []
        self.closed = False
        self.config = DebriefLLM(self.stream, timeout_s, max_words, banned)

    async def stream(self, prompt: str):
        self.prompts.append(prompt)
        try:
            for step in self.script:
                if isinstance(step, Exception):
                    raise step
                if isinstance(step, float):
                    await asyncio.sleep(step)
                else:
                    yield step
        finally:
            self.closed = True


def rig_with(llm: FakeLLM | None = None, rendered=(FILLER,), fillers=(FILLER,), **kwargs) -> Rig:
    return Rig(
        {},
        rendered=list(rendered),
        fillers=fillers,
        debrief_llm=llm.config if llm else None,
        speaker_type=TurnTakingSpeaker,
        **kwargs,
    )


def captions(rig: Rig, queue: asyncio.Queue) -> list[str]:
    out = []
    while not queue.empty():
        out.append(queue.get_nowait())
    return out


async def until(condition) -> None:
    async with asyncio.timeout(WAIT_S):
        while not condition():
            await asyncio.sleep(0.001)


async def run(rig: Rig, history=None, habits=None, crux=None) -> list[str]:
    queue = rig.bus.subscribe("debrief_text")
    history = TWO_ATTEMPTS if history is None else history
    await asyncio.wait_for(rig.voice.debrief(history, habits or {}, crux), WAIT_S)
    return captions(rig, queue)


def template_for(history=None, habits=None, crux=None) -> list[str]:
    history = TWO_ATTEMPTS if history is None else history
    return template_sentences(debrief_payload(history, habits or {}, crux))


# The JSON


def test_payload_keeps_the_last_three_attempts() -> None:
    history = [attempt(no) for no in range(1, 6)]
    payload = debrief_payload(history, {}, None)
    assert [a["attempt_no"] for a in payload["attempts"]] == [3, 4, 5]


def test_payload_reads_the_mood_out_of_habits() -> None:
    payload = debrief_payload([], {"feet_low": counts(2, 3), "mood": "impressed"}, 6)
    assert payload["mood"] == "impressed"
    assert payload["habits"] == {"feet_low": counts(2, 3)}
    assert payload["crux_hold"] == 6


def test_mood_defaults_to_neutral() -> None:
    assert debrief_payload([], {}, None)["mood"] == "neutral"


def test_payload_is_plain_json() -> None:
    payload = debrief_payload([attempt(stalls=[(4, 6.5)])], {Limb.LH: 2}, None)
    assert json.loads(json.dumps(payload)) == payload
    assert payload["habits"] == {"left_hand": 2}


# The prompt


def test_prompt_carries_every_rule_and_the_data() -> None:
    payload = debrief_payload(TWO_ATTEMPTS, {"feet_low": counts(2, 3)}, 6)
    prompt = debrief_prompt(payload, 20)
    assert "At most 3 sentences" in prompt
    assert EARLIER_RULE in prompt
    assert "exactly one actionable tip" in prompt
    assert "in character" in prompt
    assert "No greetings" in prompt
    without_places = {key: value for key, value in payload.items() if key != "places"}
    assert json.dumps(without_places, separators=(",", ":")) in prompt


PLACES = {4: "middle hold", 6: "high hold", 8: "top hold"}


def test_with_places_the_prompt_has_no_hold_numbers() -> None:
    history = [attempt(1, high=4, stalls=[(4, 6.5)]), attempt(2, high=6)]
    prompt = debrief_prompt(debrief_payload(history, {}, 6, PLACES), 20)
    assert "never say a hold number" in prompt
    assert '"high_point":"high hold"' in prompt
    assert '"crux":"high hold"' in prompt
    assert "high_hold_id" not in prompt and "crux_hold" not in prompt


def test_with_places_the_template_names_holds_by_position() -> None:
    history = [attempt(1, high=4), attempt(2, high=6, stalls=[(6, 3.0)])]
    sentences = template_sentences(debrief_payload(history, {}, None, PLACES))
    assert sentences == [
        "Attempt 2, you fell at the high hold.",
        "That's higher than attempt 1.",
        "Next go, plan your move before you reach the high hold.",
    ]


@pytest.mark.parametrize("history", [[], [attempt()]])
def test_prompt_drops_the_earlier_attempt_rule_without_one(history) -> None:
    prompt = debrief_prompt(debrief_payload(history, {}, None), 20)
    assert EARLIER_RULE not in prompt
    assert "exactly one actionable tip" in prompt


def test_the_rule_counts_attempts_in_the_full_history_window() -> None:
    history = [attempt(no) for no in range(1, 6)]
    assert EARLIER_RULE in debrief_prompt(debrief_payload(history, {}, None), 20)


# Splitting


def test_sentences_split_across_chunks() -> None:
    splitter = SentenceSplitter()
    assert splitter.feed("Feet fi") == []
    assert splitter.feed("rst. Again") == ["Feet first."]
    assert splitter.feed("! Why? ") == ["Again!", "Why?"]
    assert splitter.flush() == []


def test_a_decimal_is_not_a_sentence_end() -> None:
    splitter = SentenceSplitter()
    assert splitter.feed("You sat for 3.") == []
    assert splitter.feed("5 seconds. Move.") == ["You sat for 3.5 seconds."]
    assert splitter.flush() == ["Move."]


def test_a_closing_quote_stays_with_its_sentence() -> None:
    assert SentenceSplitter().feed('He said "move." Then go. ') == ['He said "move."', "Then go."]


def test_flush_returns_text_with_no_ending_once() -> None:
    splitter = SentenceSplitter()
    splitter.feed("Feet first")
    assert splitter.flush() == ["Feet first"]
    assert splitter.flush() == []


# Sentence checks


@pytest.mark.parametrize("sentence", ["Left foot on hold 6.", "Third time on the right, twice."])
def test_numbers_and_sides_are_allowed_in_debriefs(sentence: str) -> None:
    assert phrase_reject_reason(sentence, 20, []) == "hard-coded side or number"
    assert reject_reason(sentence, 20, []) is None


def test_debrief_sentences_keep_the_word_limit() -> None:
    assert reject_reason("one two three four", 3, []) == "too long"
    assert reject_reason("one two three", 3, []) is None


def test_banned_phrases_are_caught_whatever_the_punctuation() -> None:
    assert reject_reason("GREAT, job on that.", 20, ["Great job"]) == "banned phrase"


def test_a_sentence_with_no_words_is_rejected() -> None:
    assert reject_reason("...", 20, []) == "empty"


# Template


def test_template_compares_with_the_earlier_attempt() -> None:
    assert template_for(crux=6) == [
        LATEST,
        "That beats hold 4 on attempt 1.",
        "Next go, set your feet before you reach for hold 6.",
    ]


def test_template_without_an_earlier_attempt_has_no_comparison() -> None:
    sentences = template_for([attempt(1, 4)])
    assert sentences == ["Attempt 1, you fell at hold 4.", DEFAULT_TIP]


def test_template_with_no_attempts() -> None:
    assert template_for([]) == [NO_ATTEMPTS_LINE]


@pytest.mark.parametrize(
    ("then", "now", "expected"),
    [
        (6, 6, "Same high point as attempt 1."),
        (7, 6, "Attempt 1 got to hold 7, so that's a step back."),
        (None, 6, "Attempt 1 never reached a hold."),
        (6, None, "Attempt 1 got to hold 6."),
    ],
)
def test_template_comparisons(then, now, expected) -> None:
    assert template_for([attempt(1, then), attempt(2, now)])[1] == expected


def test_tip_uses_the_longest_stall_without_a_crux() -> None:
    history = [attempt(stalls=[(3, 2.0), (5, 7.0)])]
    assert template_for(history)[-1] == "Next go, plan your move before you reach hold 5."


def test_tip_uses_the_worst_habit_without_a_crux_or_stall() -> None:
    habits = {
        "feet_low": counts(1, 2),
        "arms_bent": counts(3, 4),
        "mood": "exasperated",
        "attempt_start": counts(9, 9),  # has no tip
    }
    assert template_for([attempt()], habits)[-1] == HABIT_TIPS["arms_bent"]


def test_habit_on_more_attempts_beats_a_higher_total() -> None:
    habits = {"feet_low": counts(1, 9), "arms_bent": counts(2, 2)}
    assert template_for([attempt()], habits)[-1] == HABIT_TIPS["arms_bent"]


def test_total_breaks_a_tie_on_attempts() -> None:
    habits = {"feet_low": counts(2, 3), "arms_bent": counts(2, 5)}
    assert template_for([attempt()], habits)[-1] == HABIT_TIPS["arms_bent"]


def test_a_habit_counted_only_this_attempt_still_gets_its_tip() -> None:
    assert template_for([attempt()], {"feet_low": counts(0, 2)})[-1] == HABIT_TIPS["feet_low"]


@pytest.mark.parametrize("habits", [{}, {"feet_low": counts(0, 0)}, {"feet_low": 3}])
def test_no_usable_habit_gives_the_default_tip(habits: dict) -> None:
    assert template_for([attempt()], habits)[-1] == DEFAULT_TIP


@pytest.mark.parametrize("outcome", ["top", "fall", "downclimb"])
@pytest.mark.parametrize("high", [None, 1, 25])
@pytest.mark.parametrize("crux", [None, 25])
def test_template_keeps_the_persona_rules(outcome, high, crux) -> None:
    persona = yaml.safe_load(Path("config/personas/coach.yaml").read_text(encoding="utf-8"))
    history = [attempt(1, 25), attempt(2, high, outcome)]
    sentences = template_for(history, {"feet_low": counts(1, 2)}, crux)
    assert len(sentences) <= MAX_SENTENCES
    limit = persona["max_words"]["debrief_sentence"]
    assert all(reject_reason(s, limit, persona["banned_phrases"]) is None for s in sentences)


def test_every_fixed_tip_is_one_sentence_within_the_limit() -> None:
    assert all(word_count(tip) <= 20 for tip in [*HABIT_TIPS.values(), DEFAULT_TIP])


# Filler


async def test_filler_plays_before_the_stream_produces_anything() -> None:
    llm = FakeLLM(1.0, timeout_s=WAIT_S)
    rig = rig_with(llm)
    task = asyncio.create_task(rig.voice.debrief(TWO_ATTEMPTS, {}, None))
    await until(lambda: rig.speaker.played)
    assert len(rig.speaker.played) == 1
    rig.voice.stop()
    await task


async def test_cached_filler_does_not_wait_for_the_tts_worker() -> None:
    rig = rig_with(FakeLLM(1.0, timeout_s=WAIT_S))
    release = threading.Event()
    rig.pool.submit(release.wait, 5)  # Kokoro warming up
    task = asyncio.create_task(rig.voice.debrief(TWO_ATTEMPTS, {}, None))
    await until(lambda: rig.speaker.played)
    assert rig.tts.synthesised == []
    rig.voice.stop()
    release.set()
    await task


async def test_filler_waits_for_the_line_the_arbiter_just_spoke() -> None:
    """Attempt end: Memory schedules the debrief, then the arbiter says fall in the same tick."""
    rig = cuttable(FakeLLM("One here. "))
    queue = rig.bus.subscribe("debrief_text")
    task = asyncio.create_task(rig.voice.debrief(TWO_ATTEMPTS, {}, None))
    assert rig.voice.say(rig.request())
    await asyncio.sleep(voice_module.SPEAKER_POLL_S * 3)
    assert len(rig.speaker.played) == 1  # still only the fall line
    assert rig.speaker.stops == 0
    rig.speaker.held = False
    rig.speaker.finish()
    await asyncio.wait_for(task, WAIT_S)
    assert rig.speaker.overlaps == 0
    assert rig.texts() == ["Feet up."]
    assert captions(rig, queue) == [FILLER, "One here."]


async def test_stream_runs_while_the_filler_waits_its_turn() -> None:
    llm = FakeLLM("One here. ")
    rig = cuttable(llm)
    task = asyncio.create_task(rig.voice.debrief(TWO_ATTEMPTS, {}, None))
    assert rig.voice.say(rig.request())
    await until(lambda: llm.closed)
    assert len(rig.speaker.played) == 1
    rig.speaker.held = False
    rig.speaker.finish()
    await asyncio.wait_for(task, WAIT_S)


async def test_stop_while_the_filler_waits_plays_nothing_more() -> None:
    rig = cuttable(FakeLLM(1.0, timeout_s=WAIT_S))
    queue = rig.bus.subscribe("debrief_text")
    task = asyncio.create_task(rig.voice.debrief(TWO_ATTEMPTS, {}, None))
    assert rig.voice.say(rig.request())
    await asyncio.sleep(voice_module.SPEAKER_POLL_S)
    rig.voice.stop()
    await asyncio.wait_for(task, WAIT_S)
    assert len(rig.speaker.played) == 1
    assert captions(rig, queue) == []


async def test_uncached_filler_is_synthesised_and_goes_first() -> None:
    llm = FakeLLM("Hold 4 again. ")
    rig = rig_with(llm, rendered=[])
    assert await run(rig) == [FILLER, "Hold 4 again."]
    assert rig.tts.synthesised == [FILLER, "Hold 4 again."]


async def test_no_fillers_goes_straight_to_the_debrief() -> None:
    rig = rig_with(FakeLLM("Hold 4 again. "), fillers=())
    assert await run(rig) == ["Hold 4 again."]


# Streaming


async def test_fast_stream_is_captioned_and_played_in_order() -> None:
    llm = FakeLLM("Hold 4 ag", "ain. Attempt 1 was", " better. Feet first.")
    rig = rig_with(llm)
    said = await run(rig)
    assert said == [FILLER, "Hold 4 again.", "Attempt 1 was better.", "Feet first."]
    assert len(rig.speaker.played) == 4
    assert rig.speaker.overlaps == 0


async def test_each_sentence_waits_for_the_one_before_it() -> None:
    rig = rig_with(FakeLLM("One here. Two here. "))
    rig.speaker.held = True
    task = asyncio.create_task(rig.voice.debrief(TWO_ATTEMPTS, {}, None))
    for expected in (1, 2, 3):
        await until(lambda n=expected: len(rig.speaker.played) == n)
        await asyncio.sleep(voice_module.SPEAKER_POLL_S * 2)  # the loop is free; nothing jumps in
        assert len(rig.speaker.played) == expected
        rig.speaker.finish()
    await task
    assert rig.speaker.overlaps == 0


async def test_debrief_returns_only_when_the_last_sentence_ends() -> None:
    rig = rig_with(FakeLLM("One here. "))
    rig.speaker.held = True
    task = asyncio.create_task(rig.voice.debrief(TWO_ATTEMPTS, {}, None))
    await until(lambda: len(rig.speaker.played) == 1)
    rig.speaker.finish()
    await until(lambda: len(rig.speaker.played) == 2)
    await asyncio.sleep(voice_module.SPEAKER_POLL_S * 2)
    assert not task.done()
    rig.speaker.finish()
    await task


async def test_synthesis_runs_on_the_worker_while_the_loop_stays_free() -> None:
    rig = rig_with(FakeLLM("One here. "))
    release = threading.Event()
    rig.tts.on_synth = lambda _text: release.wait(5)
    task = asyncio.create_task(rig.voice.debrief(TWO_ATTEMPTS, {}, None))
    await until(lambda: rig.tts.synthesised)  # this polling only runs if the loop isn't blocked
    assert not task.done()
    release.set()
    await task


async def test_next_sentence_renders_while_the_previous_one_plays() -> None:
    rig = rig_with(FakeLLM("One here. Two here. "))
    rig.speaker.held = True
    task = asyncio.create_task(rig.voice.debrief(TWO_ATTEMPTS, {}, None))
    await until(lambda: len(rig.tts.synthesised) == 1)
    rig.speaker.finish()
    await until(lambda: len(rig.tts.synthesised) == 2)
    assert len(rig.speaker.played) == 2  # the filler and sentence one; two is ready behind it
    rig.speaker.held = False
    await task


async def test_a_long_reply_is_cut_at_three_sentences_and_the_stream_closed() -> None:
    llm = FakeLLM("One here. Two here. Three here. Four here. ", 1.0, timeout_s=WAIT_S)
    rig = rig_with(llm)
    said = await run(rig)
    assert said == [FILLER, "One here.", "Two here.", "Three here."]
    assert llm.closed


async def test_text_with_no_final_punctuation_is_still_spoken() -> None:
    assert await run(rig_with(FakeLLM("Hold 4 again"))) == [FILLER, "Hold 4 again"]


async def test_prompt_reaches_the_llm_with_the_mood() -> None:
    llm = FakeLLM("Hold 4 again. ")
    await run(rig_with(llm), habits={"mood": "exasperated"})
    assert '"mood":"exasperated"' in llm.prompts[0]
    assert "at most 20 words" in llm.prompts[0]


async def test_bad_sentences_are_dropped_and_good_ones_kept() -> None:
    llm = FakeLLM(
        "Great job on hold 4. Left foot on hold 4. one two three four five six. ",
        max_words=5,
        banned=["Great job"],
    )
    assert await run(rig_with(llm)) == [FILLER, "Left foot on hold 4."]


# Fallbacks


@pytest.mark.parametrize(
    "script",
    [
        pytest.param([1.0], id="slow"),
        pytest.param([RuntimeError("503")], id="error"),
        pytest.param([], id="empty"),
        pytest.param(["Great job. Great job. "], id="all rejected"),
        pytest.param(["Half a sent", RuntimeError("503")], id="error mid sentence"),
        pytest.param(["Half a sent", 1.0], id="hangs mid sentence"),
    ],
)
async def test_template_plays_when_no_usable_first_sentence_arrives(script) -> None:
    llm = FakeLLM(*script, banned=["Great job"])
    rig = rig_with(llm)
    assert await run(rig) == [FILLER, *template_for()]
    assert llm.closed or not script


async def test_slow_stream_that_beats_the_timeout_is_used() -> None:
    llm = FakeLLM(0.01, "Hold 4 again. ", timeout_s=WAIT_S)
    assert await run(rig_with(llm)) == [FILLER, "Hold 4 again."]


async def test_rejected_sentence_does_not_restart_the_first_sentence_timeout() -> None:
    llm = FakeLLM("Great job. ", 0.03, "Great job. ", 0.03, "Hold 4. ", banned=["Great job"])
    assert await run(rig_with(llm)) == [FILLER, *template_for()]


async def test_error_after_a_sentence_keeps_it_and_skips_the_template() -> None:
    llm = FakeLLM("Hold 4 again. Half a", RuntimeError("503"))
    assert await run(rig_with(llm)) == [FILLER, "Hold 4 again."]


async def test_a_stream_hanging_after_a_sentence_ends_the_debrief() -> None:
    llm = FakeLLM("Hold 4 again. ", 1.0)
    assert await run(rig_with(llm)) == [FILLER, "Hold 4 again."]
    assert llm.closed


async def test_without_an_llm_the_template_plays() -> None:
    assert await run(rig_with()) == [FILLER, *template_for()]


async def test_first_attempt_template_plays_end_to_end() -> None:
    history = [attempt(1, 4)]
    assert await run(rig_with(), history) == [FILLER, *template_for(history)]


async def test_no_attempts_skips_the_llm() -> None:
    llm = FakeLLM("Invented history. ")
    assert await run(rig_with(llm), []) == [FILLER, NO_ATTEMPTS_LINE]
    assert llm.prompts == []


async def test_failed_synthesis_loses_one_sentence_only(caplog) -> None:
    rig = rig_with(FakeLLM("Bad one. Good one. "))

    def boom(text: str) -> None:
        if text == "Bad one.":
            raise RuntimeError("engine down")

    rig.tts.on_synth = boom
    assert await run(rig) == [FILLER, "Good one."]
    assert "debrief synthesis failed" in caplog.text


async def test_kokoro_exiting_while_it_loads_loses_sentences_not_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Seen live: spaCy's model download failed inside Kokoro's load and called sys.exit(), which
    stopped run.py's event loop, so the dashboard froze before any debrief text reached it."""

    def kpipeline(lang_code: str) -> None:
        sys.exit(1)

    monkeypatch.setitem(sys.modules, "kokoro", SimpleNamespace(KPipeline=kpipeline))
    params = TTSParams(cache_dir=tmp_path)
    tts = TTS(KokoroTTS(params), params)
    np.save(tts.cache_path(FILLER), FakeTTS.clip(FILLER))  # fillers are pre-rendered
    rig = rig_with(FakeLLM("Feet first. "), tts=tts)
    assert await run(rig) == [FILLER]
    assert "debrief synthesis failed" in caplog.text


# Cutting a debrief


def cut_by_stop(rig: Rig) -> None:
    rig.voice.stop()


def cut_by_say(rig: Rig) -> None:
    assert rig.voice.say(rig.request())


CUTS = pytest.mark.parametrize("cut", [cut_by_stop, cut_by_say])


def cuttable(llm: FakeLLM | None = None, rendered=(FILLER, "Feet up.")) -> Rig:
    rig = Rig(
        bank_of(["Feet up."]),
        rendered=list(rendered),
        fillers=[FILLER],
        debrief_llm=llm.config if llm else None,
        speaker_type=TurnTakingSpeaker,
    )
    rig.speaker.held = True
    return rig


@CUTS
async def test_cut_cancels_the_stream(cut) -> None:
    llm = FakeLLM("One here. ", 1.0, "Two here. ", timeout_s=WAIT_S)
    rig = cuttable(llm)
    task = asyncio.create_task(rig.voice.debrief(TWO_ATTEMPTS, {}, None))
    await until(lambda: llm.prompts)
    await asyncio.sleep(0)
    cut(rig)
    await asyncio.wait_for(task, WAIT_S)
    assert llm.closed
    assert "Two here." not in rig.tts.synthesised


@CUTS
async def test_cut_does_not_fall_back_to_the_template(cut) -> None:
    llm = FakeLLM(1.0, timeout_s=WAIT_S)
    rig = cuttable(llm)
    queue = rig.bus.subscribe("debrief_text")
    task = asyncio.create_task(rig.voice.debrief(TWO_ATTEMPTS, {}, None))
    await until(lambda: llm.prompts)
    cut(rig)
    await asyncio.wait_for(task, WAIT_S)
    assert captions(rig, queue) == [FILLER]
    assert rig.tts.synthesised == []


@CUTS
async def test_cut_drops_a_sentence_mid_synthesis(cut) -> None:
    rig = cuttable(FakeLLM("One here. ", 1.0, timeout_s=WAIT_S))
    release = threading.Event()
    rig.tts.on_synth = lambda _text: release.wait(5)
    queue = rig.bus.subscribe("debrief_text")
    task = asyncio.create_task(rig.voice.debrief(TWO_ATTEMPTS, {}, None))
    await until(lambda: rig.tts.synthesised)
    cut(rig)
    await asyncio.wait_for(task, WAIT_S)
    release.set()
    rig.finish_worker()
    assert captions(rig, queue) == [FILLER]
    assert FakeTTS.clip("One here.").tolist() not in [p.tolist() for p in rig.speaker.played]


@CUTS
async def test_cut_cancels_synthesis_still_queued_on_the_worker(cut) -> None:
    rig = cuttable(FakeLLM("One here. ", 1.0, timeout_s=WAIT_S))
    release = threading.Event()
    rig.pool.submit(release.wait, 5)
    task = asyncio.create_task(rig.voice.debrief(TWO_ATTEMPTS, {}, None))
    await until(lambda: rig.speaker.played)
    await asyncio.sleep(0.01)  # the sentence is now waiting behind the blocked worker
    cut(rig)
    await asyncio.wait_for(task, WAIT_S)
    release.set()
    rig.finish_worker()
    assert rig.tts.synthesised == []


@CUTS
async def test_cut_drops_audio_waiting_for_its_turn(cut) -> None:
    rig = cuttable(FakeLLM("One here. Two here. "))
    queue = rig.bus.subscribe("debrief_text")
    task = asyncio.create_task(rig.voice.debrief(TWO_ATTEMPTS, {}, None))
    await until(lambda: len(rig.tts.synthesised) == 1)  # rendered, waiting on the filler
    await asyncio.sleep(0.01)
    cut(rig)
    rig.speaker.finish()
    await asyncio.wait_for(task, WAIT_S)
    await asyncio.sleep(voice_module.SPEAKER_POLL_S * 2)
    assert captions(rig, queue) == [FILLER]


async def test_stop_cuts_audio_at_once() -> None:
    rig = cuttable(FakeLLM(1.0, timeout_s=WAIT_S))
    task = asyncio.create_task(rig.voice.debrief(TWO_ATTEMPTS, {}, None))
    await until(lambda: rig.speaker.played)
    rig.voice.stop()
    assert rig.speaker.stops >= 1
    assert not rig.speaker.speaking
    await task


async def test_say_needing_synthesis_still_silences_the_debrief() -> None:
    rig = cuttable(FakeLLM(1.0, timeout_s=WAIT_S), rendered=[FILLER])
    task = asyncio.create_task(rig.voice.debrief(TWO_ATTEMPTS, {}, None))
    await until(lambda: rig.speaker.played)
    release = threading.Event()
    rig.tts.on_synth = lambda _text: release.wait(5)
    assert not rig.voice.say(rig.request())  # stuck rendering on the worker
    assert not rig.speaker.speaking
    await task
    release.set()


async def test_say_keeps_its_own_line_playing_after_the_cut() -> None:
    rig = cuttable(FakeLLM(1.0, timeout_s=WAIT_S))
    task = asyncio.create_task(rig.voice.debrief(TWO_ATTEMPTS, {}, None))
    await until(lambda: rig.speaker.played)
    assert rig.voice.say(rig.request())
    await task
    assert rig.speaker.speaking
    assert rig.texts() == ["Feet up."]


async def test_a_late_say_does_not_cut_the_debrief() -> None:
    llm = FakeLLM(0.02, "Hold 4 again. ", timeout_s=WAIT_S)
    rig = rig_with(llm)
    queue = rig.bus.subscribe("debrief_text")
    task = asyncio.create_task(rig.voice.debrief(TWO_ATTEMPTS, {}, None))
    await until(lambda: rig.speaker.played)
    assert not rig.voice.say(rig.request(deadline_in=-1.0))
    await task
    assert captions(rig, queue) == [FILLER, "Hold 4 again."]


async def test_stop_works_from_another_thread() -> None:
    llm = FakeLLM(1.0, timeout_s=WAIT_S)
    rig = cuttable(llm)
    task = asyncio.create_task(rig.voice.debrief(TWO_ATTEMPTS, {}, None))
    await until(lambda: llm.prompts)
    await asyncio.to_thread(rig.voice.stop)
    await asyncio.wait_for(task, WAIT_S)
    assert llm.closed


async def test_cancelling_the_caller_stops_the_debrief_and_propagates() -> None:
    llm = FakeLLM(1.0, timeout_s=WAIT_S)
    rig = cuttable(llm)
    task = asyncio.create_task(rig.voice.debrief(TWO_ATTEMPTS, {}, None))
    await until(lambda: llm.prompts)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert llm.closed
    assert not rig.speaker.speaking


async def test_a_new_debrief_replaces_the_running_one() -> None:
    first = FakeLLM(1.0, timeout_s=WAIT_S)
    rig = rig_with(first)
    running = asyncio.create_task(rig.voice.debrief(TWO_ATTEMPTS, {}, None))
    await until(lambda: first.prompts)
    second = FakeLLM("Hold 4 again. ")
    rig.voice._debrief_llm = second.config
    said = await run(rig)
    await asyncio.wait_for(running, WAIT_S)
    assert first.closed
    assert said[-1] == "Hold 4 again."


async def in_the_gap_after_first_sentence(rig: Rig) -> None:
    """Wait until the speaker has gone quiet with the next sentence still to come."""
    await until(lambda: len(rig.speaker.played) == 2 and not rig.speaker.speaking)


def gappy() -> Rig:
    rig = cuttable(FakeLLM("One here. ", GAP_S, "Two here. ", timeout_s=WAIT_S))
    rig.speaker.held = False
    return rig


async def test_is_speaking_stays_true_between_sentences() -> None:
    rig = gappy()
    task = asyncio.create_task(rig.voice.debrief(TWO_ATTEMPTS, {}, None))
    await in_the_gap_after_first_sentence(rig)
    assert rig.voice.is_speaking()
    await asyncio.wait_for(task, WAIT_S)
    assert not rig.voice.is_speaking()


async def test_is_speaking_goes_false_as_soon_as_a_debrief_is_cut() -> None:
    rig = gappy()
    task = asyncio.create_task(rig.voice.debrief(TWO_ATTEMPTS, {}, None))
    await in_the_gap_after_first_sentence(rig)
    rig.voice.stop()
    assert not rig.voice.is_speaking()
    await asyncio.wait_for(task, WAIT_S)


async def test_a_cue_in_a_gap_is_suppressed_and_the_debrief_finishes() -> None:
    rig = gappy()
    suppressed = []
    arbiter = Arbiter(
        rig.voice, ARBITER, clock=rig.clock, publish=lambda _topic, item: suppressed.append(item)
    )
    queue = rig.bus.subscribe("debrief_text")
    task = asyncio.create_task(rig.voice.debrief(TWO_ATTEMPTS, {}, None))
    await in_the_gap_after_first_sentence(rig)
    arbiter.offer(CoachEvent(rig.clock.t, "feet_low", 1, 1.2, {}), 0, "neutral")
    assert arbiter.tick() is None
    await asyncio.wait_for(task, WAIT_S)
    assert [item.reason for item in suppressed] == ["busy"]
    assert captions(rig, queue) == [FILLER, "One here.", "Two here."]


async def test_a_top_priority_cue_in_a_gap_still_cuts_the_debrief() -> None:
    rig = gappy()
    arbiter = Arbiter(rig.voice, ARBITER, clock=rig.clock)
    queue = rig.bus.subscribe("debrief_text")
    task = asyncio.create_task(rig.voice.debrief(TWO_ATTEMPTS, {}, None))
    await in_the_gap_after_first_sentence(rig)
    event = CoachEvent(rig.clock.t, "feet_low", 0, 1.2, {})
    arbiter.offer(event, 0, "neutral")
    assert arbiter.tick() is event
    await asyncio.wait_for(task, WAIT_S)
    assert captions(rig, queue) == [FILLER, "One here."]
    assert [line.text for line in rig.lines()] == ["Feet up."]


async def test_stop_after_a_debrief_is_harmless() -> None:
    rig = rig_with()
    await run(rig)
    rig.voice.stop()
    assert rig.voice._debrief is None


async def test_close_cuts_a_running_debrief() -> None:
    llm = FakeLLM(1.0, timeout_s=WAIT_S)
    rig = cuttable(llm)
    task = asyncio.create_task(rig.voice.debrief(TWO_ATTEMPTS, {}, None))
    await until(lambda: llm.prompts)
    rig.voice.close()
    await asyncio.wait_for(task, WAIT_S)
    assert llm.closed


# make_voice


@pytest.fixture
def voice_env(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Real config and persona, with no audio device and a recorded load_dotenv."""
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(OutputStream=FakeStream))
    monkeypatch.setattr(Voice, "warm_up", lambda self: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(voice_module, "load_dotenv", lambda: None)
    built = []
    monkeypatch.setattr(
        debrief_module,
        "make_gemini_stream",
        lambda *args: built.append(args) or FakeLLM().stream,
    )
    cfg = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))
    return {"cfg": cfg, "built": built}


def test_make_voice_loads_the_key_from_dotenv(voice_env, monkeypatch) -> None:
    monkeypatch.setattr(
        voice_module, "load_dotenv", lambda: monkeypatch.setenv("GEMINI_API_KEY", "from-dotenv")
    )
    voice = make_voice(voice_env["cfg"], Bus(), bank={})
    api_key, model, instruction = voice_env["built"][0]
    assert api_key == "from-dotenv"
    assert model == voice_env["cfg"]["voice"]["llm_model"]
    assert "Coach Harrow" in instruction
    llm = voice._debrief_llm
    assert llm.first_sentence_timeout_s == 4
    assert llm.max_words == 20
    assert "Great job" in llm.banned
    voice.close()


def test_make_voice_with_no_llm_never_builds_a_client(voice_env, monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "set")
    voice = make_voice(voice_env["cfg"], Bus(), no_llm=True, bank={})
    assert voice._debrief_llm is None
    assert voice_env["built"] == []
    voice.close()


def test_make_voice_without_a_key_warns_and_uses_the_template(voice_env, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        voice = make_voice(voice_env["cfg"], Bus(), bank={})
    assert voice._debrief_llm is None
    assert "GEMINI_API_KEY is not set" in caplog.text
    voice.close()


def test_docstring_no_longer_says_no_llm_does_nothing() -> None:
    assert "C5" not in make_voice.__doc__


# Gemini client


class FakeGenai:
    def __init__(self, chunks: list, hang: bool = False) -> None:
        self.chunks = chunks
        self.hang = hang
        self.calls: list[dict] = []
        self.closed = False
        self.aio = SimpleNamespace(models=SimpleNamespace(generate_content_stream=self._stream))

    async def _stream(self, **kwargs):
        self.calls.append(kwargs)

        async def gen():
            try:
                for text in self.chunks:
                    yield SimpleNamespace(text=text)
                if self.hang:
                    await asyncio.sleep(WAIT_S)
            finally:
                self.closed = True

        return gen()


def gemini(monkeypatch, fake: FakeGenai):
    from google import genai

    monkeypatch.setattr(genai, "Client", lambda api_key: fake)
    return make_gemini_stream("key", "a-model", "Be dry.")


async def test_gemini_stream_uses_the_async_client_and_persona(monkeypatch) -> None:
    fake = FakeGenai(["Hold ", None, "4."])
    stream = gemini(monkeypatch, fake)
    assert [chunk async for chunk in stream("the prompt")] == ["Hold ", "4."]
    call = fake.calls[0]
    assert (call["model"], call["contents"]) == ("a-model", "the prompt")
    assert call["config"].system_instruction == "Be dry."


async def test_cancelling_gemini_stream_closes_the_response(monkeypatch) -> None:
    fake = FakeGenai(["Hold 4. "], hang=True)
    llm = DebriefLLM(gemini(monkeypatch, fake), WAIT_S, 20, ())
    rig = rig_with()
    rig.voice._debrief_llm = llm
    rig.speaker.held = True
    task = asyncio.create_task(rig.voice.debrief(TWO_ATTEMPTS, {}, None))
    await until(lambda: fake.calls)
    await asyncio.sleep(0.01)
    rig.voice.stop()
    await asyncio.wait_for(task, WAIT_S)
    assert fake.closed
