import pytest
import yaml

from coach.contracts import EVENT_KINDS
from coach.voice.phrasebank import (
    blocked_phrases,
    fragments,
    hold_phrase,
    line_pieces,
    reject_reason,
    render_texts,
    select_lines,
    word_count,
)
from coach.voice.tts import TTSParams
from scripts.build_phrasebank import (
    KEEP,
    KIND_MEANINGS,
    LEVELS,
    MODEL_PLACEHOLDER,
    MOODS,
    build_bank,
    build_prompt,
    count_table,
    missing_llm_setting,
)


@pytest.fixture
def persona() -> dict:
    with open("config/personas/coach.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_every_event_kind_has_a_meaning() -> None:
    assert set(KIND_MEANINGS) == set(EVENT_KINDS)


def test_persona_sets_tts_without_warning(persona: dict, caplog) -> None:
    params = TTSParams.from_config({}, persona)
    assert (params.kokoro_voice, params.kokoro_lang, params.speed) == ("bm_lewis", "b", 0.95)
    assert caplog.text == ""


@pytest.mark.parametrize(
    ("hold_id", "spoken"),
    [(1, "hold one"), (19, "hold nineteen"), (20, "hold twenty"), (25, "hold twenty-five")],
)
def test_hold_phrase(hold_id: int, spoken: str) -> None:
    assert hold_phrase(hold_id) == spoken


def test_word_count_uses_spoken_slot_length() -> None:
    assert word_count("Feet up.") == 2
    assert word_count("{dir}, {colour} one.") == 5


@pytest.mark.parametrize(
    ("line", "reason"),
    [
        ("", "empty"),
        ("Move {foot}.", "bad slot"),
        ("Move {limb", "bad slot"),
        ("One two three four five", "too long"),
        ("Feet up 🔥", "emoji"),
        ("Left foot up.", "hard-coded side or number"),
        ("Hold 6.", "hard-coded side or number"),
        ("Grab hold six.", "hard-coded side or number"),
        ("Hold one first.", "hard-coded side or number"),
        ("Third time.", "hard-coded side or number"),
        ("GREAT JOB, champ!", "banned phrase"),
        ("You’ve got this", "banned phrase"),
    ],
)
def test_reject_reason(line: str, reason: str) -> None:
    assert reject_reason(line, 4, ["Great job", "You've got this"]) == reason


def test_reject_reason_accepts_slots_and_whole_word_matches() -> None:
    assert reject_reason("{limb} up.", 4, []) is None
    assert reject_reason("Bright idea.", 4, ["right"]) is None
    assert reject_reason("{colour} one.", 4, []) is None


def test_colour_names_are_blocked(persona: dict) -> None:
    assert reject_reason("Grab the yellow.", 4, blocked_phrases(persona)) == "banned phrase"


def test_select_lines_filters_dedupes_and_prefers_short() -> None:
    lines = ["Move your feet now.", "Feet up.", "feet up!", "Left foot.", 7]
    assert select_lines(lines, 4, [], 2) == ["Feet up.", "Move your feet now."]


def test_select_lines_prefers_the_slot_over_shorter_lines() -> None:
    assert select_lines(["Feet up.", "Move your {limb}."], 4, [], 1, "limb") == [
        "Move your {limb}."
    ]


def test_line_pieces_are_the_fixed_text() -> None:
    assert line_pieces("{dir}, {colour} one.") == ["one."]
    assert line_pieces("Feet up.") == ["Feet up."]
    assert line_pieces("{limb}, {dir}.") == []


def test_fragments_cover_every_slot_value() -> None:
    clips = fragments(["pink"], ["Right. Come here."])
    for clip in ("up and left", "pink", "left foot", "hold twenty-five", "Right. Come here."):
        assert clip in clips


def test_render_texts_has_whole_lines_and_slot_pieces() -> None:
    bank = {"feet_low": {"1": {"neutral": ["Feet up.", "{limb} higher."]}}}
    texts = render_texts(bank, [], [])
    assert {"Feet up.", "higher."} <= set(texts)
    assert "{limb} higher." not in texts


def test_prompt_has_the_rules_and_drops_unusable_examples(persona: dict) -> None:
    prompt = build_prompt(persona, "feet_low", 2, "exasperated")
    assert KIND_MEANINGS["feet_low"] in prompt
    assert "at most 8 words" in prompt
    assert "{hold}" in prompt
    assert "Left foot, not right" not in prompt


def test_build_bank_covers_every_key(persona: dict) -> None:
    lines = [f"Move {word}." for word in "abcdefghij"]
    bank = build_bank(lambda _prompt: lines, persona, EVENT_KINDS, {})
    assert all(
        len(bank[kind][str(level)][mood]) == KEEP
        for kind in EVENT_KINDS
        for level in LEVELS
        for mood in MOODS
    )
    assert count_table(bank)[1] == 0


def test_build_bank_only_touches_requested_kind_and_keeps_lines_on_failure(persona: dict) -> None:
    previous = {
        "fall": {"0": {"neutral": ["Again."]}},
        "top": {"0": {"neutral": ["Now that's climbing."]}},
    }

    def fail(_prompt: str) -> list:
        raise RuntimeError("rate limited")

    bank = build_bank(fail, persona, ("fall",), previous)
    assert bank["top"] == previous["top"]
    assert bank["fall"]["0"]["neutral"] == ["Again."]
    assert bank["fall"]["1"]["neutral"] == []


def test_missing_llm_setting() -> None:
    assert "GEMINI_API_KEY" in missing_llm_setting(None, "model")
    assert "llm_model" in missing_llm_setting("key", MODEL_PLACEHOLDER)
    assert missing_llm_setting("key", "model") is None
