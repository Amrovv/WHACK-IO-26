import json
import sys

import numpy as np
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
from coach.voice.tts import TTS, TTSParams
from scripts import build_phrasebank
from scripts.build_phrasebank import (
    KEEP,
    KIND_MEANINGS,
    LEVELS,
    MODEL_PLACEHOLDER,
    MOODS,
    build_bank,
    build_prompt,
    check_clips,
    count_table,
    missing_clips,
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


# Checking the audio cache


class SilentEngine:
    name = "fake"

    def voice_id(self) -> str:
        return "fake"

    def speed(self) -> float:
        return 1.0

    def raw(self, text: str) -> np.ndarray:
        return np.full(2400, 0.1, dtype=np.float32)


CHECK_PERSONA = {"colours": ["pink"], "fillers": ["Right. Listen."]}
CHECK_BANK = {"fall": {"0": {"neutral": ["Off again.", "{hold} again."]}}}


@pytest.fixture
def cache_tts(tmp_path) -> TTS:
    return TTS(SilentEngine(), TTSParams(cache_dir=tmp_path))


def render_all(tts: TTS) -> list[str]:
    texts = render_texts(CHECK_BANK, CHECK_PERSONA["colours"], CHECK_PERSONA["fillers"])
    for text in texts:
        tts.synth(text)
    return texts


def test_check_passes_when_every_clip_is_cached(cache_tts, capsys) -> None:
    texts = render_all(cache_tts)
    assert check_clips(CHECK_BANK, CHECK_PERSONA, cache_tts) == 0
    assert f"{len(texts)} of {len(texts)} clips are cached" in capsys.readouterr().out


def test_check_names_a_missing_line_and_fails(cache_tts, capsys) -> None:
    render_all(cache_tts)
    cache_tts.cache_path("Off again.").unlink()
    assert check_clips(CHECK_BANK, CHECK_PERSONA, cache_tts) == 1
    assert "missing: 'Off again.'" in capsys.readouterr().out


@pytest.mark.parametrize("clip", ["again.", "hold six", "pink", "up and left", "Right. Listen."])
def test_check_covers_slot_pieces_fragments_and_fillers(cache_tts, clip: str) -> None:
    texts = render_all(cache_tts)
    cache_tts.cache_path(clip).unlink()
    assert missing_clips(texts, cache_tts) == [clip]


def test_a_corrupt_clip_counts_as_missing(cache_tts) -> None:
    texts = render_all(cache_tts)
    cache_tts.cache_path("Off again.").write_bytes(b"not audio")
    assert missing_clips(texts, cache_tts) == ["Off again."]


def test_clips_from_another_voice_do_not_count(cache_tts, tmp_path) -> None:
    render_all(cache_tts)
    other = TTS(SilentEngine(), TTSParams(cache_dir=tmp_path))
    other.engine.voice_id = lambda: "another"
    assert check_clips(CHECK_BANK, CHECK_PERSONA, other) == 1


def test_an_empty_bank_fails_rather_than_passing_on_nothing(cache_tts) -> None:
    assert check_clips({}, CHECK_PERSONA, cache_tts) == 1


def test_render_fills_gaps_without_touching_the_bank(cache_tts, tmp_path, monkeypatch) -> None:
    bank_path = tmp_path / "phrasebank.json"
    bank_path.write_text(json.dumps(CHECK_BANK), encoding="utf-8")
    before = bank_path.read_bytes()
    monkeypatch.setattr(build_phrasebank, "BANK_PATH", bank_path)
    monkeypatch.setattr(build_phrasebank, "make_tts", lambda *_args: cache_tts)
    monkeypatch.setattr(build_phrasebank, "load_dotenv", lambda: None)
    monkeypatch.setattr(build_phrasebank, "make_ask", lambda *_args: pytest.fail("called Gemini"))
    monkeypatch.setenv("GEMINI_API_KEY", "set")
    monkeypatch.setattr(sys, "argv", ["build_phrasebank", "--render"])
    build_phrasebank.main()
    assert bank_path.read_bytes() == before
    assert cache_tts.cached("Off again.") is not None


def test_check_never_synthesises(cache_tts) -> None:
    cache_tts.engine.raw = lambda _text: pytest.fail("the check must not render")
    assert check_clips(CHECK_BANK, CHECK_PERSONA, cache_tts) == 1


def test_prefer_direction_drops_hold_lines_when_a_direction_is_known() -> None:
    from coach.voice.phrasebank import prefer_direction

    lines = ["Grab the {hold}.", "Reach {dir}.", "Stop patting."]
    assert prefer_direction(lines, {"dir": "up", "hold": "hold six"}) == ["Reach {dir}."]
    assert prefer_direction(lines, {"hold": "hold six"}) == lines
    assert prefer_direction(["Grab the {hold}.", "Stop."], {"dir": "up"}) == ["Stop."]
    assert prefer_direction(["Grab the {hold}."], {"dir": "up"}) == ["Grab the {hold}."]


def test_hand_searching_level_two_asks_for_direction_not_hold() -> None:
    from scripts.build_phrasebank import level_slot

    persona = yaml.safe_load(open("config/personas/coach.yaml", encoding="utf-8"))
    prompt = build_prompt(persona, "hand_searching", 2, "neutral")
    assert "never use the {hold} slot" in prompt
    assert level_slot("hand_searching", 2) == "dir"
    assert level_slot("feet_low", 2) == "hold"


def test_direction_keys_reject_hard_coded_directions() -> None:
    from scripts.build_phrasebank import drop_fixed_directions

    lines = ["Straight up.", "Reach {dir}.", "Stop hunting. Up and left.", "Grab it."]
    assert drop_fixed_directions("hand_searching", 2, lines) == ["Reach {dir}.", "Grab it."]
    assert drop_fixed_directions("feet_low", 2, ["Feet up."]) == ["Feet up."]


def test_hold_places_name_holds_by_height_on_the_route() -> None:
    from coach.contracts import Hold, HoldMap
    from coach.voice.phrasebank import hold_places, slot_values

    def hold(i: int, y: float) -> Hold:
        return Hold(i, (0.5, y), (0.4, y - 0.05, 0.6, y + 0.05))

    holdmap = HoldMap(
        "p", "pink", [hold(1, 0.9), hold(2, 0.7), hold(3, 0.4), hold(4, 0.1)], (640, 480), "manual"
    )
    bands = [[0.25, "low hold"], [0.6, "middle hold"], [0.85, "high hold"], [1.0, "top hold"]]
    places = hold_places(holdmap, bands)
    assert places == {1: "low hold", 2: "middle hold", 3: "high hold", 4: "top hold"}
    assert slot_values({"hold_id": 3}, None, places) == {"hold": "high hold"}
    assert slot_values({"hold_id": 3}) == {"hold": "hold three"}  # no hold map yet
