"""Generate data/phrasebank.json with Gemini and pre-render all audio.

Run from the repo root: uv run python -m scripts.build_phrasebank [--only KIND] [--dry-run]

Without GEMINI_API_KEY or a real voice.llm_model, line generation is skipped with a warning; the
existing bank is kept and the fragments are still rendered. --model overrides voice.llm_model.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Callable
from pathlib import Path

import yaml
from dotenv import load_dotenv

from coach.contracts import EVENT_KINDS
from coach.voice.phrasebank import (
    MIN_LINES,
    blocked_phrases,
    persona_instruction,
    reject_reason,
    render_texts,
    select_lines,
)
from coach.voice.tts import make_tts

log = logging.getLogger(__name__)

CONFIG_PATH = Path("config/default.yaml")
BANK_PATH = Path("data/phrasebank.json")
MODEL_PLACEHOLDER = "CURRENT-FLASH-LITE-MODEL"
CANDIDATES = 10
KEEP = 6
LEVELS = (0, 1, 2)
MOODS = ("neutral", "impressed", "exasperated")
RETRY_ATTEMPTS = 6  # the free tier rate-limits a run of 126 requests
RETRY_INITIAL_DELAY_S = 5.0

# Placeholder wording from TEST_CLIMBS.md; B can correct it when the detectors land.
KIND_MEANINGS = {
    "attempt_start": "the climber is setting off on a new attempt",
    "hand_searching": "a hand is patting around near the next hold instead of grabbing it",
    "feet_low": "the feet are too low compared with the hips",
    "arms_bent": "the climber is hanging on bent arms and wasting strength",
    "stalling": "the climber has stayed in one position too long",
    "fall": "the climber has just fallen and is off the wall; give no move to make now",
    "top": "the climber has topped out and the climb is over: celebrate warmly, ask what's next, "
    "and give no move to make",
    "high_point": "the climber has just got higher than on any earlier attempt",
    "quiet_feet": "praise: feet are being placed slowly and precisely with no re-adjusting",
    "feet_cut": "the feet have swung off the wall",
    "weight_shift": "the climber is reaching sideways with the hips still over the other foot",
    "crux_warning": "the climber is approaching the move that beat them on earlier attempts",
    "route_preview": "a big reach near full arm span is coming up next",
    "told_you_so": "the climber ignored an earlier cue and fell on that same move; they are off "
    "the wall",
}
LEVEL_RULES = {
    0: "Generic: no limb, no hold, no reference to earlier attempts.",
    1: "If the event involves a limb, at least half the lines must use the {limb} slot.",
    2: "If the event involves a hold, at least half the lines must use the {hold} slot. Refer to "
    "earlier attempts in general terms.",
}
LEVEL_SLOTS = {1: "limb", 2: "hold"}

Ask = Callable[[str], list]


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def build_prompt(persona: dict, kind: str, level: int, mood: str) -> str:
    max_words = persona["max_words"][f"level{level}"]
    examples = [
        line
        for line in persona["examples"][mood]
        if reject_reason(line, max_words, blocked_phrases(persona)) is None
    ]
    return "\n".join(
        (
            f"Event: {kind}: {KIND_MEANINGS[kind]}.",
            f"Specificity: {LEVEL_RULES[level]}",
            f"Mood: {mood}: {persona['moods'][mood]}",
            "These examples only show the voice; do not reuse them: " + " | ".join(examples),
            f"Write {CANDIDATES} different lines to say aloud, at most {max_words} words each; "
            "fewer is better. Every line must be about this event and fit what the climber is "
            "doing right now. Tell them what to do, unless the event is pure praise.",
            "Slots, used only where relevant: {dir} is a direction such as 'up and left', "
            "{colour} is a hold colour, {hold} says 'hold six', {limb} says 'left hand'. "
            "A slot counts as its full spoken length.",
            "You don't know the side, hold number, colour or how many times something happened, "
            "so never state them: no left or right, no numbers, no colour names, no counts. Use "
            "a slot or leave it out.",
            "Return a JSON list of strings.",
        )
    )


def build_bank(ask: Ask, persona: dict, kinds: tuple[str, ...], previous: dict) -> dict:
    """Generate lines for `kinds`; a failed key keeps its previous lines."""
    bank = {kind: levels for kind, levels in previous.items() if kind not in kinds}
    for kind in kinds:
        bank[kind] = {}
        for level in LEVELS:
            max_words = persona["max_words"][f"level{level}"]
            bank[kind][str(level)] = {}
            for mood in MOODS:
                try:
                    candidates = ask(build_prompt(persona, kind, level, mood))
                    lines = select_lines(
                        candidates,
                        max_words,
                        blocked_phrases(persona),
                        KEEP,
                        LEVEL_SLOTS.get(level),
                    )
                    if len(lines) < MIN_LINES:
                        log.warning("%s L%d %s: only %d lines", kind, level, mood, len(lines))
                except Exception as exc:  # one bad key must not lose a whole run
                    log.warning("%s L%d %s failed: %s", kind, level, mood, exc)
                    lines = previous.get(kind, {}).get(str(level), {}).get(mood, [])
                bank[kind][str(level)][mood] = lines
    return bank


def make_ask(api_key: str, model: str, system_instruction: str) -> Ask:
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(
                attempts=RETRY_ATTEMPTS, initial_delay=RETRY_INITIAL_DELAY_S
            )
        ),
    )
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=list[str],
    )

    def ask(prompt: str) -> list:
        response = client.models.generate_content(model=model, contents=prompt, config=config)
        return json.loads(response.text)

    return ask


def count_table(bank: dict) -> tuple[str, int]:
    """A table of line counts per key, and how many keys fall short of MIN_LINES."""
    rows = [f"{'kind':16s} L {''.join(f'{mood:>13s}' for mood in MOODS)}"]
    short = 0
    for kind in EVENT_KINDS:
        for level in LEVELS:
            moods = bank.get(kind, {}).get(str(level), {})
            counts = [len(moods.get(mood, [])) for mood in MOODS]
            short += sum(count < MIN_LINES for count in counts)
            rows.append(f"{kind:16s} {level} {''.join(f'{count:13d}' for count in counts)}")
    return "\n".join(rows), short


def write_bank(bank: dict, path: Path = BANK_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(bank, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def missing_llm_setting(api_key: str | None, model: str | None) -> str | None:
    if not api_key:
        return "GEMINI_API_KEY is not set"
    if not model or model == MODEL_PLACEHOLDER:
        return "voice.llm_model is still a placeholder (or pass --model)"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", choices=EVENT_KINDS, metavar="KIND", help="regenerate one kind")
    ap.add_argument("--dry-run", action="store_true", help="show the lines, write nothing")
    ap.add_argument("--model", help="override voice.llm_model")
    args = ap.parse_args()
    load_dotenv()

    voice_cfg = load_yaml(CONFIG_PATH)["voice"]
    persona = load_yaml(voice_cfg["persona"])
    bank = json.loads(BANK_PATH.read_text(encoding="utf-8")) if BANK_PATH.exists() else {}

    api_key = os.environ.get("GEMINI_API_KEY")
    model = args.model or voice_cfg.get("llm_model")
    missing = missing_llm_setting(api_key, model)
    if missing:
        log.warning("skipping line generation: %s", missing)
    else:
        ask = make_ask(api_key, model, persona_instruction(persona))
        kinds = (args.only,) if args.only else EVENT_KINDS
        bank = build_bank(ask, persona, kinds, bank)

    texts = render_texts(bank, persona["colours"], persona["fillers"])
    if args.dry_run:
        print(json.dumps(bank, indent=2, ensure_ascii=False))
    else:
        if not missing:
            write_bank(bank)
        tts = make_tts(voice_cfg, persona)
        for text in texts:
            tts.synth(text)

    table, short = count_table(bank)
    print(table)
    print(f"{len(texts)} clips {'to render' if args.dry_run else 'rendered'}")
    print(f"{short} keys have fewer than {MIN_LINES} lines")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
