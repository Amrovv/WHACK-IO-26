"""Generate data/phrasebank.json with Gemini and pre-render all audio.

Run from the repo root: uv run python -m scripts.build_phrasebank [--only KIND ...] [--dry-run]

--check generates and renders nothing: it exits 1 if any clip Voice could need isn't on disk. Run
it before a demo, because a line with a missing clip has to render live and can be lost.
--render fills in missing clips from the bank as it stands, so the lines don't change.

Without GEMINI_API_KEY or a real voice.llm_model, line generation is skipped with a warning; the
existing bank is kept and the fragments are still rendered. --model overrides voice.llm_model.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections.abc import Callable, Iterable
from pathlib import Path

import yaml
from dotenv import load_dotenv

from coach.contracts import EVENT_KINDS
from coach.voice.phrasebank import (
    MIN_LINES,
    SLOT_RE,
    blocked_phrases,
    persona_instruction,
    place_names,
    reject_reason,
    render_texts,
    select_lines,
)
from coach.voice.tts import TTS, make_tts

log = logging.getLogger(__name__)

CONFIG_PATH = Path("config/default.yaml")
BANK_PATH = Path("data/phrasebank.json")
MODEL_PLACEHOLDER = "CURRENT-FLASH-LITE-MODEL"
CANDIDATES = 10
KEEP = 3  # lines kept per (kind, level, mood); Voice cycles through them without repeats
LEVELS = (0, 1, 2)
MOODS = ("neutral", "impressed", "exasperated")
RETRY_ATTEMPTS = 6  # the free tier rate-limits a run of 126 requests
RETRY_INITIAL_DELAY_S = 5.0

# What each event means at the moment B's brain fires it (coach/brain), so every line is true
# when spoken. Where TEST_CLIMBS.md describes an event differently, the brain's behaviour wins.
KIND_MEANINGS = {
    "attempt_start": "the climber is setting off on a new attempt",
    "hand_searching": "a hand is patting around near the next hold instead of grabbing it",
    "feet_low": "the feet are too low compared with the hips",
    "arms_bent": "the climber is hanging on bent arms and wasting strength",
    "stalling": "the climber has stayed in one position too long",
    "fall": "the climber has just fallen and is off the wall; give no move to make now",
    "top": "the climber has topped out and the climb is over: celebrate warmly, ask what's next, "
    "and give no move to make",
    # Also fires on the first attempt, so it can't claim to beat an earlier one.
    "high_point": "the climber has just reached their highest hold of the session so far",
    # QuietFeet fires on a fidget: a foot leaves a hold and goes straight back onto it.
    "quiet_feet": "a foot came off its hold and went straight back onto the same one: fidgety, "
    "noisy footwork; tell them to place each foot once, precisely, and leave it",
    "feet_cut": "the feet have swung off the wall",
    "weight_shift": "the climber is reaching sideways with the hips still over the other foot",
    "crux_warning": "the climber is approaching the move that beat them on earlier attempts",
    # RoutePreview fires on the ground, before the attempt starts.
    "route_preview": "the climber is on the ground before an attempt; the route has a reach near "
    "full arm span, to the hold the {hold} slot names; warn them before they set off",
    # Memory fires it once a spoken cue is followed and the climber then gets higher.
    "told_you_so": "mid-climb, the climber just did what an earlier cue told them and it worked: "
    "they have reached a higher hold, the one the {hold} slot names; a brief, dry 'told you', "
    "then back to climbing",
}
LEVEL_RULES = {
    0: "Generic: no limb, no hold, no reference to earlier attempts.",
    1: "If the event involves a limb, at least half the lines must use the {limb} slot.",
    2: "If the event involves a hold, at least half the lines must use the {hold} slot. Refer to "
    "earlier attempts in general terms.",
}
LEVEL_SLOTS = {1: "limb", 2: "hold"}
# Team decision: hand_searching names a relative direction, never a hold number.
DIRECTION_KINDS = {"hand_searching"}
DIRECTION_RULE = (
    "Point the climber to the next hold with the {dir} slot; at least half the lines must use "
    "it, written literally as {dir}. Never write a direction word such as up, down or straight "
    "yourself, and never use the {hold} slot. Refer to earlier attempts in general terms."
)
# A direction outside the slot could be false, e.g. "Straight up" when the next hold is left.
_FIXED_DIRECTION_RE = re.compile(r"\b(up|down|straight|above|below|higher|lower)\b", re.I)


def level_rule(kind: str, level: int) -> str:
    return DIRECTION_RULE if level == 2 and kind in DIRECTION_KINDS else LEVEL_RULES[level]


def level_slot(kind: str, level: int) -> str | None:
    return "dir" if level == 2 and kind in DIRECTION_KINDS else LEVEL_SLOTS.get(level)


def drop_fixed_directions(kind: str, level: int, candidates: list) -> list:
    """For direction keys, drop lines that state a direction instead of using {dir}."""
    if level_slot(kind, level) != "dir":
        return candidates
    kept = []
    for line in candidates:
        if isinstance(line, str) and _FIXED_DIRECTION_RE.search(SLOT_RE.sub(" ", line)):
            log.info("rejected (hard-coded direction): %r", line)
        else:
            kept.append(line)
    return kept


Ask = Callable[[str], list]


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def build_prompt(persona: dict, kind: str, level: int, mood: str) -> str:
    max_words = persona["max_words"][f"level{level}"]
    examples = [
        line
        for line in persona["examples"][mood]
        if reject_reason(line, max_words, blocked_phrases(persona, kind)) is None
    ]
    off_topic = (persona.get("off_topic") or {}).get(kind, [])
    return "\n".join(
        (
            f"Event: {kind}: {KIND_MEANINGS[kind]}.",
            f"Specificity: {level_rule(kind, level)}",
            f"Mood: {mood}: {persona['moods'][mood]}",
            "These examples only show the voice; do not reuse them: " + " | ".join(examples),
            f"Write {CANDIDATES} different lines to say aloud, at most {max_words} words each; "
            "fewer is better. Every line must be about this event and fit what the climber is "
            "doing right now. Tell them what to do, unless the event is pure praise.",
            "Slots, used only where relevant: {dir} is a direction such as 'up and left', "
            "{colour} is a hold colour, {hold} says 'high hold', {limb} says 'left hand'. "
            "A slot counts as its full spoken length.",
            "You don't know the side, hold number, colour or how many times something happened, "
            "so never state them: no left or right, no numbers, no colour names, no counts. Use "
            "a slot or leave it out.",
            *([f"Never say, for this event: {' | '.join(off_topic)}"] if off_topic else []),
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
                    candidates = drop_fixed_directions(kind, level, candidates)
                    lines = select_lines(
                        candidates,
                        max_words,
                        blocked_phrases(persona, kind),
                        KEEP,
                        level_slot(kind, level),
                    )
                    if len(lines) < MIN_LINES:
                        log.warning("%s L%d %s: only %d lines", kind, level, mood, len(lines))
                except Exception as exc:  # one bad key must not lose a whole run
                    log.warning("%s L%d %s failed: %s", kind, level, mood, exc)
                    lines = previous.get(kind, {}).get(str(level), {}).get(mood, [])
                bank[kind][str(level)][mood] = lines
    return bank


def prune_bank(bank: dict, persona: dict) -> tuple[dict, int]:
    """The bank without lines today's rules reject (e.g. a new off_topic phrase), and how many
    were dropped. Lets a rule change clean an existing bank without regenerating it."""
    pruned: dict = {}
    dropped = 0
    for kind, levels in bank.items():
        banned = blocked_phrases(persona, kind)
        pruned[kind] = {}
        for level, moods in levels.items():
            max_words = persona["max_words"][f"level{level}"]
            pruned[kind][level] = {}
            for mood, lines in moods.items():
                kept = [line for line in lines if reject_reason(line, max_words, banned) is None]
                for line in set(lines) - set(kept):
                    log.info("pruned %s L%s %s: %r", kind, level, mood, line)
                dropped += len(lines) - len(kept)
                pruned[kind][level][mood] = kept
    return pruned, dropped


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


def format_bank(bank: dict) -> str:
    """The bank as JSON with each (kind, level, mood) group's lines on one row, so it reads as a
    table instead of one line per phrase."""

    def block(items: list[tuple[str, str]], indent: str) -> str:
        rows = [f"{indent}  {json.dumps(key)}: {value}" for key, value in items]
        return "{\n" + ",\n".join(rows) + f"\n{indent}}}"

    def moods(groups: dict) -> str:
        return block(
            [(m, json.dumps(lines, ensure_ascii=False)) for m, lines in groups.items()], "    "
        )

    def levels(kind_levels: dict) -> str:
        return block([(level, moods(groups)) for level, groups in kind_levels.items()], "  ")

    return block([(kind, levels(kind_levels)) for kind, kind_levels in bank.items()], "") + "\n"


def write_bank(bank: dict, path: Path = BANK_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(format_bank(bank), encoding="utf-8")
    tmp.replace(path)


def missing_llm_setting(api_key: str | None, model: str | None) -> str | None:
    if not api_key:
        return "GEMINI_API_KEY is not set"
    if not model or model == MODEL_PLACEHOLDER:
        return "voice.llm_model is still a placeholder (or pass --model)"
    return None


def missing_clips(texts: Iterable[str], tts: TTS) -> list[str]:
    """Clips that can't be loaded from the cache, whether absent or corrupt."""
    return [text for text in texts if tts.cached(text) is None]


def check_clips(bank: dict, persona: dict, tts: TTS) -> int:
    """Report on the audio cache. Returns the process exit code."""
    if not bank:
        log.error("%s is missing or empty", BANK_PATH)
        return 1
    texts = render_texts(
        bank, persona["colours"], persona["fillers"], place_names(persona.get("hold_places", ()))
    )
    missing = missing_clips(texts, tts)
    for text in missing:
        print(f"missing: {text!r}")
    print(f"{len(texts) - len(missing)} of {len(texts)} clips are cached")
    return 1 if missing else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--only", nargs="+", choices=EVENT_KINDS, metavar="KIND", help="regenerate these kinds"
    )
    ap.add_argument("--dry-run", action="store_true", help="show the lines, write nothing")
    ap.add_argument("--model", help="override voice.llm_model")
    ap.add_argument("--check", action="store_true", help="only check every clip is on disk")
    ap.add_argument("--render", action="store_true", help="only render clips for the current bank")
    args = ap.parse_args()
    load_dotenv()

    voice_cfg = load_yaml(CONFIG_PATH)["voice"]
    persona = load_yaml(voice_cfg["persona"])
    on_disk = json.loads(BANK_PATH.read_text(encoding="utf-8")) if BANK_PATH.exists() else {}
    bank, pruned = prune_bank(on_disk, persona)

    if args.check:
        # Voice loads the file as it is, pruned or not, so check every line in it.
        if pruned:
            log.warning(
                "%d lines in %s break today's rules; --render drops them", pruned, BANK_PATH
            )
        raise SystemExit(check_clips(on_disk, persona, make_tts(voice_cfg, persona)))

    api_key = os.environ.get("GEMINI_API_KEY")
    model = args.model or voice_cfg.get("llm_model")
    missing = "--render was set" if args.render else missing_llm_setting(api_key, model)
    if missing:
        log.warning("skipping line generation: %s", missing)
    else:
        ask = make_ask(api_key, model, persona_instruction(persona))
        kinds = tuple(args.only) if args.only else EVENT_KINDS
        bank = build_bank(ask, persona, kinds, bank)

    texts = render_texts(
        bank, persona["colours"], persona["fillers"], place_names(persona.get("hold_places", ()))
    )
    if args.dry_run:
        print(json.dumps(bank, indent=2, ensure_ascii=False))
    else:
        if not missing or pruned:
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
