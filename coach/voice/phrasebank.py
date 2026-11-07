"""Phrase bank rules shared by the builder and Voice: slots, fragments and line validation."""

from __future__ import annotations

import logging
import random
import re
import unicodedata
from collections.abc import Callable, Hashable, Iterable, Sequence

from coach.contracts import HoldMap, Limb

log = logging.getLogger(__name__)

MIN_LINES = 5  # every (kind, level, mood) needs this many approved lines
MAX_HOLD = 25
NEUTRAL = "neutral"
UNKNOWN_COLOUR = "unknown"  # HoldMap.colour when no hold was clicked; must never be spoken
SLOT_RE = re.compile(r"\{(\w*)\}")
SLOT_WORDS = {"dir": 3, "colour": 1, "hold": 2, "limb": 2}  # longest spoken form of each slot
DIRECTIONS = (
    "up",
    "down",
    "left",
    "right",
    "up and left",
    "up and right",
    "down and left",
    "down and right",
)
LIMBS = ("left hand", "right hand", "left foot", "right foot")
_ONES = (
    "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen"
).split()
_TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty"}
# Sides and counts must come from the event, so a fixed one in a line could be false when spoken.
_FIXED_FACT_RE = re.compile(
    r"\b(left|right|hold one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|"
    r"thirty|twice|second|third|fourth|fifth)\b|\d",
    re.IGNORECASE,
)


def hold_phrase(hold_id: int) -> str:
    """Spoken form of a hold id, e.g. 25 -> "hold twenty-five"."""
    if not 1 <= hold_id < 60:
        raise ValueError(f"no spoken form for hold {hold_id}")
    if hold_id < 20:
        return f"hold {_ONES[hold_id - 1]}"
    tens, ones = divmod(hold_id, 10)
    return f"hold {_TENS[tens]}" + (f"-{_ONES[ones - 1]}" if ones else "")


def hold_places(holdmap: HoldMap, bands: Iterable[Sequence]) -> dict[int, str]:
    """Each hold's spoken name from its height on the route, e.g. {7: "high hold"}.

    `bands` are (max fraction of the route's height, name) pairs from the persona, lowest first;
    the lowest hold is at 0 and the highest at 1. Team decision: holds are named by where they
    are, never by number.
    """
    bands = sorted((float(top), str(name)) for top, name in bands)
    if not holdmap.holds or not bands:
        return {}
    ys = [hold.center[1] for hold in holdmap.holds]
    bottom, top = max(ys), min(ys)  # y grows downwards
    span = bottom - top
    places = {}
    for hold in holdmap.holds:
        height = (bottom - hold.center[1]) / span if span > 0 else 0.0
        places[hold.id] = next((name for limit, name in bands if height <= limit), bands[-1][1])
    return places


def place_names(bands: Iterable[Sequence]) -> list[str]:
    return [str(name) for _, name in bands]


def word_count(line: str) -> int:
    """Spoken words, counting each slot at its longest form."""
    slots = SLOT_RE.findall(line)
    words = [w for w in SLOT_RE.sub(" ", line).split() if any(c.isalnum() for c in w)]
    return len(words) + sum(SLOT_WORDS.get(s, 1) for s in slots)


def _normalise(text: str) -> str:
    """Lower-case words only, so punctuation and curly quotes can't hide a match."""
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold().replace("’", "'").replace("'", "")))


def has_banned_phrase(line: str, banned: Iterable[str]) -> bool:
    normalised = f" {_normalise(line)} "
    return any(f" {_normalise(phrase)} " in normalised for phrase in banned)


def reject_reason(line: object, max_words: int, banned: Iterable[str]) -> str | None:
    """Why a generated line can't be used, or None if it passes."""
    if not isinstance(line, str) or not line.strip():
        return "empty"
    unknown = set(SLOT_RE.findall(line)) - set(SLOT_WORDS)
    if unknown or "{" in SLOT_RE.sub("", line) or "}" in SLOT_RE.sub("", line):
        return "bad slot"
    if word_count(line) > max_words:
        return "too long"
    if any(unicodedata.category(c) == "So" for c in line):
        return "emoji"
    if _FIXED_FACT_RE.search(SLOT_RE.sub(" ", line)):
        return "hard-coded side or number"
    if has_banned_phrase(line, banned):
        return "banned phrase"
    return None


def select_lines(
    candidates: Iterable[object],
    max_words: int,
    banned: Iterable[str],
    keep: int,
    prefer_slot: str | None = None,
) -> list[str]:
    """Valid, de-duplicated lines: those using `prefer_slot` first, then shortest first."""
    banned = list(banned)
    seen: set[str] = set()
    lines: list[str] = []
    for line in candidates:
        reason = reject_reason(line, max_words, banned)
        if reason:
            log.info("rejected (%s): %r", reason, line)
            continue
        key = _normalise(SLOT_RE.sub(r"\1", line))
        if key not in seen:
            seen.add(key)
            lines.append(line.strip())
    slot = f"{{{prefer_slot}}}"
    return sorted(lines, key=lambda line: (slot not in line, word_count(line)))[:keep]


def line_pieces(line: str) -> list[str]:
    """The fixed text around a line's slots: "{dir}, {colour} one." -> ["one."]."""
    pieces = (piece.strip(" ,") for piece in SLOT_RE.sub("\0", line).split("\0"))
    return [piece for piece in pieces if any(c.isalnum() for c in piece)]


def limb_phrase(limb: str) -> str:
    """Spoken form of a limb given as a Limb, its name ("LH") or its value ("left_hand")."""
    member = Limb.__members__.get(limb) or Limb(limb)
    return member.value.replace("_", " ")


def slot_values(
    data: dict, route_colour: str | None = None, places: dict[int, str] | None = None
) -> dict[str, str]:
    """Spoken slot values from an event's data; missing or unspeakable values are left out.

    A problem's holds all share one colour, so `route_colour` fills {colour} unless the event
    names its own. With `places` (from hold_places) {hold} is spoken by position ("high hold");
    without a hold map it falls back to the number.
    """
    places = places or {}

    def spoken_hold(hold_id: int) -> str:
        return places.get(hold_id) or hold_phrase(hold_id)

    data = {"colour": route_colour, "limb": data.get("foot"), **data}  # weight_shift sends "foot"
    sources = {
        "dir": ("dir", str),
        "colour": ("colour", str),
        "hold": ("hold_id", spoken_hold),
        "limb": ("limb", limb_phrase),
    }
    values: dict[str, str] = {}
    for slot, (key, spoken) in sources.items():
        if not data.get(key) or data[key] == UNKNOWN_COLOUR:
            continue
        try:
            values[slot] = spoken(data[key])
        except (TypeError, ValueError):
            log.warning("no spoken form for %s=%r", key, data[key])
    return values


def stitch(line: str, values: dict[str, str]) -> tuple[list[str], str]:
    """A line's clips in spoken order, and its full text. Every slot must be in `values`."""
    clips: list[str] = []
    for i, part in enumerate(SLOT_RE.split(line)):  # odd entries are the slot names
        clips.extend([values[part]] if i % 2 else line_pieces(part))
    return clips, SLOT_RE.sub(lambda match: values[match[1]], line)


def fallback_keys(kind: str, level: int, mood: str) -> list[tuple[str, int, str]]:
    """Keys to try in order: this mood down through the levels, then neutral the same way."""
    moods = [mood] if mood == NEUTRAL else [mood, NEUTRAL]
    return [(kind, lvl, m) for m in moods for lvl in range(level, -1, -1)]


def prefer_direction(lines: list[str], values: dict[str, str]) -> list[str]:
    """Lines to choose from: with a direction known, never one that speaks a hold number.

    Team decision: cues name a relative direction ("up and left"), not a hold ("hold seven").
    Lines that say {dir} come first; then any other line without {hold}; then all `lines`.
    """
    if "dir" not in values:
        return lines
    no_hold = [line for line in lines if "{hold}" not in line]
    by_direction = [line for line in no_hold if "{dir}" in line]
    return by_direction or no_hold or lines


class LinePicker:
    """Draws persona lines from the bank, shuffled without replacement per (kind, level, mood)."""

    def __init__(self, bank: dict, rng: random.Random | None = None) -> None:
        self.bank = bank
        self.rng = rng or random.Random()
        self.decks: dict[Hashable, list[str]] = {}
        self._last: dict[Hashable, str] = {}

    def pick(self, kind: str, level: int, mood: str, values: dict[str, str]) -> str | None:
        """The next line from the first key with one whose slots the event can fill."""
        for key in fallback_keys(kind, level, mood):
            _, lvl, m = key
            lines = self.bank.get(kind, {}).get(str(lvl), {}).get(m, [])
            fillable = [line for line in lines if set(SLOT_RE.findall(line)) <= values.keys()]
            if fillable:
                usable = set(prefer_direction(fillable, values))
                return self.draw(key, lines, usable.__contains__)
        return None

    def draw(self, key: Hashable, lines: list[str], usable: Callable[[str], bool]) -> str:
        """Shuffle without replacement; a fresh deck never opens with the line just used."""
        deck = self.decks.setdefault(key, [])
        options = [line for line in deck if usable(line)]
        if not options:
            deck[:] = self.rng.sample(lines, len(lines))
            options = [line for line in deck if usable(line)]
        last = self._last.get(key)
        line = next((option for option in options if option != last), options[0])
        deck.remove(line)
        self._last[key] = line
        return line


def fragments(
    colours: Iterable[str], fillers: Iterable[str], places: Iterable[str] = ()
) -> list[str]:
    """Every fixed clip that slots and debriefs stitch together."""
    holds = (hold_phrase(i) for i in range(1, MAX_HOLD + 1))
    return [*DIRECTIONS, *colours, *LIMBS, *holds, *places, *fillers]


def render_texts(
    bank: dict, colours: Iterable[str], fillers: Iterable[str], places: Iterable[str] = ()
) -> list[str]:
    """All text to pre-render: whole lines, the fixed pieces of slot lines, and fragments."""
    texts = set(fragments(colours, fillers, places))
    for levels in bank.values():
        for moods in levels.values():
            for lines in moods.values():
                for line in lines:
                    texts.update(line_pieces(line))
    return sorted(texts)


def blocked_phrases(persona: dict, kind: str | None = None) -> list[str]:
    """Banned phrases plus colour names, which must come from the {colour} slot.

    With `kind`, also the persona's off_topic phrases for that event.
    """
    off_topic = (persona.get("off_topic") or {}).get(kind, []) if kind else []
    return [*persona["banned_phrases"], *persona["colours"], *off_topic]


def persona_instruction(persona: dict) -> str:
    """System instruction shared by the phrase bank and the debriefs, so the voice can't drift."""
    return "\n".join(
        (
            f"You are {persona['name']}, a bouldering coach speaking aloud to a climber.",
            persona["character"].strip(),
            f"Delivery: {persona['voice'].get('tone', '')}",
            "Catchphrases, only where they truly fit: " + " | ".join(persona["catchphrases"]),
            "Never say: " + " | ".join(persona["banned_phrases"]),
        )
    )
