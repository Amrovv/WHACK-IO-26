"""Shared contracts v1.1: the only types that cross layer boundaries.

Owned by B. Change only by PR + a message in the team chat.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

Point = tuple[float, float]  # normalised image coords 0-1, origin top-left, y down


class Limb(str, Enum):
    LH = "left_hand"
    RH = "right_hand"
    LF = "left_foot"
    RF = "right_foot"


@dataclass
class Hold:
    id: int  # 1 = lowest ... N = highest (spoken as "hold 6")
    center: Point
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1
    role: str = "mid"  # "start" | "mid" | "top"


@dataclass
class HoldMap:  # A -> B, D; once per problem; data/holdmaps/<problem_id>.json
    problem_id: str
    colour: str  # spoken name, e.g. "pink"
    holds: list[Hold]
    image_size: tuple[int, int]
    mode: str  # "detector" | "colour" | "manual"


@dataclass
class ClimberState:  # A -> B every processed frame (D gets ~10 Hz)
    t: float  # capture time, time.monotonic()
    frame_idx: int
    person_found: bool
    keypoints: dict[str, tuple[float, float, float]]  # name -> (x, y, visibility); L/R fixed
    limbs: dict[Limb, Point]  # smoothed hand and foot points
    contacts: dict[Limb, int | None]  # limb -> hold id
    hip: Point


@dataclass
class CoachEvent:  # B; published for D
    t: float  # capture time of the triggering frame
    kind: str  # see EVENT_KINDS
    priority: int  # 0 big moment, 1 cue, 2 banter
    ttl_s: float
    data: dict = field(default_factory=dict)  # {"limb": "LH", "hold_id": 7, "dir": "up and left"}


@dataclass
class SpeakRequest:  # B -> C via Voice.say()
    event: CoachEvent
    level: int  # specificity 0-2, from habit counters
    mood: str  # "neutral" | "impressed" | "exasperated"
    deadline: float  # monotonic; audio must start by then or say() returns False


@dataclass
class AttemptSummary:  # B -> C (debriefs), D
    attempt_no: int
    duration_s: float
    high_hold_id: int | None
    outcome: str  # "top" | "fall" | "downclimb"
    stalls: list[tuple[int, float]]  # (hold id, seconds)
    cues_given: list[str]
    cues_followed: list[str]


@dataclass
class SpokenLine:  # C -> B (compliance), D
    t_event: float
    t_audio_start: float
    kind: str
    text: str


@dataclass
class Suppressed:  # B -> D
    event: CoachEvent
    reason: str  # "stale" | "busy" | "cooldown" | "budget" | "late" | "lower_priority"


class Voice(Protocol):  # C implements, B calls
    def say(self, req: SpeakRequest) -> bool: ...  # True if audio started before the deadline

    def is_speaking(self) -> bool: ...

    def stop(self) -> None: ...  # cut audio within ~50 ms

    async def debrief(
        self, history: list[AttemptSummary], habits: dict, crux_hold: int | None
    ) -> None: ...


# Frozen at H5.
EVENT_KINDS = (
    "attempt_start",
    "hand_searching",
    "feet_low",
    "arms_bent",
    "stalling",
    "fall",
    "top",
    "high_point",
    "quiet_feet",
    "feet_cut",
    "weight_shift",
    "crux_warning",
    "route_preview",
    "told_you_so",
)

# Bus topics: publisher -> payload.
TOPICS = (
    "holdmap",  # A -> HoldMap
    "state",  # A -> ClimberState
    "frame",  # A -> latest frame as JPEG, ~10 fps
    "event",  # B -> CoachEvent
    "suppressed",  # B -> Suppressed
    "attempt",  # B -> AttemptSummary
    "spoken",  # C -> SpokenLine
    "debrief_text",  # C -> streamed debrief sentences
)
