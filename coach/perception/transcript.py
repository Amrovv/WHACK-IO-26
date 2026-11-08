"""Local, incremental transcript of the perception and coach-brain decisions."""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from coach.brain.arbiter import Arbiter
from coach.brain.attempts import AttemptTracker
from coach.brain.detectors import DetectorContext, good_tier_detectors, mvp_detectors
from coach.brain.features import FeatureTracker
from coach.brain.memory import Memory
from coach.contracts import ClimberState, HoldMap, Limb, SpeakRequest, SpokenLine
from coach.perception.config import load_config
from coach.perception.holds import load_holdmap_with_shapes
from coach.perception.serialize import state_from_line
from coach.voice.debrief import debrief_payload, template_sentences
from coach.voice.phrasebank import LinePicker, hold_places, slot_values, stitch

log = logging.getLogger(__name__)


@dataclass
class TranscriptEntry:
    t: float
    source: str
    kind: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    spoken: bool = False


class _TextVoice:
    """Phrase-bank compatible voice recorder; never prints, synthesizes, or plays audio."""

    def __init__(self, config: dict, holdmap: HoldMap | None, clock: list[float]) -> None:
        bank_path = Path("data/phrasebank.json")
        self._picker = LinePicker(
            json.loads(bank_path.read_text()) if bank_path.exists() else {}, random.Random(0)
        )
        persona_path = Path(config.get("voice", {}).get("persona", ""))
        persona = yaml.safe_load(persona_path.read_text()) if persona_path.exists() else {}
        self._places = hold_places(holdmap, persona.get("hold_places", ())) if holdmap else {}
        self._colour = holdmap.colour if holdmap else None
        self.clock = clock
        self.busy_until = float("-inf")
        self.lines: list[SpokenLine] = []

    def say(self, req: SpeakRequest) -> bool:
        now = self.clock[0]
        if now > req.deadline:
            return False
        data = req.event.data
        values = slot_values(data, self._colour, self._places)
        line = self._picker.pick(req.event.kind, req.level, req.mood, values)
        phrase = (
            stitch(line, values)[1]
            if line is not None
            else str(data.get("text") or req.event.kind.replace("_", " "))
        )
        self.lines.append(SpokenLine(req.event.t, now, req.event.kind, phrase))
        self.busy_until = now + len(phrase.split()) * 0.35
        return True

    def is_speaking(self) -> bool:
        return self.clock[0] < self.busy_until

    def stop(self) -> None:
        self.busy_until = self.clock[0]

    def debrief(self, history, habits, crux_hold) -> None:
        payload = debrief_payload(history, habits, crux_hold, self._places)
        for sentence in template_sentences(payload):
            self.lines.append(SpokenLine(self.clock[0], self.clock[0], "debrief", sentence))


class _CapturingArbiter(Arbiter):
    def __init__(self, *args, **kwargs):
        self.rejections: list[tuple[Any, str]] = []
        super().__init__(*args, **kwargs)

    def _suppress(self, event, reason: str) -> None:
        self.rejections.append((event, reason))
        super()._suppress(event, reason)


class TranscriptEngine:
    """Run one state at a time through the same trackers, detectors and arbiter as replay."""

    def __init__(self, config: dict, holdmap: HoldMap | None, **_kwargs) -> None:
        self.config, self.holdmap = config, holdmap
        self.clock = [0.0]
        self.voice = _TextVoice(config, holdmap, self.clock)
        self.features = FeatureTracker(float(config["brain"]["move_speed_eps"]))
        self.attempts = AttemptTracker(holdmap, config)
        self.detectors = mvp_detectors(config) + good_tier_detectors(config)
        self.memory = Memory(self.voice, config)
        self.arbiter = _CapturingArbiter(self.voice, config, clock=lambda: self.clock[0])
        self.entries: list[TranscriptEntry] = []
        self._previous_contacts = None
        self._previous_person = None
        self._phase = self.attempts.phase
        self._top_since = None
        self._seen_summaries = 0
        self._last_fall = None
        self._next_state: float | None = None
        self._spoken_count = 0
        self.t0: float | None = None  # capture time of the first state (a camera clock has no zero)

    def update(self, state: ClimberState) -> list[TranscriptEntry]:
        fresh: list[TranscriptEntry] = []
        if self.t0 is None:
            self.t0 = state.t

        def add(source, kind, summary, details=None, spoken=False, t=None):
            row = TranscriptEntry(
                state.t if t is None else float(t), source, kind, summary, details or {}, spoken
            )
            self.entries.append(row)
            fresh.append(row)

        self.clock[0] = state.t
        features = self.features.update(state)
        if self._previous_person != state.person_found:
            add(
                "perception",
                "person_found" if state.person_found else "person_lost",
                "Person found" if state.person_found else "Person lost",
                {"person_found": state.person_found},
            )
        self._previous_person = state.person_found
        if self._previous_contacts is not None:
            for limb in Limb:
                old, new = self._previous_contacts.get(limb), state.contacts.get(limb)
                if old != new:
                    old_text = old if old is not None else "none"
                    new_text = new if new is not None else "none"
                    add(
                        "perception",
                        "contact_change",
                        f"{limb.value}: {old_text} -> {new_text}",
                        {"limb": limb.value, "from": old, "to": new},
                    )
        self._previous_contacts = dict(state.contacts)
        events = list(self.attempts.update(state, features))
        if any(event.kind == "attempt_start" for event in events):
            self.features.reset_highest_hold()
        if self.attempts.phase != self._phase:
            add(
                "attempt",
                "phase_transition",
                f"{self._phase} -> {self.attempts.phase}",
                {"from": self._phase, "to": self.attempts.phase},
            )
            self._phase = self.attempts.phase
        for event in events:
            summary = (
                f"Attempt {event.data.get('attempt_no', '?')} started"
                if event.kind == "attempt_start"
                else f"Attempt event {event.kind}: {event.data}"
            )
            add("attempt", event.kind, summary, event.data, t=event.t)
        if self.attempts._top_since != self._top_since:
            if self.attempts._top_since is not None:
                add(
                    "attempt",
                    "top_timer_start",
                    "Top-hold timer started",
                    {"timer_started_at": self.attempts._top_since},
                )
            elif self._top_since is not None:
                add(
                    "attempt",
                    "top_timer_reset",
                    "Top-hold timer reset before completion",
                    {"previous_start": self._top_since},
                )
            self._top_since = self.attempts._top_since
        while self._seen_summaries < len(self.attempts.finished_attempts):
            summary = self.attempts.finished_attempts[self._seen_summaries]
            self._seen_summaries += 1
            self.memory.on_attempt_end(summary)
            add(
                "memory",
                "attempt_summary",
                f"Attempt {summary.attempt_no} summary recorded: {summary.outcome}",
                asdict(summary),
            )
        ctx = DetectorContext(
            self.config,
            self.holdmap,
            self.attempts.phase == "ON_WALL",
            self._last_fall,
            self.memory.crux_hold(),
        )
        detected = []
        for detector in self.detectors:
            for event in detector.update(features, state, ctx):
                detected.append(event)
                events.append(event)
                add(
                    "detector",
                    event.kind,
                    f"Emitted {event.kind} (priority {event.priority}): {event.data}",
                    {"priority": event.priority, "payload": event.data},
                    t=event.t,
                )
        mem_events = self.memory.observe(state, features)
        events.extend(mem_events)
        for event in mem_events:
            add(
                "memory",
                "memory_event",
                f"Memory emitted {event.kind}: {event.data}",
                {"payload": event.data},
                t=event.t,
            )
        metadata = {}
        for event in events:
            self.memory.on_event(event)
            level, mood = self.memory.level_for(event.kind), self.memory.mood()
            metadata[(event.kind, event.t)] = (mood, level)
            self.arbiter.offer(event, level, mood)
        before = len(self.arbiter.rejections)
        selected = self.arbiter.tick()
        for event, reason in self.arbiter.rejections[before:]:
            mood, level = metadata.get((event.kind, event.t), ("neutral", 0))
            add(
                "arbiter",
                "rejected",
                f"Rejected {event.kind}: {reason}",
                {"event_kind": event.kind, "reason": reason, "mood": mood, "level": level},
                t=state.t,
            )
        if selected:
            mood, level = metadata.get((selected.kind, selected.t), ("neutral", 0))
            add(
                "arbiter",
                "accepted",
                f"Accepted {selected.kind}; mood {mood}, level {level}",
                {
                    "event_kind": selected.kind,
                    "reason": "selected highest priority eligible event",
                    "mood": mood,
                    "level": level,
                },
                t=state.t,
            )
            self.memory.on_spoken(SpokenLine(selected.t, state.t, selected.kind, ""))
            if selected.kind == "fall":
                self._last_fall = selected.t
        for line in self.voice.lines[self._spoken_count :]:
            add(
                "voice",
                "spoken",
                line.text,
                {"text": line.text, "event_kind": line.kind, "event_time": line.t_event},
                True,
                t=state.t,
            )
        self._spoken_count = len(self.voice.lines)
        if self._next_state is None:
            self._next_state = state.t
        if state.t + 1e-5 >= self._next_state:
            n = sum(value is not None for value in state.contacts.values())
            details = {
                "hip_y": state.hip[1],
                "limbs_on_holds": n,
                "phase": self.attempts.phase,
                "current_high_hold": features.highest_hold_id,
                "hip_velocity_y": features.hip_velocity_y,
                "hip_speed": features.hip_speed,
            }
            add(
                "state",
                "periodic_state",
                f"State: hip y={state.hip[1]:.3f}; {n}/4 limbs on holds; "
                f"{self.attempts.phase}; high={features.highest_hold_id}",
                details,
            )
            self._next_state = state.t + 1.0
        return fresh


def run_over_states(states, config, holdmap, **kwargs) -> list[TranscriptEntry]:
    engine = TranscriptEngine(config, holdmap, **kwargs)
    for state in states:
        engine.update(state)
    return engine.entries


def format_line(entry: TranscriptEntry, t0: float = 0.0) -> str:
    """One terminal line; `t0` is subtracted so a live camera clock reads as time since start."""
    return f"{entry.t - t0:6.2f}s [{entry.source}/{entry.kind}] {entry.summary}"


PANEL_DISPLAY_H = 800  # the window is shown about this tall; the panel is drawn for that size
PANEL_BASE_W = 420


def panel_scale(height: int) -> float:
    """Factor that keeps the panel text the same size in a shrunk window, for any frame height."""
    return max(1.0, height / PANEL_DISPLAY_H)


def draw_panel(entries, now_t, height, width=None, max_lines=25, t0=0.0) -> np.ndarray:
    """Colour-coded scrolling log (newest at the bottom), sized for the window the frame is shown
    in: text, spacing and the default width scale with the frame height."""
    k = panel_scale(height)

    def px(base: float) -> int:
        return max(1, round(base * k))

    width = width or px(PANEL_BASE_W)
    image = np.full((height, width, 3), (27, 28, 32), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    colors = {
        "perception": (170, 170, 170),
        "attempt": (240, 160, 75),
        "detector": (20, 150, 255),
        "arbiter": (100, 220, 180),
        "voice": (100, 240, 130),
        "memory": (200, 120, 220),
        "state": (130, 180, 200),
    }
    scale = 0.48 * k
    thin, bold = px(1), px(2)
    pad = px(12)
    cv2.putText(image, "COACH TRANSCRIPT", (pad, px(24)), font, scale, (235, 235, 235), thin)
    latest = next((e for e in reversed(entries) if e.source == "state" and e.t <= now_t), None)
    y = px(44)
    if latest:
        box_scale = scale * 0.78
        box_lines = _wrap_pixels(latest.summary, font, box_scale, thin, width - 2 * pad)[:2]
        box_bottom = min(height - 3, px(31) + px(20) + len(box_lines) * px(18))
        cv2.rectangle(image, (px(7), px(31)), (width - px(7), box_bottom), (46, 48, 54), -1)
        line_y = px(49)
        for line in box_lines:
            cv2.putText(image, line, (pad, line_y), font, box_scale, (190, 205, 215), thin)
            line_y += px(18)
        y = box_bottom + px(8)
    visible = [e for e in entries if e.source != "state" and e.t <= now_t][-max_lines:]
    rows = []
    available = height - y - px(5)
    line_h = px(20)
    gap = px(4)
    for entry in visible:
        label = f"{entry.t - t0:5.1f} {entry.source[:3].upper()} {entry.summary}"
        msg = _wrap_pixels(label, font, scale, bold if entry.spoken else thin, width - 2 * pad)
        rows.append((entry, msg or [""]))
    needed = sum(len(lines) * line_h + gap for _, lines in rows)
    while rows and needed > available:
        _, lines = rows.pop(0)
        needed -= len(lines) * line_h + gap
    for entry, lines in rows:
        color = (
            (70, 70, 245)
            if entry.source == "arbiter" and entry.kind == "rejected"
            else colors.get(entry.source, (200, 200, 200))
        )
        if entry.spoken:
            color = (100, 255, 130)
        fade = max(0.45, 1 - (now_t - entry.t) / 30)
        color = tuple(int(c * fade) for c in color)
        for line in lines:
            cv2.putText(image, line, (pad, y), font, scale, color, bold if entry.spoken else thin)
            y += line_h
        y += gap
    return image


def _wrap_pixels(text: str, font: int, scale: float, thickness: int, max_width: int) -> list[str]:
    """Wrap on words using OpenCV's measured rendered width."""
    lines: list[str] = []
    line = ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        rendered_width = cv2.getTextSize(candidate, font, scale, thickness)[0][0]
        if line and rendered_width > max_width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--json")
    args = parser.parse_args()
    config = load_config()
    holdmap, _ = load_holdmap_with_shapes(args.problem)
    states = [
        state_from_line(line)
        for line in Path(args.cache).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    entries = run_over_states(states, config, holdmap)
    for row in entries:
        print(format_line(row))
    if args.json:
        Path(args.json).write_text(
            json.dumps([asdict(row) for row in entries], indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
