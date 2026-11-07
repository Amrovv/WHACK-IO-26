"""Replay pose-cache JSONL through the FULL coach brain pipeline.

Runs FeatureTracker -> AttemptTracker -> detectors -> Memory -> Arbiter in the
exact order specified by docs/INTEGRATION-contract.md.  Use --voice print to
see what the coach would actually say on real gym footage.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Allow running as `python scripts/replay_brain.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coach.brain.arbiter import Arbiter  # noqa: E402
from coach.brain.attempts import AttemptTracker  # noqa: E402
from coach.brain.detectors import DetectorContext, good_tier_detectors, mvp_detectors  # noqa: E402
from coach.brain.features import FeatureTracker  # noqa: E402
from coach.brain.memory import Memory  # noqa: E402
from coach.contracts import (  # noqa: E402
    AttemptSummary,
    ClimberState,
    Hold,
    HoldMap,
    Limb,
    SpeakRequest,
    SpokenLine,
)

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PrintVoice — implements the Voice protocol; outputs via logging only.
# ---------------------------------------------------------------------------


class PrintVoice:
    """Minimal Voice implementation that logs cues instead of playing audio."""

    def say(self, req: SpeakRequest) -> bool:
        LOG.info(
            "[CUE] kind=%-20s priority=%d mood=%-12s level=%d deadline=%.3f",
            req.event.kind,
            req.event.priority,
            req.mood,
            req.level,
            req.deadline,
        )
        return True

    def is_speaking(self) -> bool:
        return False

    def stop(self) -> None:
        pass  # no-op for a print voice

    def debrief(
        self,
        history: list[AttemptSummary],
        habits: dict,
        crux_hold: int | None,
    ) -> None:
        attempt = history[-1] if history else None
        LOG.info(
            "[DEBRIEF] attempt=%s outcome=%s high_hold=%s crux=%s",
            attempt.attempt_no if attempt else "?",
            attempt.outcome if attempt else "?",
            attempt.high_hold_id if attempt else "?",
            crux_hold,
        )


# ---------------------------------------------------------------------------
# Helpers (kept from the original; do not modify per task spec)
# ---------------------------------------------------------------------------


def _state(data: dict[str, Any]) -> ClimberState:
    limbs = {Limb(key): tuple(value) for key, value in data["limbs"].items()}
    contacts = {Limb(key): value for key, value in data["contacts"].items()}
    return ClimberState(
        data["t"],
        data["frame_idx"],
        data["person_found"],
        data["keypoints"],
        limbs,
        contacts,
        tuple(data["hip"]),
    )


def _holdmap(path: Path) -> HoldMap:
    # A's hold-map files may carry extra fields (e.g. "polygon") alongside the
    # four contract fields; Hold only takes the contract fields.
    data = json.loads(path.read_text(encoding="utf-8"))
    holds = [
        Hold(
            id=hold["id"],
            center=tuple(hold["center"]),
            bbox=tuple(hold["bbox"]),
            role=hold.get("role", "mid"),
        )
        for hold in data["holds"]
    ]
    return HoldMap(
        data["problem_id"], data["colour"], holds, tuple(data["image_size"]), data["mode"]
    )


def _config() -> dict[str, Any]:
    import yaml

    return yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay a pose-cache JSONL through the full brain."
    )
    parser.add_argument(
        "--pose", required=True, type=Path, help="Path to the pose-cache JSONL file."
    )
    parser.add_argument("--holdmap", type=Path, help="Optional hold-map JSON file.")
    parser.add_argument(
        "--voice",
        choices=["print"],
        default="print",
        help="Voice backend (only 'print' is supported).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.pose.exists():
        LOG.error("Pose file does not exist: %s", args.pose)
        return 1

    config = _config()
    holdmap = _holdmap(args.holdmap) if args.holdmap else None

    # ------------------------------------------------------------------
    # Construct once, at startup  (INTEGRATION-contract.md §"Construct once")
    # ------------------------------------------------------------------
    voice = PrintVoice()
    ft = FeatureTracker(float(config["brain"]["move_speed_eps"]))
    at = AttemptTracker(holdmap, config)
    detectors = mvp_detectors(config) + good_tier_detectors(config)
    mem = Memory(voice, config)

    # Mutable single-element list so the lambda always returns the latest state.t
    # (mirrors the clock_now = [0.0] pattern in tests/test_pipeline.py).
    clock_now: list[float] = [0.0]
    arbiter = Arbiter(voice, config, clock=lambda: clock_now[0])

    last_fall_t: float | None = None
    seen_summaries: int = 0  # replaces the old _replay_reported hack

    # ------------------------------------------------------------------
    # Per-frame loop
    # ------------------------------------------------------------------
    with args.pose.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue

            state = _state(json.loads(line))

            # 1. Feature extraction.
            features = ft.update(state)

            # 2. Attempt tracking; reset highest-hold on new attempt.
            events = list(at.update(state, features))
            for ev in events:
                if ev.kind == "attempt_start":
                    ft.reset_highest_hold()

            # 3. Drain finished attempts into memory BEFORE querying it.
            while seen_summaries < len(at.finished_attempts):
                summary = at.finished_attempts[seen_summaries]
                LOG.info(
                    "[ATTEMPT] #%d outcome=%-12s high_hold=%s duration=%.1fs",
                    summary.attempt_no,
                    summary.outcome,
                    summary.high_hold_id,
                    summary.duration_s,
                )
                mem.on_attempt_end(summary)
                seen_summaries += 1

            # 4. Build detector context for this frame.
            on_wall = at.phase == "ON_WALL"
            ctx = DetectorContext(config, holdmap, on_wall, last_fall_t, mem.crux_hold())

            # 5. Run detectors and memory's live view.
            for det in detectors:
                events += det.update(features, state, ctx)
            events += mem.observe(state, features)

            # 6. Feed every event into memory and offer to arbiter.
            for ev in events:
                LOG.debug("t=%.3f event=%-20s %s", ev.t, ev.kind, ev.data)
                mem.on_event(ev)
                arbiter.offer(ev, mem.level_for(ev.kind), mem.mood())

            # 7. Set the arbiter clock to this frame's capture time, then tick.
            clock_now[0] = state.t
            spoken = arbiter.tick()

            # 8. Handle a spoken event.
            if spoken is not None:
                LOG.info("[SPOKEN] t=%.3f kind=%s", spoken.t, spoken.kind)
                if spoken.kind == "fall":
                    last_fall_t = spoken.t
                # Simulate C publishing the SpokenLine back to the brain so that
                # compliance tracking, mood, and told_you_so all work correctly.
                mem.on_spoken(
                    SpokenLine(
                        t_event=spoken.t,
                        t_audio_start=state.t,
                        kind=spoken.kind,
                        text="",
                    )
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
