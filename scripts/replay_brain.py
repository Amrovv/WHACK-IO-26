"""Replay pose-cache JSONL through the local feature and attempt brain."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Allow running as `python scripts/replay_brain.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coach.brain.attempts import AttemptTracker  # noqa: E402
from coach.brain.features import FeatureTracker  # noqa: E402
from coach.contracts import ClimberState, Hold, HoldMap, Limb  # noqa: E402

LOG = logging.getLogger(__name__)


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
    data = json.loads(path.read_text(encoding="utf-8"))
    holds = [Hold(**hold) for hold in data["holds"]]
    return HoldMap(
        data["problem_id"], data["colour"], holds, tuple(data["image_size"]), data["mode"]
    )


def _config() -> dict[str, Any]:
    import yaml

    return yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose", required=True, type=Path)
    parser.add_argument("--holdmap", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not args.pose.exists():
        LOG.error("Pose file does not exist: %s", args.pose)
        return 1
    config = _config()
    tracker = FeatureTracker(float(config["brain"]["move_speed_eps"]))
    attempts = AttemptTracker(_holdmap(args.holdmap) if args.holdmap else None, config)
    with args.pose.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            state = _state(json.loads(line))
            events = attempts.update(state, tracker.update(state))
            for event in events:
                LOG.info("t=%.3f event=%s %s", event.t, event.kind, event.data)
            while len(getattr(attempts, "_replay_reported", [])) < len(attempts.finished_attempts):
                reported = getattr(attempts, "_replay_reported", [])
                summary = attempts.finished_attempts[len(reported)]
                LOG.info("t=%.3f summary=%s", state.t, asdict(summary))
                reported.append(summary)
                attempts._replay_reported = reported
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
