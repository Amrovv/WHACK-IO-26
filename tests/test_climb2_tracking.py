"""Climb 2 (skipped if the local pose cache is missing): the pose must follow the climber, not the
big, partly cropped bystander standing in the foreground at the right edge (hip x about 0.97).

Record the cache with:
    uv run python scripts/record_pose.py --source file:data/footage/climb2_a1.mp4 \
        --out data/pose_cache/climb2_a1.jsonl
"""

from pathlib import Path

import pytest

from coach.perception.serialize import state_from_line

CACHE = Path(__file__).resolve().parents[1] / "data" / "pose_cache" / "climb2_a1.jsonl"
BYSTANDER_X = 0.8  # hip x beyond this is the foreground bystander at the right edge
CLIMBER_DETECTED_FROM = 40  # frames 0-34 (the climber crouches on the floor) are not detected


@pytest.fixture(scope="module")
def states():
    if not CACHE.exists():
        pytest.skip("local Climb 2 pose cache not present")
    return [state_from_line(line) for line in CACHE.read_text(encoding="utf-8").splitlines()]


def test_follows_the_climber_not_the_bystander_once_the_climber_is_detected(states):
    wrong = [
        s.frame_idx
        for s in states[CLIMBER_DETECTED_FROM:]
        if s.person_found and s.hip[0] > BYSTANDER_X
    ]
    assert wrong == []


def test_the_climber_is_found_in_almost_every_frame(states):
    found = sum(s.person_found for s in states[CLIMBER_DETECTED_FROM:])
    assert found >= len(states[CLIMBER_DETECTED_FROM:]) - 2
