"""Hand-labelled left/right checks on a real clip (skipped if the pose cache isn't present).

The cache is gitignored: record it with
    uv run python scripts/record_pose.py --source file:data/footage/climb1_a1.mp4 \
        --out data/pose_cache/climb1_a1.jsonl

Ground truth (watched by the user): which of the climber's hands is higher.
"""

from pathlib import Path

import pytest

from coach.contracts import Limb
from coach.perception.serialize import state_from_line

CACHE = Path(__file__).resolve().parents[1] / "data" / "pose_cache" / "climb1_a1.jsonl"

# (time s, frame at 30 fps, higher hand): "the climber's right hand is higher", y smaller = higher
RIGHT_HAND_HIGHER = [(14.5, 435), (17.93, 538), (18.17, 545)]


@pytest.fixture(scope="module")
def states():
    if not CACHE.exists():
        pytest.skip(f"pose cache not found: {CACHE}")
    return [state_from_line(line) for line in CACHE.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize(("t", "frame"), RIGHT_HAND_HIGHER)
def test_right_hand_is_the_higher_hand(states, t, frame):
    s = states[frame]
    assert s.person_found
    assert s.limbs[Limb.RH][1] < s.limbs[Limb.LH][1], f"at {t}s the right hand should be higher"
