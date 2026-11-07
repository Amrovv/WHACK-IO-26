"""Hand-labelled limb/hold contacts on Climb 2 (skipped if the local data is missing).

Needs, all gitignored:
  data/holdmaps/climb2.json         the polygon hold map (problem id "climb2")
  data/pose_cache/climb2_a1.jsonl   pose states recorded from data/footage/climb2_a1.mp4

Labels are what the user sees in the video, by frame number. Hold ids: 1 = start triangle,
2 = small start hold, 3/4/5 = the three diagonal slabs, 6 = pill, 7 = pocket, 8 = top slab.
None = not on any hold.

KNOWN_MISSES are labels the current code gets wrong; xfail(strict=True) flags a fix.
"""

from pathlib import Path

import pytest

from coach.contracts import Limb
from coach.perception.contacts import ContactTracker
from coach.perception.holds import holdmap_path, load_holdmap_with_shapes
from coach.perception.pose import load_perception_config
from coach.perception.serialize import state_from_line

CACHE = Path(__file__).resolve().parents[1] / "data" / "pose_cache" / "climb2_a1.jsonl"
PROBLEM = "climb2"
LH, RH, LF, RF = Limb.LH, Limb.RH, Limb.LF, Limb.RF

TRUTH = {  # frame -> {limb: hold id or None}
    42: {LH: 2, RH: 3, LF: 1, RF: None},  # the right foot is NOT on hold 1
    53: {LH: 2, LF: 1, RF: None},  # the left hand stays on 2 (f52 had it)
    78: {LH: 4, RH: 3, LF: None},  # the left foot has left hold 1 (a triangle, not a box)
    96: {LH: 5, RH: 5},  # both hands on hold 5
}
KNOWN_MISSES = {
    (78, LH),  # the hand is still arriving: it has not been still for the settle time
    (78, LF),  # the smoothed foot point lags the real foot, so it is still near hold 1
    (96, LH),  # smoothing lag keeps the left hand point just outside hold 5
}


@pytest.fixture(scope="module")
def contacts():
    if not CACHE.exists() or not holdmap_path(PROBLEM).exists():
        pytest.skip("local Climb 2 data (hold map + pose cache) not present")
    holdmap, shapes = load_holdmap_with_shapes(PROBLEM)
    tracker = ContactTracker(load_perception_config(), holdmap, shapes)
    lines = CACHE.read_text(encoding="utf-8").splitlines()
    return [tracker.update(state_from_line(line)) for line in lines]


CASES = [
    pytest.param(
        frame,
        limb,
        hold,
        marks=[pytest.mark.xfail(strict=True, reason="known contact miss")]
        if (frame, limb) in KNOWN_MISSES
        else [],
        id=f"f{frame}-{limb.name}",
    )
    for frame, expected in TRUTH.items()
    for limb, hold in expected.items()
]


@pytest.mark.parametrize(("frame", "limb", "hold"), CASES)
def test_contact_matches_hand_label(contacts, frame, limb, hold):
    assert contacts[frame][limb] == hold
