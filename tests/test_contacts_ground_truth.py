"""Hand-labelled limb/hold contacts and hand heights on climb1_a1 (skipped if data is missing).

Needs, all gitignored:
  data/holdmaps/test..json                  the 6-hold map built in the editor (problem id "test.")
  data/pose_cache/climb1_a1.test.jsonl   pose states recorded with that map

Labels come from the user watching the clip at 30 fps (frame = seconds x 30). Hold ids:
1 = S1, 2 = S2 (the two lowest yellow holds), 3 and 5 = the next yellow holds, 4 = the
right-hand rest, 6 = the top hold. None = no map hold (e.g. holding the wall's top edge).
Frames 270-390 (9-13 s) and 680 were "correct as shown in the viewer": stored as they were then.

KNOWN_MISSES are labels the current code gets wrong (it says "no contact" where the user says a
hold; it never picks a wrong hold). They are xfail(strict=True), so a fix that makes one pass
is noticed and the entry should then be removed.
"""

from pathlib import Path

import pytest

from coach.contracts import Limb
from coach.perception.contacts import ContactTracker
from coach.perception.holds import holdmap_path, load_holdmap
from coach.perception.pose import load_perception_config
from coach.perception.serialize import state_from_line

CACHE = Path(__file__).resolve().parents[1] / "data" / "pose_cache" / "climb1_a1.test.jsonl"
PROBLEM = "test."
LH, RH, LF, RF = Limb.LH, Limb.RH, Limb.LF, Limb.RF

TRUTH = {  # frame -> {limb: hold id or None}
    60: {LH: 5, RH: 4, LF: 1, RF: 1},  # 2 s
    120: {LH: 5, RH: 4, LF: 2, RF: 1},  # 4 s
    150: {LH: 5, RH: 4, LF: 2, RF: 2},  # 5 s
    180: {LH: 5, RH: 4, LF: 2, RF: 2},  # 6 s
    210: {LH: 5, RH: 4, LF: 3, RF: 2},  # 7 s
    240: {LH: 5, RH: None, LF: 3, RF: 2},  # 8 s
    270: {LH: 5, RH: None, LF: 3, RF: 2},  # 9-13 s: as shown in the viewer
    300: {LH: 5, RH: None, LF: 3, RF: 2},
    330: {LH: 5, RH: None, LF: 3, RF: 2},
    360: {LH: 5, RH: None, LF: 3, RF: 2},
    390: {LH: 5, RH: None, LF: 3, RF: 2},
    420: {LH: 5, RH: None, LF: 3, RF: 3},  # 14 s (right hand on the wall's top edge)
    450: {LH: 5, RH: None, LF: 3, RF: 3},  # 15 s
    543: {LH: None, RH: None, LF: 5, RF: 3},  # f543
    634: {LF: 5, RF: 5},  # f634 (hands not stated)
    642: {LF: 5, RF: 5},  # f642: both feet on hold 5
    680: {LH: None, RH: None, LF: 6, RF: 5},  # f680: as shown in the viewer
}
KNOWN_MISSES = {
    (150, RF),  # settles 0.15 s after stepping from hold 1 to 2, so f150 is just inside the window
    (240, RH),  # still inside hold 4's box at its top edge; releases at f246 (hysteresis lag)
}


@pytest.fixture(scope="module")
def states():
    if not CACHE.exists() or not holdmap_path(PROBLEM).exists():
        pytest.skip("local clip data (hold map + pose cache) not present")
    return [state_from_line(line) for line in CACHE.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def contacts(states):
    tracker = ContactTracker(load_perception_config(), load_holdmap(PROBLEM))
    return [tracker.update(s) for s in states]


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


def test_never_assigns_a_wrong_hold_where_the_user_labelled_one(contacts):
    """Misses are 'no contact'; a different hold than the label would be a worse error."""
    wrong = [
        (f, limb.name, hold, contacts[f][limb])
        for f, expected in TRUTH.items()
        for limb, hold in expected.items()
        if hold is not None and contacts[f][limb] not in (None, hold)
    ]
    assert wrong == []


def test_right_hand_is_above_the_left_hand_while_hidden_behind_the_head(states):
    """16 s (f478-f500): the right hand is blocked by the head but is really just above the left.
    Its landmarks are almost invisible there, so the dot is held instead of following the guess."""
    for frame in (466, 478, 480, 490, 500):
        s = states[frame]
        assert s.limbs[RH][1] < s.limbs[LH][1], f"f{frame}: right hand should be above the left"


def test_right_hand_has_not_moved_since_f642_and_is_not_below_the_waist(states):
    """User: the right hand (hidden/blocked) has not moved since f642 and is on the wall's top
    edge, so its dot must stay put and above the waist until the end of the clip (f678 was drawn
    below the waist when low-visibility guesses were followed). The waist is the median hip
    height, because the hip itself glitches for single frames (e.g. f655)."""
    frames = [f for f in range(650, 685) if states[f].person_found]
    waist_y = sorted(states[f].hip[1] for f in frames)[len(frames) // 2]
    ref = states[642].limbs[RH]
    for frame in frames:
        x, y = states[frame].limbs[RH]
        assert y < waist_y, f"f{frame}: right hand drawn below the waist"
        assert abs(x - ref[0]) < 0.05 and abs(y - ref[1]) < 0.05, f"f{frame}: right hand moved"
