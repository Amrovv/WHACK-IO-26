"""Diagonal holds: the polygon is saved beside the bbox and used for contacts."""

import json

import numpy as np
import pytest

from coach.contracts import ClimberState, Hold, HoldMap, Limb
from coach.perception.contacts import ContactTracker
from coach.perception.holds import (
    build_holdmap,
    build_holdmap_with_shapes,
    draw_holds,
    load_hold_shapes,
    load_holdmap,
    load_holdmap_with_shapes,
    save_holdmap,
)
from coach.perception.pose import LANDMARK_NAMES, PoseTracker, load_perception_config
from coach.perception.shapes import polygon_bounds

CFG_HOLDS = {"start_count": 2}
W, H = 1000, 500

# Two parallel slanted slabs (like the white slabs in Climb 2) whose bounding boxes overlap.
SLAB_A = [(0.20, 0.20), (0.26, 0.20), (0.46, 0.50), (0.40, 0.50)]
SLAB_B = [(0.34, 0.20), (0.40, 0.20), (0.60, 0.50), (0.54, 0.50)]


def assert_same_shapes(a, b):
    assert set(a) == set(b)
    for key in a:
        flat_a = [c for point in a[key] for c in point]
        flat_b = [c for point in b[key] for c in point]
        assert flat_a == pytest.approx(flat_b)


def _px(poly):
    return [(round(x * W), round(y * H)) for x, y in poly]


def _box(poly):
    x0, y0, x1, y1 = polygon_bounds(_px(poly))
    return (x0, y0, x1, y1)


def test_polygon_hold_keeps_a_contract_bbox_and_gets_the_polygon_centroid():
    boxes = [_box(SLAB_A), _box(SLAB_B)]
    polys = {_box(SLAB_A): _px(SLAB_A), _box(SLAB_B): _px(SLAB_B)}
    hm, shapes = build_holdmap_with_shapes(
        "t", "white", boxes, (W, H), "manual", CFG_HOLDS, polygons=polys
    )
    assert [h.id for h in hm.holds] == [1, 2] and set(shapes) == {1, 2}
    for hold in hm.holds:
        poly = shapes[hold.id]
        assert hold.bbox == pytest.approx(polygon_bounds(poly))  # contract box = polygon bounds
    centre_a = next(h for h in hm.holds if h.bbox[0] == pytest.approx(0.20)).center
    assert centre_a == pytest.approx((0.33, 0.35), abs=0.005)  # centroid of the parallelogram


def test_build_holdmap_without_shapes_is_unchanged():
    boxes = [(100, 100, 150, 150), (300, 300, 360, 380)]
    hm = build_holdmap("t", "red", boxes, (W, H), "colour", CFG_HOLDS)
    hm2, shapes = build_holdmap_with_shapes("t", "red", boxes, (W, H), "colour", CFG_HOLDS)
    assert hm == hm2 and shapes == {}


def test_save_load_roundtrip_keeps_polygons_and_old_maps_have_none(tmp_path):
    boxes = [_box(SLAB_A), (500, 400, 560, 450)]
    hm, shapes = build_holdmap_with_shapes(
        "t", "white", boxes, (W, H), "manual", CFG_HOLDS, polygons={_box(SLAB_A): _px(SLAB_A)}
    )
    save_holdmap(hm, tmp_path, shapes)
    data = json.loads((tmp_path / "t.json").read_text(encoding="utf-8"))
    with_poly = [h for h in data["holds"] if "polygon" in h]
    assert len(with_poly) == 1 and len(with_poly[0]["polygon"]) == 4
    back, back_shapes = load_holdmap_with_shapes("t", tmp_path)
    assert back == hm
    assert_same_shapes(back_shapes, shapes)
    assert_same_shapes(load_hold_shapes("t", tmp_path), shapes)

    save_holdmap(hm, tmp_path)  # a map saved without shapes (the old format)
    assert load_hold_shapes("t", tmp_path) == {}
    assert load_holdmap("t", tmp_path) == hm  # the shared contract object is the same either way


def test_draw_holds_draws_a_polygon_outline_not_its_box():
    img = np.zeros((H, W, 3), np.uint8)
    hm, shapes = build_holdmap_with_shapes(
        "t",
        "white",
        [_box(SLAB_A)],
        (W, H),
        "manual",
        CFG_HOLDS,
        polygons={_box(SLAB_A): _px(SLAB_A)},
    )
    out = draw_holds(img, hm, shapes)
    x0, y0, x1, y1 = _box(SLAB_A)
    # a corner of the bounding box is empty wall for a slanted slab: nothing is drawn there
    assert not out[y0 + 3, x1 - 3].any()
    # but a point on the slab's slanted edge is drawn
    ex, ey = round(0.26 * W), round(0.20 * H)
    assert out[ey : ey + 4, ex - 2 : ex + 3].any()


# ---- contacts against shapes --------------------------------------------------------------
CFG = {
    **load_perception_config(),  # the real settings (so new keys are there), this test's on top
    "contact_margin": 0.15,
    "contact_settle_ms": 150,
    "contact_release_ratio": 1.3,
    "contact_still_speed": 0.2,
    "min_visibility": 0.5,
    "resume_visibility": 0.7,
    "resume_frames": 2,
    "hold_lookback_frames": 4,
    "lost_memory_frames": 5,
    "arm_length_frames": 60,
    "arm_length_default": 0.25,
    "num_poses": 2,
    "lr_swap_frames": 5,
    "lr_min_x_gap": 0.03,
    "max_width": 640,
    "one_euro": {"min_cutoff": 1.0, "beta": 0.05, "d_cutoff": 1.0},
}
DT = 1 / 30


def slab_map():
    """Hold 1 = slab A, hold 2 = slab B (ids are bottom to top; both span the same heights, so
    we build the ids explicitly here)."""
    a_box = (0.20, 0.20, 0.46, 0.50)
    b_box = (0.34, 0.20, 0.60, 0.50)
    holds = [
        Hold(1, (0.33, 0.35), a_box, "start"),
        Hold(2, (0.47, 0.35), b_box, "top"),
    ]
    hm = HoldMap("t", "white", holds, (W, H), "manual")
    return hm, {1: SLAB_A, 2: SLAB_B}


def make_state(t, limbs):
    kp = {name: (0.5, 0.5, 1.0) for name in LANDMARK_NAMES}
    for side in ("left", "right"):
        kp[f"{side}_shoulder"], kp[f"{side}_elbow"], kp[f"{side}_wrist"] = (
            (0.5, 0.40, 1.0),
            (0.5, 0.55, 1.0),
            (0.5, 0.70, 1.0),
        )
    return ClimberState(
        t, round(t / DT), True, kp, limbs, {limb: None for limb in Limb}, (0.5, 0.5)
    )


def settle(tracker, point, limb=Limb.RH, seconds=0.4):
    out = None
    for i in range(int(seconds / DT) + 1):
        out = tracker.update(make_state(i * DT, {limb: point}))[limb]
    return out


def test_a_limb_on_one_slab_inside_the_other_slabs_box_goes_to_the_right_slab():
    hm, shapes = slab_map()
    on_a = (0.42, 0.47)  # on slab A's body, but also inside slab B's bounding box
    assert shapes[1] and settle(ContactTracker(CFG, hm, shapes), on_a) == 1
    # box only: both boxes contain the point and the nearer centre (slab B) wins: wrong
    assert settle(ContactTracker(CFG, hm), on_a) == 2


def test_a_limb_in_an_empty_corner_of_the_box_is_not_in_contact_with_a_slanted_slab():
    hm, shapes = slab_map()
    empty_corner = (0.24, 0.47)  # inside slab A's bbox, far from the slanted slab
    assert settle(ContactTracker(CFG, hm, shapes), empty_corner) is None
    assert settle(ContactTracker(CFG, hm), empty_corner) == 1  # a box would claim it


def test_a_polygon_contact_is_released_only_beyond_the_release_zone():
    hm, shapes = slab_map()
    tracker = ContactTracker(CFG, hm, shapes)
    assert settle(tracker, (0.42, 0.47)) == 1  # arm 0.30: margin 0.045, release 0.0585
    t = 1.0
    just_off = (0.42 + 0.05, 0.47)  # 0.05 right of A's edge: outside the margin, inside release
    assert tracker.update(make_state(t, {Limb.RH: just_off}))[Limb.RH] == 1
    far = (0.42 + 0.30, 0.47)  # moves far: released (and B's slab is far enough not to catch it)
    assert tracker.update(make_state(t + DT, {Limb.RH: far}))[Limb.RH] is None


def test_holds_without_a_polygon_still_use_their_box_next_to_polygon_holds():
    hm, shapes = slab_map()
    shapes = {1: shapes[1]}  # only slab A has a polygon; B stays a box
    tracker = ContactTracker(CFG, hm, shapes)
    assert settle(tracker, (0.56, 0.22)) == 2  # inside B's box (the box is all B has)


def test_pose_tracker_passes_shapes_to_the_contact_tracker():
    hm, shapes = slab_map()
    lms = {name: (0.5, 0.5, 0.9) for name in LANDMARK_NAMES}
    for side in ("left", "right"):
        lms[f"{side}_shoulder"], lms[f"{side}_elbow"] = (0.5, 0.40, 0.9), (0.5, 0.55, 0.9)
    lms["right_wrist"] = lms["right_index"] = (0.42, 0.47, 0.9)  # on slab A, inside B's box
    pose = [lms[n] for n in LANDMARK_NAMES]
    frame = np.zeros((48, 64, 3), np.uint8)
    cfg = {**CFG}
    for use_shapes, expected in ((True, 1), (False, 2)):
        tracker = PoseTracker(
            cfg, detector=lambda f, ts: [pose], holdmap=hm, shapes=shapes if use_shapes else None
        )
        state = None
        for i in range(15):
            state = tracker.process(i * DT, i, frame)
        assert state.contacts[Limb.RH] == expected
