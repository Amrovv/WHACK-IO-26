import cv2
import numpy as np
import pytest

from coach.perception import holds
from coach.perception.holds import (
    build_holdmap,
    circular_median_hue,
    colour_name,
    colour_only_holds,
    detect_hold_polygons,
    detect_holds,
    filter_by_colour,
    hue_distance,
    load_holdmap,
    load_holds_config,
    sample_colour,
    save_holdmap,
)

CFG = {
    "detector_model": "fake/1",
    "api_url": "https://example.invalid",
    "polygon_epsilon": 0.02,
    "hue_tolerance": 12,
    "min_saturation": 60,
    "white_min_value": 210,
    "white_max_saturation": 55,
    "pale_min_saturation": 35,
    "pale_min_fraction": 0.4,
    "white_min_fraction": 0.15,
    "patch_radius_px": 12,
    "morph_kernel_px": 5,
    "min_area": 0.0003,
    "max_area": 0.02,
    "split_peak_ratio": 0.75,
    "start_count": 2,
}
W, H = 300, 200
GREY = (120, 120, 120)


def bgr(hue, sat=200, val=220):
    return tuple(
        int(v) for v in cv2.cvtColor(np.uint8([[[hue, sat, val]]]), cv2.COLOR_HSV2BGR)[0, 0]
    )


def blob_image(blobs):
    """Grey wall with filled circles: blobs = [(cx, cy, radius, hue), ...]."""
    img = np.full((H, W, 3), GREY, np.uint8)
    for cx, cy, r, hue in blobs:
        cv2.circle(img, (cx, cy), r, bgr(hue), -1)
    return img


def test_config_has_hold_keys():
    cfg = load_holds_config()
    for key in ("patch_radius_px", "morph_kernel_px", "min_area", "max_area", "start_count"):
        assert key in cfg


def test_hue_distance_wraps_around():
    assert hue_distance(2, 177) == 5
    assert hue_distance(0, 179) == 1
    assert hue_distance(10, 10) == 0
    assert hue_distance(0, 90) == 90


def test_circular_median_of_reds_on_both_sides_of_zero():
    m = circular_median_hue(np.array([2, 177, 178, 1, 179]))
    assert hue_distance(m, 0) <= 2  # near 0/179, not the naive median 2 or the mean ~107


def test_colour_name():
    assert colour_name(2) == "red" and colour_name(178) == "red"
    assert colour_name(25) == "yellow" and colour_name(160) == "pink"


def test_sample_colour_ignores_grey_pixels_near_the_edge():
    img = blob_image([(100, 100, 12, 30)])
    # click right on the blob edge so the patch includes plenty of grey wall
    hue, sat = sample_colour(img, (111, 100), CFG)
    assert hue == pytest.approx(30, abs=1) and sat > 150


def test_sample_colour_on_grey_raises():
    with pytest.raises(ValueError):
        sample_colour(blob_image([]), (50, 50), CFG)


def test_sample_white_and_filter_white_holds_only():
    img = np.full((H, W, 3), (180, 190, 200), np.uint8)  # beige/grey wall, V=200
    cv2.rectangle(img, (30, 30), (70, 60), (245, 245, 245), -1)
    cv2.rectangle(img, (100, 30), (140, 60), bgr(2), -1)
    cv2.rectangle(img, (170, 30), (210, 60), (10, 10, 10), -1)
    boxes = [(30, 30, 71, 61), (100, 30, 141, 61), (170, 30, 211, 61), (220, 30, 260, 61)]
    ref, _ = sample_colour(img, (50, 45), CFG)
    assert ref < 0 and colour_name(ref) == "white"
    assert filter_by_colour(img, boxes, ref, CFG) == boxes[:1]


def test_polygon_pixels_are_used_instead_of_wall_in_box():
    img = np.full((H, W, 3), (180, 190, 200), np.uint8)
    cv2.rectangle(img, (100, 70), (120, 90), bgr(30), -1)
    box = (90, 60, 130, 100)
    polygon = {box: [(100, 70), (120, 70), (120, 90), (100, 90)]}
    hue = filter_by_colour(img, [box], 30, CFG, polygon)
    assert hue == [box]


def test_colour_only_finds_red_blobs_across_the_hue_wrap_and_ignores_others():
    img = blob_image(
        [
            (40, 40, 10, 2),  # red just above 0
            (120, 90, 10, 177),  # red just below 180
            (200, 150, 10, 179),
            (250, 50, 10, 110),  # blue: must be ignored
        ]
    )
    boxes = colour_only_holds(img, ref_hue=0, cfg=CFG)
    assert len(boxes) == 3
    centres = sorted(((x0 + x1) // 2, (y0 + y1) // 2) for x0, y0, x1, y1 in boxes)
    assert centres == [(40, 40), (120, 90), (200, 150)]


def test_colour_only_applies_area_limits():
    img = blob_image([(40, 40, 1, 30), (120, 90, 10, 30), (230, 100, 60, 30)])
    boxes = colour_only_holds(img, ref_hue=30, cfg=CFG)  # speck too small, giant too big
    assert len(boxes) == 1
    assert (boxes[0][0] + boxes[0][2]) // 2 == 120


def test_filter_by_colour_keeps_matching_hues_including_wrap():
    img = blob_image([(40, 40, 10, 2), (120, 90, 10, 177), (200, 150, 10, 110), (250, 50, 10, 60)])
    boxes = [
        (c[0] - 10, c[1] - 10, c[0] + 10, c[1] + 10)
        for c in ((40, 40), (120, 90), (200, 150), (250, 50))
    ]
    kept = filter_by_colour(img, boxes, ref_hue=179, cfg=CFG)
    assert kept == boxes[:2]


def test_filter_by_colour_drops_boxes_with_no_saturated_pixels():
    img = blob_image([])
    assert filter_by_colour(img, [(10, 10, 50, 50)], ref_hue=30, cfg=CFG) == []


def test_build_holdmap_numbers_bottom_to_top_and_marks_roles():
    # y grows downward: the box with the largest y is lowest on the wall and must be hold 1
    boxes = [(50, 20, 70, 40), (50, 150, 70, 170), (100, 90, 120, 110), (100, 120, 120, 140)]
    hm = build_holdmap("demo", "yellow", boxes, (W, H), "colour", CFG)
    assert [h.id for h in hm.holds] == [1, 2, 3, 4]
    assert [h.center[1] for h in hm.holds] == sorted((h.center[1] for h in hm.holds), reverse=True)
    assert [h.role for h in hm.holds] == ["start", "start", "mid", "top"]
    assert hm.holds[0].center == pytest.approx((60 / W, 160 / H))
    assert hm.holds[0].bbox == pytest.approx((50 / W, 150 / H, 70 / W, 170 / H))
    assert hm.image_size == (W, H) and hm.mode == "colour" and hm.colour == "yellow"


def test_build_holdmap_single_hold_is_start_not_top():
    hm = build_holdmap("demo", "red", [(10, 10, 20, 20)], (W, H), "manual", CFG)
    assert [h.role for h in hm.holds] == ["start"]


def test_holdmap_save_load_roundtrip(tmp_path):
    boxes = [(50, 20, 70, 40), (50, 150, 70, 170), (100, 90, 120, 110)]
    hm = build_holdmap("pink-v3", "pink", boxes, (W, H), "detector", CFG)
    path = save_holdmap(hm, tmp_path)
    assert path.name == "pink-v3.json"
    assert load_holdmap("pink-v3", tmp_path) == hm


def test_detect_holds_needs_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    monkeypatch.setattr(holds, "REPO_ROOT", tmp_path)  # no .env here
    with pytest.raises(RuntimeError, match="ROBOFLOW_API_KEY"):
        detect_holds(blob_image([(10, 10, 3, 30)]), CFG)


def test_detect_holds_converts_boxes_and_caches(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, api_url, api_key):
            assert api_key == "k"

        def infer(self, image, model_id):
            calls.append(model_id)
            return {"predictions": [{"x": 100, "y": 80, "width": 20, "height": 10}]}

    import inference_sdk

    monkeypatch.setenv("ROBOFLOW_API_KEY", "k")
    monkeypatch.setattr(inference_sdk, "InferenceHTTPClient", FakeClient)
    holds._detect_cache.clear()
    img = blob_image([(100, 80, 5, 30)])
    assert detect_holds(img, CFG) == [(90, 75, 110, 85)]
    assert detect_holds(img, CFG) == [(90, 75, 110, 85)]
    assert calls == ["fake/1"]  # second call served from the cache


def test_colour_only_splits_touching_holds_that_merge_into_one_big_blob():
    # Two same-colour holds overlapping: merged area exceeds max_area, but each alone is fine.
    img = blob_image([(100, 100, 18, 30), (130, 100, 18, 30), (230, 150, 18, 30)])
    boxes = colour_only_holds(img, ref_hue=30, cfg=CFG)
    centres = sorted(((x0 + x1) // 2, (y0 + y1) // 2) for x0, y0, x1, y1 in boxes)
    assert len(boxes) == 3
    assert centres[0][0] == pytest.approx(100, abs=6) and centres[1][0] == pytest.approx(130, abs=6)


def test_detect_hold_polygons_simplifies_outline_and_keys_by_box(monkeypatch):
    dense = [{"x": 90 + i, "y": 75.0} for i in range(21)]  # many points along the top edge
    pts = dense + [{"x": 110.0, "y": 85.0}, {"x": 90.0, "y": 85.0}]

    class FakeClient:
        def __init__(self, api_url, api_key):
            pass

        def infer(self, image, model_id):
            return {
                "predictions": [
                    {"x": 100, "y": 80, "width": 20, "height": 10, "points": pts},
                    {"x": 30, "y": 30, "width": 10, "height": 10},  # no outline
                ]
            }

    import inference_sdk

    monkeypatch.setenv("ROBOFLOW_API_KEY", "k")
    monkeypatch.setattr(inference_sdk, "InferenceHTTPClient", FakeClient)
    holds._detect_cache.clear()
    img = blob_image([(100, 80, 5, 30)])
    polys = detect_hold_polygons(img, CFG)
    assert list(polys) == [(90, 75, 110, 85)]  # the box detect_holds returns; no outline, no entry
    assert polys[(90, 75, 110, 85)] == [(90, 75), (110, 75), (110, 85), (90, 85)]
    assert detect_holds(img, CFG) == [(90, 75, 110, 85), (25, 25, 35, 35)]


def test_detect_holds_reads_the_key_from_dot_env(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("ROBOFLOW_API_KEY=from-dotenv", encoding="utf-8")
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    monkeypatch.setattr(holds, "REPO_ROOT", tmp_path)
    seen = []

    class FakeClient:
        def __init__(self, api_url, api_key):
            seen.append(api_key)

        def infer(self, image, model_id):
            return {"predictions": []}

    import inference_sdk

    monkeypatch.setattr(inference_sdk, "InferenceHTTPClient", FakeClient)
    holds._detect_cache.clear()
    assert detect_holds(blob_image([(10, 10, 3, 30)]), CFG) == []
    assert seen == ["from-dotenv"]
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)  # do not leak into other tests


# ---- Colour matching in Lab (match_colour) ---------------------------------------------------

LAB_CFG = {**load_holds_config(), **CFG}  # the Lab settings come from the real config


@pytest.mark.parametrize(
    ("lab1", "lab2", "expected"),
    [  # reference pairs from Sharma, Wu & Dalal (2005), the CIEDE2000 test data
        ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
        ((50.0, 0.0, 0.0), (50.0, -1.0, 2.0), 2.3669),
        ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0), 27.1492),
        ((63.0109, -31.0961, -5.8663), (62.8187, -29.7946, -4.0864), 1.2630),
        ((2.0776, 0.0795, -1.1350), (0.9033, -0.0636, -0.5514), 0.9082),
    ],
)
def test_ciede2000_matches_the_published_reference_values(lab1, lab2, expected):
    got = holds.ciede2000(np.array(lab1), np.array([lab2]))[0]
    assert got == pytest.approx(expected, abs=1e-4)


def test_ciede2000_with_a_higher_lightness_weight_counts_shade_for_less():
    ref, darker = np.array([60.0, 0.0, 50.0]), np.array([[45.0, 0.0, 50.0]])
    assert holds.ciede2000(ref, darker, k_l=2.0)[0] < holds.ciede2000(ref, darker)[0] / 1.8


def route_wall():
    """A beige wall with a yellow route (one hold in shadow, one chalky) among the colours hue
    matching confused with it: orange-yellow, pale beige-yellow; and a red next to a pink."""
    img = np.full((240, 400, 3), (180, 200, 215), np.uint8)  # beige wall (BGR)
    named_colours = {
        "yellow": (40, 60, bgr(28, 210, 235)),
        "yellow shadow": (100, 60, bgr(28, 210, 185)),  # ~20% darker (far darker turns olive)
        "yellow chalky": (160, 60, bgr(28, 150, 240)),
        "orange-yellow": (220, 60, bgr(19, 220, 235)),
        "pale beige": (280, 60, bgr(25, 55, 225)),
        "red": (60, 170, bgr(178, 220, 200)),
        "red shadow": (140, 170, bgr(178, 220, 130)),
        "pink": (220, 170, bgr(165, 150, 230)),
        "orange": (300, 170, bgr(10, 230, 235)),
    }
    boxes = {}
    for name, (cx, cy, colour) in named_colours.items():
        cv2.circle(img, (cx, cy), 22, colour, -1)
        boxes[name] = (cx - 22, cy - 22, cx + 23, cy + 23)
    return img, boxes


@pytest.mark.parametrize(
    ("clicked", "route"),
    [
        ("yellow", {"yellow", "yellow shadow", "yellow chalky"}),
        ("yellow shadow", {"yellow", "yellow shadow", "yellow chalky"}),
        ("red", {"red", "red shadow"}),
    ],
)
def test_match_colour_keeps_the_route_through_shade_and_drops_near_colours(clicked, route):
    img, named = route_wall()
    ref = holds.hold_lab(img, named[clicked], LAB_CFG)
    kept = holds.match_colour(img, list(named.values()), ref, LAB_CFG, ref_box=named[clicked])
    assert {name for name, box in named.items() if box in kept} == route


def test_hue_matching_confused_the_yellow_route_with_orange_yellow():
    """Why match_colour exists: hue alone keeps the orange-yellow hold."""
    img, named = route_wall()
    kept = filter_by_colour(img, list(named.values()), 28.0, LAB_CFG)
    assert named["orange-yellow"] in kept


def test_hold_lab_reads_the_hold_inside_its_outline_not_the_wall_in_its_box():
    img, named = route_wall()
    x0, y0, x1, y1 = named["yellow"]
    outline = [(x0 + 6, y0 + 6), (x1 - 6, y0 + 6), (x1 - 6, y1 - 6), (x0 + 6, y1 - 6)]
    measured = holds.hold_lab(img, named["yellow"], LAB_CFG, outline)
    painted = holds._lab(np.uint8([[bgr(28, 210, 235)]]))[0]
    assert holds.ciede2000(painted, np.array([measured]))[0] < 1.0
    assert holds.hold_lab(img, (0, 0, 0, 0), LAB_CFG) is None  # an empty region


def test_patch_lab_reads_a_click_on_a_hold_the_detector_missed():
    img, named = route_wall()
    x0, y0, x1, y1 = named["red"]
    clicked = holds.patch_lab(img, ((x0 + x1) // 2, (y0 + y1) // 2), LAB_CFG)
    whole = holds.hold_lab(img, named["red"], LAB_CFG)
    assert holds.ciede2000(clicked, np.array([whole]))[0] < 1.0
