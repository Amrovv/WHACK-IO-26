import cv2
import numpy as np
import pytest

from coach.perception.holdmap_editor import KEY_ENTER, KEY_ESC, Editor, EditorState
from coach.perception.holds import (
    load_hold_shapes,
    load_holdmap,
    load_holdmap_with_shapes,
)

CFG = {
    "detector_model": "fake/1",
    "api_url": "https://example.invalid",
    "hue_tolerance": 12,
    "min_saturation": 60,
    "patch_radius_px": 4,
    "morph_kernel_px": 5,
    "min_area": 0.0003,
    "max_area": 0.02,
    "split_peak_ratio": 0.75,
    "start_count": 2,
    "drag_min_px": 6,
}
W, H = 300, 200


def bgr(hue):
    return tuple(
        int(v) for v in cv2.cvtColor(np.uint8([[[hue, 200, 220]]]), cv2.COLOR_HSV2BGR)[0, 0]
    )


@pytest.fixture
def image():
    img = np.full((H, W, 3), (120, 120, 120), np.uint8)
    for cx, cy in ((60, 150), (80, 100), (100, 50)):  # route: three yellow holds, low to high
        cv2.circle(img, (cx, cy), 10, bgr(30), -1)
    cv2.circle(img, (200, 100), 10, bgr(110), -1)  # a blue hold
    return img


@pytest.fixture
def editor(image, tmp_path):
    return Editor(EditorState(image, CFG), "demo", root=tmp_path)


def click(ed, x, y):
    ed.on_mouse(cv2.EVENT_LBUTTONDOWN, x, y)
    ed.on_mouse(cv2.EVENT_LBUTTONUP, x, y)


def drag(ed, p0, p1):
    ed.on_mouse(cv2.EVENT_LBUTTONDOWN, *p0)
    ed.on_mouse(cv2.EVENT_MOUSEMOVE, *p1)
    ed.on_mouse(cv2.EVENT_LBUTTONUP, *p1)


def key(ed, ch):
    ed.on_key(ord(ch) if isinstance(ch, str) else ch)


def test_first_click_picks_the_route_colour_and_finds_the_holds(editor):
    assert editor.state.included() == []
    click(editor, 60, 150)
    assert len(editor.state.included()) == 3  # the blue hold is a different colour
    assert editor.state.colour_label() == "yellow"


def test_click_toggles_a_box_out_and_back_in(editor):
    click(editor, 60, 150)
    click(editor, 80, 100)
    assert len(editor.state.included()) == 2
    click(editor, 80, 100)
    assert len(editor.state.included()) == 3


def test_drag_on_empty_wall_draws_a_missing_hold_and_renumbers(editor):
    click(editor, 60, 150)
    drag(editor, (150, 20), (175, 40))  # a new hold above the others
    hm = editor.state.build("demo")
    assert len(hm.holds) == 4 and hm.mode == "manual"
    top = max(hm.holds, key=lambda h: h.id)
    assert top.center[1] == pytest.approx(30 / H)  # highest on the wall is the last id


def test_clicking_a_drawn_box_removes_it(editor):
    click(editor, 60, 150)
    drag(editor, (150, 20), (175, 40))
    click(editor, 160, 30)
    assert len(editor.state.included()) == 3 and editor.state.manual == []


def test_short_drag_counts_as_a_click_and_drag_before_colour_is_ignored(editor):
    drag(editor, (10, 10), (60, 60))  # no colour picked yet: ignored
    assert editor.state.manual == [] and editor.state.ref_hue is None
    click(editor, 60, 150)
    ed = editor
    ed.on_mouse(cv2.EVENT_LBUTTONDOWN, 80, 100)
    ed.on_mouse(cv2.EVENT_LBUTTONUP, 83, 102)  # 3 px: a click, toggles the box out
    assert len(ed.state.included()) == 2


def test_s_and_t_then_click_mark_start_and_top(editor):
    click(editor, 60, 150)
    key(editor, "s")
    click(editor, 80, 100)  # make the middle hold the (only) start
    key(editor, "t")
    click(editor, 100, 50)
    hm = editor.state.build("demo")
    roles = {h.id: h.role for h in hm.holds}
    assert roles == {1: "mid", 2: "start", 3: "top"}


def test_marking_start_twice_unmarks_it_and_a_hold_cannot_be_start_and_top(editor):
    click(editor, 60, 150)
    key(editor, "s")
    click(editor, 60, 150)
    key(editor, "t")
    click(editor, 60, 150)
    assert editor.state.start == set() and len(editor.state.top) == 1


def test_c_repicks_the_colour(editor):
    click(editor, 60, 150)
    key(editor, "c")
    assert editor.state.ref_hue is None
    click(editor, 200, 100)  # now pick the blue hold
    assert len(editor.state.included()) == 1 and editor.state.colour_label() == "blue"


def test_m_falls_back_to_colour_mode_when_the_detector_is_unavailable(editor, monkeypatch):
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    click(editor, 60, 150)
    key(editor, "m")  # detector mode needs the API key
    assert editor.state.mode == "colour" and "detector unavailable" in editor.state.message
    assert len(editor.state.included()) == 3


def test_enter_saves_a_loadable_holdmap_and_esc_does_not(editor, tmp_path):
    key(editor, KEY_ENTER)  # nothing to save yet
    assert not editor.done and editor.saved is None
    click(editor, 60, 150)
    key(editor, KEY_ENTER)
    assert editor.done and editor.saved == tmp_path / "demo.json"
    hm = load_holdmap("demo", tmp_path)
    assert [h.id for h in hm.holds] == [1, 2, 3] and hm.colour == "yellow" and hm.mode == "colour"


def test_esc_quits_without_saving(editor, tmp_path):
    click(editor, 60, 150)
    key(editor, KEY_ESC)
    assert editor.done and editor.saved is None and not list(tmp_path.iterdir())


def test_render_adds_a_help_bar_and_draws_without_error(editor, image):
    out = editor.render()
    assert out.shape[0] > image.shape[0] and out.shape[1] == image.shape[1]
    click(editor, 60, 150)
    editor.on_mouse(cv2.EVENT_LBUTTONDOWN, 10, 10)
    editor.on_mouse(cv2.EVENT_MOUSEMOVE, 50, 50)  # a rubber-band box mid-drag
    assert editor.render().shape == out.shape


def right_click(ed, x, y):
    ed.on_mouse(cv2.EVENT_RBUTTONDOWN, x, y)


def test_right_click_deletes_a_detected_hold_for_good(editor):
    click(editor, 60, 150)
    right_click(editor, 80, 100)
    assert len(editor.state.included()) == 2
    click(editor, 80, 100)  # nothing is there any more: a click can't bring it back
    assert len(editor.state.included()) == 2 and editor.state.box_at((80, 100)) is None
    hm = editor.state.build("demo")
    assert [h.id for h in hm.holds] == [1, 2]  # renumbered, no gap


def test_right_click_also_deletes_a_hold_that_was_toggled_off_and_a_drawn_one(editor):
    click(editor, 60, 150)
    click(editor, 80, 100)  # toggled off: dim outline still shown
    right_click(editor, 80, 100)  # now deleted outright
    assert editor.state.box_at((80, 100)) is None
    drag(editor, (150, 20), (175, 40))
    right_click(editor, 160, 30)
    assert editor.state.manual == [] and len(editor.state.included()) == 2


def test_right_click_before_a_colour_is_picked_or_on_empty_wall_does_nothing(editor):
    right_click(editor, 60, 150)
    assert editor.state.ref_hue is None
    click(editor, 60, 150)
    right_click(editor, 250, 20)
    assert len(editor.state.included()) == 3 and "no box" in editor.state.message


def test_u_undoes_deletes_toggles_drawn_boxes_and_marks_in_reverse_order(editor):
    click(editor, 60, 150)
    right_click(editor, 80, 100)
    drag(editor, (150, 20), (175, 40))
    key(editor, "s")
    click(editor, 60, 150)
    assert len(editor.state.included()) == 3 and editor.state.start
    key(editor, "u")  # undo the start mark
    assert not editor.state.start
    key(editor, "u")  # undo the drawn box
    assert editor.state.manual == [] and len(editor.state.included()) == 2
    key(editor, "u")  # undo the delete: the hold is back
    assert len(editor.state.included()) == 3
    key(editor, "u")
    assert "nothing to undo" in editor.state.message


def test_changing_colour_resets_undo_history(editor):
    click(editor, 60, 150)
    right_click(editor, 80, 100)
    key(editor, "c")
    click(editor, 200, 100)  # re-pick: automatic boxes changed
    key(editor, "u")
    assert "nothing to undo" in editor.state.message


# ---- manual mode (white holds) and polygon holds ---------------------------------------------


@pytest.fixture
def white_wall():
    """Grey wall with white holds: no saturation, so the colour tool can not find them."""
    img = np.full((H, W, 3), (120, 120, 120), np.uint8)
    cv2.circle(img, (60, 150), 10, (240, 240, 240), -1)
    cv2.circle(img, (200, 100), 10, (240, 240, 240), -1)
    return img


@pytest.fixture
def white_editor(white_wall, tmp_path):
    return Editor(EditorState(white_wall, CFG, colour="white"), "white-route", root=tmp_path)


def type_corners(ed, corners):
    for x, y in corners:
        click(ed, x, y)


def test_a_click_on_a_white_hold_explains_that_n_starts_manual_mode(white_editor):
    click(white_editor, 60, 150)
    assert not white_editor.state.ready
    assert "press n" in white_editor.state.message


def test_n_allows_drawing_holds_without_a_colour(white_editor, tmp_path):
    drag(white_editor, (50, 140), (70, 160))  # ignored: no colour and not in manual mode yet
    assert white_editor.state.manual == []
    key(white_editor, "n")
    assert white_editor.state.ready and white_editor.state.ref_hue is None
    drag(white_editor, (50, 140), (70, 160))
    drag(white_editor, (190, 90), (210, 110))
    assert len(white_editor.state.included()) == 2
    right_click(white_editor, 60, 150)
    assert len(white_editor.state.included()) == 1
    key(white_editor, KEY_ENTER)
    hm = load_holdmap("white-route", tmp_path)
    assert hm.colour == "white" and hm.mode == "manual" and len(hm.holds) == 1


def test_p_needs_a_colour_or_manual_mode_first(white_editor):
    key(white_editor, "p")
    assert white_editor.poly is None and "press n" in white_editor.state.message


SLAB = [(100, 100), (130, 100), (230, 190), (200, 190)]  # a slanted slab (pixels)
SLAB_B = [(160, 100), (190, 100), (290, 190), (260, 190)]  # parallel: bounding boxes overlap


def test_polygon_mode_adds_a_diagonal_hold_with_a_bbox_and_saves_its_shape(white_editor, tmp_path):
    key(white_editor, "n")
    key(white_editor, "p")
    assert white_editor.poly == []
    type_corners(white_editor, SLAB)
    assert len(white_editor.poly) == 4
    key(white_editor, KEY_ENTER)  # closes the polygon; it does NOT save or quit
    assert white_editor.poly is None and not white_editor.done and white_editor.saved is None
    hm, shapes = white_editor.state.build_with_shapes("white-route")
    assert len(hm.holds) == 1 and len(shapes[1]) == 4
    assert hm.holds[0].bbox == pytest.approx((100 / W, 100 / H, 230 / W, 190 / H))
    key(white_editor, KEY_ENTER)  # now it saves
    assert white_editor.saved is not None
    back, back_shapes = load_holdmap_with_shapes("white-route", tmp_path)
    assert back == hm and len(back_shapes[1]) == 4
    assert load_hold_shapes("white-route", tmp_path)[1][0] == pytest.approx((100 / W, 100 / H))


def test_esc_cancels_a_polygon_without_quitting_and_p_also_closes_it(white_editor):
    key(white_editor, "n")
    key(white_editor, "p")
    type_corners(white_editor, SLAB[:3])
    key(white_editor, KEY_ESC)
    assert white_editor.poly is None and not white_editor.done
    assert white_editor.state.manual == []
    key(white_editor, "p")
    type_corners(white_editor, SLAB)
    key(white_editor, "p")  # p again closes it, like Enter
    assert white_editor.poly is None and len(white_editor.state.manual) == 1


def test_right_click_removes_the_last_corner_and_then_leaves_polygon_mode(white_editor):
    key(white_editor, "n")
    key(white_editor, "p")
    type_corners(white_editor, SLAB[:3])
    right_click(white_editor, 0, 0)
    assert white_editor.poly == SLAB[:2]
    right_click(white_editor, 0, 0)
    right_click(white_editor, 0, 0)
    assert white_editor.poly == []
    right_click(white_editor, 0, 0)  # none left: leave polygon mode
    assert white_editor.poly is None


def test_a_polygon_needs_three_corners_that_are_not_in_a_line(white_editor):
    key(white_editor, "n")
    key(white_editor, "p")
    type_corners(white_editor, SLAB[:2])
    key(white_editor, KEY_ENTER)  # only 2 corners: stays in polygon mode
    assert white_editor.poly is not None and "at least 3" in white_editor.state.message
    type_corners(white_editor, [(160, 100)])  # a third corner on the same horizontal line
    key(white_editor, KEY_ENTER)
    assert white_editor.state.manual == [] and "not in a line" in white_editor.state.message
    assert white_editor.poly is not None  # still drawing: fix a corner and try again


def test_clicks_pick_the_hold_the_point_is_on_not_the_one_whose_box_is_smaller(white_editor):
    key(white_editor, "n")
    for slab in (SLAB, SLAB_B):
        key(white_editor, "p")
        type_corners(white_editor, slab)
        key(white_editor, KEY_ENTER)
    st = white_editor.state
    on_a = (205, 180)  # on slab A's body; inside slab B's bounding box (x 160-290, y 100-190)
    assert st.box_at(on_a) is not None
    owner = st.box_at(on_a)
    assert st.polygons[owner] == SLAB
    assert st.box_at((115, 180)) is None  # empty corner of A's box, nothing there
    right_click(white_editor, *on_a)  # delete slab A only
    assert len(st.included()) == 1 and st.polygons[st.manual[0]][0] == (160, 100)


def test_undo_restores_a_deleted_polygon_with_its_shape(white_editor):
    key(white_editor, "n")
    key(white_editor, "p")
    type_corners(white_editor, SLAB)
    key(white_editor, KEY_ENTER)
    right_click(white_editor, 205, 180)
    assert white_editor.state.included() == []
    key(white_editor, "u")
    assert len(white_editor.state.included()) == 1 and len(white_editor.state.polygons) == 1
    key(white_editor, "u")  # undo the polygon itself
    assert white_editor.state.included() == [] and white_editor.state.polygons == {}


def test_s_and_t_marks_work_on_polygon_holds(white_editor):
    key(white_editor, "n")
    for slab in (SLAB, SLAB_B):
        key(white_editor, "p")
        type_corners(white_editor, slab)
        key(white_editor, KEY_ENTER)
    key(white_editor, "s")
    click(white_editor, 205, 180)  # slab A
    hm = white_editor.state.build("white-route")
    roles = {
        tuple(round(v * k) for v, k in zip(h.bbox, (W, H, W, H), strict=True)): h.role
        for h in hm.holds
    }
    assert roles[(100, 100, 230, 190)] == "start"


def test_render_draws_polygons_and_the_polygon_in_progress(white_editor, white_wall):
    key(white_editor, "n")
    key(white_editor, "p")
    type_corners(white_editor, SLAB)
    key(white_editor, KEY_ENTER)
    key(white_editor, "p")
    type_corners(white_editor, SLAB_B[:3])  # in progress
    out = white_editor.render()
    h, w = white_wall.shape[:2]
    assert out.shape[1] == w and out.shape[0] > h  # the help bar is added below
    drawn = (out[:h] != white_wall).any(axis=2)
    # slab A's slanted left edge runs (100,100) -> (200,190): (150,145) is on it
    assert drawn[143:148, 148:153].any()
    # a corner of slab A's bounding box, far from the slab, stays untouched wall
    assert not drawn[185:189, 103:108].any()


def test_edit_loads_a_saved_map_with_polygons_and_roles_and_saves_it_back(white_editor, tmp_path):
    key(white_editor, "n")
    for slab in (SLAB, SLAB_B):
        key(white_editor, "p")
        type_corners(white_editor, slab)
        key(white_editor, KEY_ENTER)
    drag(white_editor, (10, 10), (40, 40))  # a plain box too
    key(white_editor, "t")
    click(white_editor, 275, 180)  # mark slab B as top
    key(white_editor, KEY_ENTER)
    first, first_shapes = load_holdmap_with_shapes("white-route", tmp_path)

    state = EditorState(white_editor.state.image.copy(), CFG)
    state.load_existing(first, first_shapes)
    again = Editor(state, "white-route2", root=tmp_path)
    assert len(state.included()) == 3 and len(state.polygons) == 2
    key(again, KEY_ENTER)
    second, second_shapes = load_holdmap_with_shapes("white-route2", tmp_path)
    flat = lambda m: [v for h in m.holds for v in h.bbox]  # noqa: E731
    assert flat(second) == pytest.approx(flat(first))
    assert [h.role for h in second.holds] == [h.role for h in first.holds]
    assert set(second_shapes) == set(first_shapes)
    assert second.colour == "white"
