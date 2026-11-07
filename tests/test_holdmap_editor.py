import cv2
import numpy as np
import pytest

from coach.perception.holdmap_editor import KEY_ENTER, KEY_ESC, Editor, EditorState
from coach.perception.holds import load_holdmap

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
