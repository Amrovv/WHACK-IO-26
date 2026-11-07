from coach.contracts import ClimberState, Limb
from coach.perception.review import KEY_LEFT, KEY_RIGHT, Reviewer, contact_text, key_name


def test_key_names():
    assert key_name(ord(" ")) == "toggle"
    assert key_name(ord("d")) == key_name(ord(".")) == key_name(KEY_RIGHT) == "next"
    assert key_name(ord("a")) == key_name(ord(",")) == key_name(KEY_LEFT) == "prev"
    assert key_name(ord("w")) == "fwd" and key_name(ord("s")) == "back"
    assert key_name(ord("q")) == key_name(27) == "quit"
    assert key_name(-1) is None and key_name(ord("z")) is None


def test_playing_advances_and_pauses_on_the_last_frame():
    rv = Reviewer(n_frames=3, fps=30)
    rv.tick()
    rv.tick()
    assert rv.idx == 2 and not rv.paused
    rv.tick()  # already on the last frame: stop there instead of running off the end
    assert rv.idx == 2 and rv.paused


def test_space_toggles_pause_and_a_paused_viewer_does_not_advance():
    rv = Reviewer(10, 30)
    rv.on_key("toggle")
    assert rv.paused
    rv.tick()
    assert rv.idx == 0
    rv.on_key("toggle")
    rv.tick()
    assert rv.idx == 1


def test_stepping_pauses_and_is_clamped_at_both_ends():
    rv = Reviewer(100, 30)
    rv.on_key("prev")
    assert rv.idx == 0 and rv.paused
    rv.on_key("next")
    rv.on_key("next")
    assert rv.idx == 2
    rv.on_key("fwd")
    assert rv.idx == 32
    rv.on_key("back")
    rv.on_key("back")
    assert rv.idx == 0
    rv.seek(10_000)
    assert rv.idx == 99


def test_quit_and_ignored_keys():
    rv = Reviewer(5, 30)
    rv.on_key(None)
    assert not rv.done and not rv.paused
    rv.on_key("quit")
    assert rv.done


def test_contact_text_lists_limbs_with_dashes_for_none():
    s = ClimberState(
        0.0, 0, True, {}, {}, {Limb.LH: 5, Limb.RH: None, Limb.LF: 2, Limb.RF: 1}, (0.5, 0.5)
    )
    assert contact_text(s) == "LH:5  RH:-  LF:2  RF:1"
