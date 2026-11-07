import logging
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import coach.voice
from coach.bus import Bus
from coach.contracts import CoachEvent, HoldMap, Limb, SpeakRequest, SpokenLine
from coach.voice.phrasebank import render_texts, slot_values, stitch
from coach.voice.speaker import concat
from coach.voice.voice import Voice, fallback_keys, make_voice

LATENCY = 0.01
LEFT_HAND = {"limb": "LH"}


class FakeClock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


class FakeStream:
    latency = 0.0

    def __init__(self, **_kwargs) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


class FakeSpeaker:
    latency = LATENCY

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.played: list[np.ndarray] = []
        self.stops = 0
        self.speaking = False

    def play(self, samples: np.ndarray) -> float:
        self.played.append(samples)
        return self.clock.t + LATENCY

    def is_speaking(self) -> bool:
        return self.speaking

    def stop(self) -> None:
        self.stops += 1

    def close(self) -> None:
        pass


class FakeTTS:
    """Clips listed in `rendered` are on disk; anything else needs synth()."""

    def __init__(self, rendered: list[str]) -> None:
        self.disk = {text: self.clip(text) for text in rendered}
        self.synthesised: list[str] = []
        self.on_synth = lambda _text: None
        self.warmups = 0

    def warmup(self) -> None:
        self.warmups += 1

    @staticmethod
    def clip(text: str) -> np.ndarray:
        return np.full(4, sum(map(ord, text)), dtype=np.float32)

    def cached(self, text: str) -> np.ndarray | None:
        return self.disk.get(text)

    def synth(self, text: str) -> np.ndarray:
        if text not in self.disk:
            self.synthesised.append(text)
            self.on_synth(text)
            self.disk[text] = self.clip(text)
        return self.disk[text]


class Rig:
    def __init__(
        self,
        bank: dict,
        rendered: list[str] | None = None,
        fillers=(),
        debrief_llm=None,
        speaker_type=None,
    ) -> None:
        self.clock = FakeClock()
        self.speaker = (speaker_type or FakeSpeaker)(self.clock)
        self.tts = FakeTTS(render_texts(bank, [], []) if rendered is None else rendered)
        self.bus = Bus()
        self.spoken = self.bus.subscribe("spoken")
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.voice = Voice(
            bank,
            self.tts,
            self.speaker,
            self.bus,
            fillers=fillers,
            clock=self.clock,
            rng=random.Random(1),
            pool=self.pool,
            debrief_llm=debrief_llm,
        )

    def request(self, kind="feet_low", level=0, mood="neutral", data=None, deadline_in=1.0):
        event = CoachEvent(self.clock.t, kind, 1, 1.2, data or {})
        return SpeakRequest(event, level, mood, self.clock.t + deadline_in)

    def lines(self) -> list[SpokenLine]:
        out = []
        while not self.spoken.empty():
            out.append(self.spoken.get_nowait())
        return out

    def texts(self) -> list[str]:
        return [line.text for line in self.lines()]

    def finish_worker(self) -> None:
        self.pool.shutdown(wait=True)


def bank_of(lines: list[str], kind="feet_low", level=0, mood="neutral") -> dict:
    return {kind: {str(level): {mood: lines}}}


# Deadlines


def test_late_request_is_not_played() -> None:
    rig = Rig(bank_of(["Feet up."]))
    assert not rig.voice.say(rig.request(deadline_in=-0.01))
    assert rig.speaker.played == []
    assert rig.lines() == []


def test_audio_reaching_the_output_exactly_on_the_deadline_plays() -> None:
    rig = Rig(bank_of(["Feet up."]))
    assert rig.voice.say(rig.request(deadline_in=LATENCY))


def test_output_latency_counts_against_the_deadline() -> None:
    rig = Rig(bank_of(["Feet up."]))
    assert not rig.voice.say(rig.request(deadline_in=LATENCY / 2))
    assert rig.speaker.played == []


def test_late_request_does_not_use_up_a_line() -> None:
    rig = Rig(bank_of(["A.", "B."]))
    rig.voice.say(rig.request(deadline_in=-1.0))
    assert rig.voice._decks == {}


def test_spoken_line_is_published_with_the_audio_start_time() -> None:
    rig = Rig(bank_of(["Feet up."]))
    req = rig.request()
    rig.clock.t += 0.2  # the event was captured a while ago
    assert rig.voice.say(req)
    assert rig.lines() == [SpokenLine(req.event.t, rig.clock.t + LATENCY, "feet_low", "Feet up.")]


# Line choice


def test_no_immediate_repeats_across_many_reshuffles() -> None:
    rig = Rig(bank_of(["A.", "B.", "C."]))
    for _ in range(300):
        assert rig.voice.say(rig.request())
    said = rig.texts()
    assert all(a != b for a, b in zip(said, said[1:], strict=False))


def test_every_line_is_used_once_before_any_comes_back() -> None:
    lines = ["A.", "B.", "C.", "D.", "E."]
    rig = Rig(bank_of(lines))
    for _ in range(20):
        rig.voice.say(rig.request())
    said = rig.texts()
    for start in range(0, 20, 5):
        assert sorted(said[start : start + 5]) == lines


def test_single_line_key_can_repeat() -> None:
    rig = Rig(bank_of(["Feet up."]))
    assert rig.voice.say(rig.request())
    assert rig.voice.say(rig.request())


def test_keys_shuffle_independently() -> None:
    bank = {"feet_low": {"0": {"neutral": ["A.", "B."]}}, "fall": {"0": {"neutral": ["C."]}}}
    rig = Rig(bank)
    rig.voice.say(rig.request())
    rig.voice.say(rig.request(kind="fall"))
    rig.voice.say(rig.request())
    assert sorted(rig.texts()) == ["A.", "B.", "C."]


# Fallbacks


def test_fallback_order_is_lower_levels_then_neutral() -> None:
    assert fallback_keys("fall", 2, "impressed") == [
        ("fall", 2, "impressed"),
        ("fall", 1, "impressed"),
        ("fall", 0, "impressed"),
        ("fall", 2, "neutral"),
        ("fall", 1, "neutral"),
        ("fall", 0, "neutral"),
    ]
    assert fallback_keys("fall", 0, "neutral") == [("fall", 0, "neutral")]


def test_falls_back_to_a_lower_level_before_neutral() -> None:
    bank = {
        "fall": {
            "0": {"exasperated": ["Lower level."]},
            "2": {"neutral": ["Neutral mood."], "exasperated": []},
        }
    }
    rig = Rig(bank)
    assert rig.voice.say(rig.request("fall", level=2, mood="exasperated"))
    assert rig.texts() == ["Lower level."]


def test_falls_back_to_neutral_mood() -> None:
    rig = Rig(bank_of(["Neutral."], level=1))
    assert rig.voice.say(rig.request(level=2, mood="impressed"))
    assert rig.texts() == ["Neutral."]


def test_never_falls_back_to_a_higher_level() -> None:
    rig = Rig(bank_of(["Too specific."], level=2))
    assert not rig.voice.say(rig.request(level=1))


def test_unknown_key_returns_false(caplog: pytest.LogCaptureFixture) -> None:
    rig = Rig({})
    with caplog.at_level(logging.WARNING):
        assert not rig.voice.say(rig.request("top"))
    assert "no usable line" in caplog.text
    assert rig.speaker.played == []


def test_line_with_an_unfillable_slot_is_skipped() -> None:
    rig = Rig(bank_of(["{hold} again.", "Feet up."]), rendered=["Feet up."])
    for _ in range(5):
        assert rig.voice.say(rig.request())
    assert set(rig.texts()) == {"Feet up."}


def test_key_with_only_unfillable_lines_falls_back() -> None:
    bank = {"feet_low": {"0": {"neutral": ["Feet up."]}, "2": {"neutral": ["{hold} again."]}}}
    rig = Rig(bank, rendered=["Feet up."])
    assert rig.voice.say(rig.request(level=2))
    assert rig.texts() == ["Feet up."]


# Slot filling


def test_slots_are_stitched_from_fragments_in_spoken_order() -> None:
    line = "{dir}, {colour} one, {hold}."
    rendered = ["up and left", "pink", "one", "hold six"]
    rig = Rig(bank_of([line]), rendered=rendered)
    data = {"dir": "up and left", "colour": "pink", "hold_id": 6}
    assert rig.voice.say(rig.request(data=data))
    assert rig.texts() == ["up and left, pink one, hold six."]
    np.testing.assert_array_equal(
        rig.speaker.played[0], concat([FakeTTS.clip(text) for text in rendered])
    )
    assert rig.tts.synthesised == []


@pytest.mark.parametrize("limb", ["LH", "left_hand", Limb.LH])
def test_limb_slot_accepts_every_form_of_limb(limb: str) -> None:
    assert slot_values({"limb": limb}) == {"limb": "left hand"}


@pytest.mark.parametrize("data", [{"limb": "tail"}, {"hold_id": 99}, {"dir": None}, {"hold_id": 0}])
def test_unspeakable_values_are_left_out(data: dict) -> None:
    assert slot_values(data) == {}


def test_colour_comes_from_the_hold_map() -> None:
    rig = Rig(bank_of(["{colour} one."]), rendered=["pink", "one."])
    rig.bus.publish("holdmap", HoldMap("p1", "pink", [], (640, 480), "detector"))
    assert rig.voice.say(rig.request())
    assert rig.texts() == ["pink one."]


def test_a_new_hold_map_replaces_the_colour() -> None:
    rig = Rig(bank_of(["{colour} one."]), rendered=["pink", "blue", "one."])
    rig.bus.publish("holdmap", HoldMap("p1", "pink", [], (640, 480), "detector"))
    rig.bus.publish("holdmap", HoldMap("p2", "blue", [], (640, 480), "detector"))
    rig.voice.say(rig.request())
    assert rig.texts() == ["blue one."]


def test_event_colour_beats_the_hold_map() -> None:
    assert slot_values({"colour": "blue"}, "pink") == {"colour": "blue"}


@pytest.mark.parametrize("colour", ["unknown", "", None])
def test_unusable_route_colour_leaves_the_slot_empty(colour: str | None) -> None:
    assert slot_values({}, colour) == {}


def test_colour_lines_are_skipped_without_a_hold_map() -> None:
    rig = Rig(bank_of(["{colour} one.", "Feet up."]), rendered=["Feet up."])
    assert rig.voice.say(rig.request())
    assert rig.texts() == ["Feet up."]


def test_every_stitched_clip_is_one_the_builder_renders() -> None:
    lines = ["Feet up.", "{limb} higher.", "Move, {limb}, then {hold}.", "{dir}, {colour} one."]
    values = {"limb": "left hand", "hold": "hold six", "dir": "up", "colour": "pink"}
    rendered = set(render_texts(bank_of(lines), ["pink"], [])) | set(values.values())
    for line in lines:
        assert set(stitch(line, values)[0]) <= rendered


# On-demand synthesis


def test_say_does_not_wait_for_synthesis() -> None:
    rig = Rig(bank_of(["{hold}."]), rendered=[])
    release = threading.Event()
    rig.tts.on_synth = lambda _text: release.wait(5)
    assert not rig.voice.say(rig.request(data={"hold_id": 30}))  # returned with the worker stuck
    assert rig.speaker.played == []
    release.set()
    rig.finish_worker()
    assert len(rig.speaker.played) == 1
    assert rig.texts() == ["hold thirty."]


def test_only_the_missing_piece_is_synthesised() -> None:
    rig = Rig(bank_of(["{hold} again."]), rendered=["again."])
    rig.voice.say(rig.request(data={"hold_id": 30}))
    rig.finish_worker()
    assert rig.tts.synthesised == ["hold thirty"]
    assert rig.texts() == ["hold thirty again."]


def test_synthesis_finishing_after_the_deadline_is_not_played_but_is_kept() -> None:
    rig = Rig(bank_of(["{hold}."]), rendered=[])

    def slow(_text: str) -> None:
        rig.clock.t += 2.0

    rig.tts.on_synth = slow
    assert not rig.voice.say(rig.request(data={"hold_id": 30}))
    rig.finish_worker()
    assert rig.speaker.played == []
    assert rig.lines() == []
    assert "hold thirty" in rig.tts.disk


def test_remaining_pieces_are_skipped_once_the_deadline_has_passed() -> None:
    rig = Rig(bank_of(["{hold} and {dir}"]), rendered=[])

    def slow(_text: str) -> None:
        rig.clock.t += 2.0

    rig.tts.on_synth = slow
    rig.voice.say(rig.request(data={"hold_id": 30, "dir": "up"}))
    rig.finish_worker()
    assert rig.tts.synthesised == ["hold thirty"]


def test_stop_cancels_a_pending_line() -> None:
    rig = Rig(bank_of(["{hold}."]), rendered=[])
    release = threading.Event()
    rig.tts.on_synth = lambda _text: release.wait(5)
    rig.voice.say(rig.request(data={"hold_id": 30}))
    rig.voice.stop()
    release.set()
    rig.finish_worker()
    assert rig.speaker.played == []


def test_a_newer_line_replaces_a_pending_one() -> None:
    bank = {"feet_low": {"0": {"neutral": ["{hold}."]}}, "fall": {"0": {"neutral": ["Again."]}}}
    rig = Rig(bank, rendered=["Again."])
    release = threading.Event()
    rig.tts.on_synth = lambda _text: release.wait(5)
    rig.voice.say(rig.request(data={"hold_id": 30}))
    assert rig.voice.say(rig.request("fall"))
    release.set()
    rig.finish_worker()
    assert rig.texts() == ["Again."]


def test_failed_synthesis_is_logged_not_raised(caplog: pytest.LogCaptureFixture) -> None:
    rig = Rig(bank_of(["{hold}."]), rendered=[])

    def boom(_text: str) -> None:
        raise RuntimeError("engine down")

    rig.tts.on_synth = boom
    assert not rig.voice.say(rig.request(data={"hold_id": 30}))
    rig.finish_worker()
    assert "on-demand synthesis failed" in caplog.text
    assert rig.speaker.played == []


# Delegation and set-up


def test_is_speaking_and_stop_delegate_to_the_speaker() -> None:
    rig = Rig({})
    assert not rig.voice.is_speaking()
    rig.speaker.speaking = True
    assert rig.voice.is_speaking()
    rig.voice.stop()
    assert rig.speaker.stops == 1


def test_warm_up_runs_on_the_worker() -> None:
    rig = Rig({})
    rig.voice.warm_up()
    rig.finish_worker()
    assert rig.tts.warmups == 1


def test_failed_warm_up_is_logged_not_raised(caplog: pytest.LogCaptureFixture) -> None:
    rig = Rig({})

    def boom() -> None:
        raise RuntimeError("no model")

    rig.tts.warmup = boom
    rig.voice.warm_up()
    rig.finish_worker()
    assert "TTS warm-up failed" in caplog.text


def test_make_voice_matches_what_run_py_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(OutputStream=FakeStream))
    cfg = yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))
    built = coach.voice.make_voice
    assert built is make_voice
    warmed = []
    monkeypatch.setattr(Voice, "warm_up", lambda self: warmed.append(self))
    voice = built(cfg, Bus(), no_llm=True, bank={})
    assert warmed == [voice]
    voice.close()
