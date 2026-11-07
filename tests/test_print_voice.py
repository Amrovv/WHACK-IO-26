import numpy as np

from coach.contracts import CoachEvent, SpeakRequest
from coach.voice.print_voice import PrintVoice
from coach.voice.speaker import concat, resample


class FakeClock:
    def __init__(self) -> None:
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def req(clock: FakeClock, text="one two three four", deadline_in=1.0) -> SpeakRequest:
    ev = CoachEvent(t=clock.t, kind="feet_low", priority=1, ttl_s=1.2, data={"text": text})
    return SpeakRequest(ev, 2, "impressed", clock.t + deadline_in)


def test_speaking_duration_scales_with_words(capsys):
    clock = FakeClock()
    v = PrintVoice(clock=clock)
    assert not v.is_speaking()
    assert v.say(req(clock))
    assert "[SPEAK] feet_low L2 impressed: one two three four" in capsys.readouterr().out
    clock.t += 1.39
    assert v.is_speaking()
    clock.t += 0.02  # 4 words * 0.35 = 1.4 s
    assert not v.is_speaking()


def test_late_request_rejected(capsys):
    clock = FakeClock()
    v = PrintVoice(clock=clock)
    r = req(clock, deadline_in=-0.1)
    assert not v.say(r)
    assert capsys.readouterr().out == ""
    assert not v.is_speaking()


def test_stop_ends_speaking():
    clock = FakeClock()
    v = PrintVoice(clock=clock)
    v.say(req(clock))
    v.stop()
    assert not v.is_speaking()


def test_concat_and_resample():
    a = np.ones(10, dtype=np.float32)
    out = concat([a, a], gap_ms=1, rate=24000)
    assert out.size == 10 + 24 + 10
    assert resample(np.ones(48), 48000, 24000).size == 24


# Bus and phrase bank


def test_say_publishes_spoken_on_the_bus(capsys):
    from coach.bus import Bus

    clock = FakeClock()
    bus = Bus()
    spoken = bus.subscribe("spoken")
    v = PrintVoice(clock=clock, bus=bus)
    r = req(clock)
    assert v.say(r)
    line = spoken.get_nowait()
    assert (line.t_event, line.t_audio_start, line.kind, line.text) == (
        r.event.t,
        clock.t,
        "feet_low",
        "one two three four",
    )


def test_late_request_publishes_nothing():
    from coach.bus import Bus

    clock = FakeClock()
    bus = Bus()
    spoken = bus.subscribe("spoken")
    assert not PrintVoice(clock=clock, bus=bus).say(req(clock, deadline_in=-0.1))
    assert spoken.empty()


def test_bank_lines_are_stitched_and_prefer_direction(capsys):
    clock = FakeClock()
    bank = {"hand_searching": {"2": {"neutral": ["Grab the {hold}.", "Reach {dir}."]}}}
    v = PrintVoice(clock=clock, bank=bank)
    data = {"limb": "LH", "hold_id": 7, "dir": "up and left"}
    for _ in range(3):
        ev = CoachEvent(t=clock.t, kind="hand_searching", priority=1, ttl_s=1.2, data=data)
        assert v.say(SpeakRequest(ev, 2, "neutral", clock.t + 1))
    assert capsys.readouterr().out.count("Reach up and left.") == 3


def test_debrief_publishes_template_sentences(capsys):
    import asyncio

    from coach.bus import Bus
    from coach.contracts import AttemptSummary

    bus = Bus()
    texts = bus.subscribe("debrief_text")
    v = PrintVoice(clock=FakeClock(), bus=bus)
    history = [AttemptSummary(1, 21.0, 4, "fall", [(4, 6.5)], ["feet_low"], [])]
    asyncio.run(v.debrief(history, {}, 4))
    assert not texts.empty()
    assert "[DEBRIEF]" in capsys.readouterr().out
