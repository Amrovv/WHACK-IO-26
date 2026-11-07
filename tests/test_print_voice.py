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
