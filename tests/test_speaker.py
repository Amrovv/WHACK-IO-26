import sys
from types import SimpleNamespace

import numpy as np
import pytest

from coach.voice.speaker import Speaker


class FakeStream:
    latency = 0.0

    def __init__(self, **_kwargs) -> None:
        pass

    def start(self) -> None:
        pass


def pull(speaker: Speaker, frames: int) -> np.ndarray:
    out = np.zeros((frames, 1), dtype=np.float32)
    speaker._callback(out, frames, None, None)
    return out[:, 0]


def test_second_stop_does_not_restart_fade(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(OutputStream=FakeStream))
    speaker = Speaker(rate=1000, fade_ms=20)
    speaker.play(np.ones(100, dtype=np.float32))
    pull(speaker, 10)
    speaker.stop()
    first_half = pull(speaker, 10)
    speaker.stop()
    second_half = pull(speaker, 20)
    assert second_half[0] < first_half[-1]
    assert not speaker.is_speaking()
