from pathlib import Path

import numpy as np
import pytest

from coach.voice.tts import (
    SAMPLE_RATE,
    TTS,
    ElevenLabsTTS,
    KokoroTTS,
    TTSParams,
    make_tts,
    normalise_loudness,
    trim_silence,
)


def _tone(seconds: float, amp: float = 0.1) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_trim_silence_removes_edges() -> None:
    sil = np.zeros(SAMPLE_RATE, dtype=np.float32)
    out = trim_silence(np.concatenate([sil, _tone(0.5), sil]), pad_ms=0)
    assert abs(out.size - SAMPLE_RATE // 2) < 5


def test_trim_all_silence_is_empty() -> None:
    assert trim_silence(np.zeros(1000, dtype=np.float32)).size == 0


def test_normalise_hits_target_and_limits_peak() -> None:
    out = normalise_loudness(_tone(0.5, 0.01), target_rms_db=-20)
    assert np.sqrt(np.mean(out**2)) == pytest.approx(0.1, rel=0.02)
    spiky = np.zeros(1000, dtype=np.float32)
    spiky[500] = 0.5
    assert np.max(np.abs(normalise_loudness(spiky, peak_limit=0.9))) <= 0.9 + 1e-6


class FakePipeline:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, text: str, voice: str, speed: float):
        self.calls += 1
        yield (
            "g",
            "p",
            np.concatenate([np.zeros(2400, np.float32), _tone(0.3), np.zeros(2400, np.float32)]),
        )


def test_kokoro_synth_processes_and_caches(tmp_path: Path) -> None:
    params = TTSParams(cache_dir=tmp_path)
    pipe = FakePipeline()
    tts = TTS(KokoroTTS(params, pipeline=pipe), params)
    a = tts.synth("hello")
    b = tts.synth("hello")
    assert pipe.calls == 1
    assert a.dtype == np.float32 and a.ndim == 1
    assert np.array_equal(a, b)
    assert a.size < SAMPLE_RATE * 0.3 + 3000  # silence trimmed


def test_cache_key_depends_on_voice_speed_text(tmp_path: Path) -> None:
    def path(**kw) -> Path:
        p = TTSParams(cache_dir=tmp_path, **kw)
        return TTS(KokoroTTS(p, pipeline=FakePipeline()), p).cache_path("hi")

    assert path() == path()
    assert path() != path(speed=1.2)
    assert path() != path(kokoro_voice="bm_george")


def test_params_from_persona() -> None:
    p = TTSParams.from_config({}, {"tts": {"kokoro_voice": "bm_lewis", "speed": 1.1, "bogus": 1}})
    assert (p.kokoro_voice, p.speed) == ("bm_lewis", 1.1)


def test_make_tts_rejects_unknown_engine() -> None:
    with pytest.raises(ValueError):
        make_tts({"tts": "nope"})


def test_elevenlabs_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        ElevenLabsTTS(TTSParams())


def test_params_from_persona_voice_key() -> None:
    assert TTSParams.from_config({}, {"voice": "bm_george"}).kokoro_voice == "bm_george"
    p = TTSParams.from_config({}, {"voice": {"kokoro_voice": "bm_lewis", "speed": 0.9}})
    assert (p.kokoro_voice, p.speed) == ("bm_lewis", 0.9)


def test_persona_tts_block_beats_voice_key() -> None:
    persona = {"voice": {"speed": 0.9}, "tts": {"speed": 1.1}}
    assert TTSParams.from_config({}, persona).speed == 1.1


def test_unknown_persona_setting_warns(caplog: pytest.LogCaptureFixture) -> None:
    TTSParams.from_config({}, {"voice": {"nme": "bm_george"}})
    assert "nme" in caplog.text
