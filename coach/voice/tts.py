"""Text to speech: synth(text) -> 24 kHz mono float32, trimmed, loudness-normalised, disk-cached.

Engines (config voice.tts): kokoro (default, local), elevenlabs (offline phrase-bank builder
only, needs ELEVENLABS_API_KEY), say (macOS fallback).

Persona YAML may carry a `voice:` key and/or a `tts:` block; all keys optional, `tts:` wins:
    voice: af_heart  (or voice: {kokoro_voice: af_heart, speed: 1.0})
    tts: {kokoro_voice: af_heart, speed: 1.0, elevenlabs_voice_id: ..., say_voice: Samantha}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml
from dotenv import load_dotenv

log = logging.getLogger(__name__)

SAMPLE_RATE = 24_000
CACHE_DIR = Path("data/audio_cache")
CONFIG_PATH = Path("config/default.yaml")
BENCH_LINES = (
    "Feet up, you're dangling.",
    "Nice, keep that hip in.",
    "Breathe. Look for the next hold.",
    "Straighten your arms and rest.",
    "That was close, try the left foot higher.",
)
SAY_BASE_WPM = 175
WARMUP_TEXT = "Ready."
ELEVENLABS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice}?output_format=pcm_24000"


@dataclass(frozen=True)
class TTSParams:
    """Post-processing and engine settings. Override via config/default.yaml `voice:` keys."""

    silence_db: float = -45.0  # samples below this (rel. full scale) count as silence
    pad_ms: float = 30.0  # keep this much around the trimmed speech
    target_rms_db: float = -20.0
    peak_limit: float = 0.95
    kokoro_voice: str = "af_heart"
    kokoro_lang: str = "a"
    speed: float = 1.0
    elevenlabs_voice_id: str = "JBFqnCBsd6RMkjVDRZzb"
    elevenlabs_model: str = "eleven_flash_v2_5"
    say_voice: str | None = None
    cache_dir: Path = field(default=CACHE_DIR)

    @classmethod
    def from_config(cls, cfg: dict | None = None, persona: dict | None = None) -> TTSParams:
        """Build from the `voice` config section (`tts_params` key) and a persona's settings."""
        persona = persona or {}
        voice = persona.get("voice") or {}
        if isinstance(voice, str):
            voice = {"kokoro_voice": voice}
        voice = {k: v for k, v in voice.items() if k != "tone"}  # `tone` is for the LLM prompt
        merged: dict[str, Any] = {}
        for src in ((cfg or {}).get("tts_params") or {}, voice, persona.get("tts") or {}):
            merged.update(src)
        known = {k: v for k, v in merged.items() if k in cls.__dataclass_fields__}
        if unknown := sorted(set(merged) - set(known)):
            log.warning("ignoring unknown TTS settings: %s", unknown)
        if "cache_dir" in known:
            known["cache_dir"] = Path(known["cache_dir"])
        return cls(**known)


def trim_silence(
    audio: np.ndarray, rate: int = SAMPLE_RATE, silence_db: float = -45.0, pad_ms: float = 30.0
) -> np.ndarray:
    """Cut leading/trailing samples quieter than `silence_db`, keeping `pad_ms` of padding."""
    if audio.size == 0:
        return audio
    loud = np.flatnonzero(np.abs(audio) > 10 ** (silence_db / 20))
    if loud.size == 0:
        return audio[:0]
    pad = int(rate * pad_ms / 1000)
    return audio[max(0, loud[0] - pad) : loud[-1] + 1 + pad]


def normalise_loudness(
    audio: np.ndarray, target_rms_db: float = -20.0, peak_limit: float = 0.95
) -> np.ndarray:
    """Scale to a target RMS, never letting the peak exceed `peak_limit`."""
    if audio.size == 0:
        return audio
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    if rms < 1e-9:
        return audio
    gain = 10 ** (target_rms_db / 20) / rms
    peak = float(np.max(np.abs(audio))) * gain
    if peak > peak_limit:
        gain *= peak_limit / peak
    return (audio * gain).astype(np.float32)


def postprocess(audio: np.ndarray, p: TTSParams, rate: int = SAMPLE_RATE) -> np.ndarray:
    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    trimmed = trim_silence(mono, rate, p.silence_db, p.pad_ms)
    return normalise_loudness(trimmed, p.target_rms_db, p.peak_limit)


class Engine(Protocol):
    name: str

    def voice_id(self) -> str: ...
    def speed(self) -> float: ...
    def raw(self, text: str) -> np.ndarray:
        """Unprocessed mono float32 at SAMPLE_RATE."""
        ...


class KokoroTTS:
    name = "kokoro"

    def __init__(self, params: TTSParams, pipeline: Any = None) -> None:
        self.p = params
        self._pipeline = pipeline  # injectable for tests

    def voice_id(self) -> str:
        return self.p.kokoro_voice

    def speed(self) -> float:
        return self.p.speed

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            from kokoro import KPipeline

            try:
                self._pipeline = KPipeline(lang_code=self.p.kokoro_lang)
            except SystemExit as exc:
                # misaki downloads spaCy's en_core_web_sm the first time if it isn't installed,
                # and spaCy calls sys.exit() when that download fails. SystemExit isn't an
                # Exception, so it would get past every handler in Voice and stop the event loop,
                # taking the dashboard and the debrief with it. Left unset, the next call retries.
                raise RuntimeError(
                    f"Kokoro failed to load (exit status {exc.code}), most likely because "
                    "spaCy's en_core_web_sm is missing and could not be downloaded"
                ) from exc
        return self._pipeline

    def warmup(self) -> None:
        """Load the model and voice pack and run the first inference, the slow one-off steps."""
        self.raw(WARMUP_TEXT)

    def raw(self, text: str) -> np.ndarray:
        gen = self._get_pipeline()(text, voice=self.p.kokoro_voice, speed=self.p.speed)
        chunks = []
        for _gs, _ps, audio in gen:
            if audio is None:
                continue
            if hasattr(audio, "detach"):  # torch tensor
                audio = audio.detach().cpu().numpy()
            chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


class ElevenLabsTTS:
    """Flash model over the REST API, raw 24 kHz PCM. Offline phrase-bank builds only."""

    name = "elevenlabs"

    def __init__(self, params: TTSParams, api_key: str | None = None) -> None:
        self.p = params
        self._key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not self._key:
            raise RuntimeError("ELEVENLABS_API_KEY is not set")

    def voice_id(self) -> str:
        return f"{self.p.elevenlabs_voice_id}:{self.p.elevenlabs_model}"

    def speed(self) -> float:
        return self.p.speed

    def raw(self, text: str) -> np.ndarray:
        body = {
            "text": text,
            "model_id": self.p.elevenlabs_model,
            "voice_settings": {"speed": self.p.speed},
        }
        req = urllib.request.Request(
            ELEVENLABS_URL.format(voice=self.p.elevenlabs_voice_id),
            data=json.dumps(body).encode(),
            headers={"xi-api-key": self._key, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - fixed https URL
            pcm = resp.read()
        return np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0


class SayTTS:
    """macOS `say` to a temp AIFF, then load."""

    name = "say"

    def __init__(self, params: TTSParams) -> None:
        if sys.platform != "darwin" or shutil.which("say") is None:
            raise RuntimeError("macOS `say` is not available")
        self.p = params

    def voice_id(self) -> str:
        return self.p.say_voice or "default"

    def speed(self) -> float:
        return self.p.speed

    def raw(self, text: str) -> np.ndarray:
        from coach.voice.speaker import load_wav

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "say.aiff"
            cmd = ["say", "-r", str(round(SAY_BASE_WPM * self.p.speed)), "-o", str(out)]
            if self.p.say_voice:
                cmd += ["-v", self.p.say_voice]
            subprocess.run([*cmd, "--", text], check=True, capture_output=True, timeout=30)
            return load_wav(out, SAMPLE_RATE)


class TTS:
    """synth(text) with post-processing and a disk cache keyed by (engine, voice, speed, text)."""

    def __init__(self, engine: Engine, params: TTSParams, use_cache: bool = True) -> None:
        self.engine = engine
        self.p = params
        self.use_cache = use_cache

    def cache_path(self, text: str) -> Path:
        key = json.dumps([self.engine.name, self.engine.voice_id(), self.engine.speed(), text])
        return self.p.cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()[:32]}.npy"

    def warmup(self) -> None:
        """Pay the engine's one-off start-up cost now, if it has one."""
        if hasattr(self.engine, "warmup"):
            self.engine.warmup()

    def cached(self, text: str) -> np.ndarray | None:
        """The clip if it is already on disk; never synthesises, so it is safe on the fast path."""
        path = self.cache_path(text)
        if not self.use_cache or not path.exists():
            return None
        try:
            return np.load(path)
        except (OSError, ValueError, EOFError):  # EOFError: an empty file, e.g. a part-copied cache
            log.warning("corrupt cache entry %s, resynthesising", path)
            return None

    def synth(self, text: str) -> np.ndarray:
        audio = self.cached(text)
        if audio is not None:
            return audio
        audio = postprocess(self.engine.raw(text), self.p)
        if self.use_cache:
            path = self.cache_path(text)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp.npy")
            np.save(tmp, audio)
            tmp.replace(path)
        return audio


ENGINES: dict[str, Callable[[TTSParams], Engine]] = {
    "kokoro": KokoroTTS,
    "elevenlabs": ElevenLabsTTS,
    "say": SayTTS,
}


def make_tts(
    cfg: dict | None = None,
    persona: dict | None = None,
    engine: str | None = None,
    use_cache: bool = True,
) -> TTS:
    """Build a TTS from the `voice` config section (key `tts`) and the persona dict."""
    name = engine or (cfg or {}).get("tts", "kokoro")
    if name not in ENGINES:
        raise ValueError(f"unknown tts engine {name!r}; choose from {sorted(ENGINES)}")
    params = TTSParams.from_config(cfg, persona)
    return TTS(ENGINES[name](params), params, use_cache)


def bench(params: TTSParams, lines: tuple[str, ...] = BENCH_LINES) -> dict[str, list[float]]:
    """Time-to-audio per line for each available engine (cache bypassed)."""
    results: dict[str, list[float]] = {}
    for name, factory in ENGINES.items():
        try:
            engine = factory(params)
            if hasattr(engine, "warmup"):
                t0 = time.perf_counter()
                engine.warmup()
                print(f"{name}: model load {time.perf_counter() - t0:.2f}s", flush=True)
            tts = TTS(engine, params, use_cache=False)
            times = []
            for line in lines:
                t0 = time.perf_counter()
                audio = tts.synth(line)
                dt = time.perf_counter() - t0
                times.append(dt)
                secs = audio.size / SAMPLE_RATE
                print(f"{name:10s} {dt * 1000:7.0f} ms  {secs:5.2f}s audio  {line}", flush=True)
            results[name] = times
        except Exception as exc:  # engine unavailable: report and move on
            print(f"{name}: skipped ({exc})", flush=True)
    for name, times in results.items():
        print(
            f"{name}: mean {sum(times) / len(times) * 1000:.0f} ms, max {max(times) * 1000:.0f} ms"
        )
    return results


def with_overrides(persona: dict, voice: str | None, speed: float | None) -> dict:
    """A copy of `persona` with one-off settings applied; the persona file is left alone."""
    tts = dict(persona.get("tts") or {})
    if voice:
        tts |= {"kokoro_voice": voice, "kokoro_lang": voice[0]}  # am_* -> a, bm_* -> b
    if speed is not None:
        tts["speed"] = speed
    return {**persona, "tts": tts}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench", action="store_true", help="time five lines on each available engine")
    ap.add_argument("--engine", help="engine for a one-off synth (default: config voice.tts)")
    ap.add_argument("--voice", help="try another Kokoro voice, e.g. bm_george")
    ap.add_argument("--speed", type=float, help="try another speed, e.g. 0.9")
    ap.add_argument("--text", default="Feet up, you're dangling.")
    args = ap.parse_args()
    load_dotenv()
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["voice"]
    persona = yaml.safe_load(Path(cfg["persona"]).read_text(encoding="utf-8"))
    persona = with_overrides(persona, args.voice, args.speed)
    if args.bench:
        bench(TTSParams.from_config(cfg, persona))
        return
    tts = make_tts(cfg, persona, engine=args.engine)
    audio = tts.synth(args.text)
    log.info("synthesised %.2fs of audio", audio.size / SAMPLE_RATE)
    from coach.voice.speaker import Speaker

    speaker = Speaker()
    speaker.play(audio)
    time.sleep(audio.size / SAMPLE_RATE + 0.3)
    speaker.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
