"""Non-blocking audio output: one sounddevice OutputStream, 24 kHz mono float32."""

from __future__ import annotations

import argparse
import logging
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 24_000
FADE_MS = 20
DEFAULT_GAP_MS = 80


def resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear-interpolation resample of a 1-D float array."""
    if src_rate == dst_rate or samples.size == 0:
        return samples.astype(np.float32, copy=False)
    n_out = max(1, round(samples.size * dst_rate / src_rate))
    x_old = np.linspace(0.0, 1.0, samples.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(x_new, x_old, samples).astype(np.float32)


def load_wav(path: str | Path, rate: int = SAMPLE_RATE) -> np.ndarray:
    """Read an audio file as mono float32 at `rate`."""
    import soundfile as sf

    data, src_rate = sf.read(str(path), dtype="float32", always_2d=True)
    return resample(data.mean(axis=1), int(src_rate), rate)


def concat(
    parts: Sequence[np.ndarray], gap_ms: float = DEFAULT_GAP_MS, rate: int = SAMPLE_RATE
) -> np.ndarray:
    """Join clips with `gap_ms` of silence between them."""
    if not parts:
        return np.zeros(0, dtype=np.float32)
    gap = np.zeros(int(rate * gap_ms / 1000), dtype=np.float32)
    out: list[np.ndarray] = []
    for i, part in enumerate(parts):
        if i:
            out.append(gap)
        out.append(np.asarray(part, dtype=np.float32))
    return np.concatenate(out)


class Speaker:
    """Plays clips through a persistent output stream (silence when idle)."""

    def __init__(
        self,
        device: str | int | None = None,
        on_finished: Callable[[], None] | None = None,
        rate: int = SAMPLE_RATE,
        fade_ms: float = FADE_MS,
    ) -> None:
        import sounddevice as sd

        self.rate = rate
        self.on_finished = on_finished
        self._fade_n = max(1, int(rate * fade_ms / 1000))
        self._lock = threading.Lock()
        self._buf = np.zeros(0, dtype=np.float32)
        self._pos = 0
        self._fade_left = 0  # >0 while fading out for stop()
        self._fade_total = 0
        self._stream = sd.OutputStream(
            samplerate=rate,
            channels=1,
            dtype="float32",
            device=device,
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        finished = False
        with self._lock:
            out = np.zeros(frames, dtype=np.float32)
            n = min(frames, self._buf.size - self._pos)
            if n > 0:
                chunk = self._buf[self._pos : self._pos + n].copy()
                if self._fade_left > 0:
                    k = min(n, self._fade_left)
                    start = self._fade_total - self._fade_left
                    ramp = 1.0 - (np.arange(start, start + k) + 1) / self._fade_total
                    chunk[:k] *= ramp.astype(np.float32)
                    chunk[k:] = 0.0
                    self._fade_left -= k
                out[:n] = chunk
                self._pos += n
                if self._fade_left == 0 and self._fade_total:
                    self._buf = np.zeros(0, dtype=np.float32)
                    self._pos = 0
                    self._fade_total = 0
                    finished = True
            if self._pos >= self._buf.size and self._buf.size:
                self._buf = np.zeros(0, dtype=np.float32)
                self._pos = 0
                finished = True
            outdata[:, 0] = out
        if finished and self.on_finished:
            try:
                self.on_finished()
            except Exception:
                log.exception("on_finished callback failed")

    def play(self, samples: np.ndarray) -> float:
        """Queue `samples` (replacing any current clip). Returns estimated monotonic start time."""
        data = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
        with self._lock:
            self._buf = data
            self._pos = 0
            self._fade_left = 0
            self._fade_total = 0
        # Audio reaches the DAC after the next callback plus the stream's output latency.
        return time.monotonic() + float(self._stream.latency)

    def is_speaking(self) -> bool:
        with self._lock:
            return self._pos < self._buf.size

    def stop(self) -> None:
        """Fade out over ~20 ms, then clear the buffer."""
        with self._lock:
            remaining = self._buf.size - self._pos
            if remaining <= 0 or self._fade_left > 0:  # nothing playing, or already fading
                return
            self._fade_total = self._fade_left = min(self._fade_n, remaining)
            # Truncate so playback ends exactly when the fade does.
            self._buf = self._buf[: self._pos + self._fade_total]

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a wav file and print start latency.")
    parser.add_argument("--wav", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    samples = load_wav(args.wav)
    speaker = Speaker(device=args.device)
    t0 = time.monotonic()
    t_start = speaker.play(samples)
    log.info("start latency: %.1f ms", (t_start - t0) * 1000)
    while speaker.is_speaking():
        time.sleep(0.02)
    time.sleep(0.2)
    speaker.close()


if __name__ == "__main__":
    main()
