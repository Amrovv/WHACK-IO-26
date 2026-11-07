"""The real Voice: picks a persona line, stitches it from pre-rendered clips and plays it.

say() never waits for synthesis. If a clip isn't on disk it returns False at once and a worker
renders the clip, playing the line only if it is ready by the deadline and nothing has replaced
it. So True always means audio started; the `spoken` topic is the full record of what was heard.

debrief() plays a filler, then streams sentences from the LLM (or a template) back to back. Any
say() or stop() cuts it: the stream, pending synthesis and queued audio all go.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

from coach.bus import Bus
from coach.contracts import AttemptSummary, CoachEvent, HoldMap, SpeakRequest, SpokenLine
from coach.voice.debrief import (
    DebriefLLM,
    debrief_payload,
    make_debrief_llm,
    stream_debrief,
    template_sentences,
)
from coach.voice.phrasebank import (  # noqa: F401  fallback_keys is re-exported for tests
    NEUTRAL,
    LinePicker,
    fallback_keys,
    slot_values,
    stitch,
)
from coach.voice.speaker import Speaker, concat
from coach.voice.tts import CONFIG_PATH, TTS, make_tts

log = logging.getLogger(__name__)

BANK_PATH = Path("data/phrasebank.json")
DEMO_DEADLINE_S = 5.0
SPEAKER_POLL_S = 0.02  # how often a queued debrief sentence checks the speaker has gone quiet
NO_DEADLINE = float("inf")


class Voice:
    def __init__(
        self,
        bank: dict,
        tts: TTS,
        speaker: Speaker,
        bus: Bus,
        fillers: Sequence[str] = (),
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
        pool: Executor | None = None,
        debrief_llm: DebriefLLM | None = None,
    ) -> None:
        self._tts = tts
        self._speaker = speaker
        self._bus = bus
        self._fillers = list(fillers)
        self._clock = clock
        # One worker: the TTS engine isn't thread-safe, and late jobs bail out quickly.
        self._pool = pool or ThreadPoolExecutor(max_workers=1, thread_name_prefix="voice-tts")
        self._lines = LinePicker(bank, rng)
        self._decks = self._lines.decks
        self._holdmaps: asyncio.Queue[HoldMap] = bus.subscribe("holdmap")
        self._route_colour: str | None = None
        self._lock = threading.Lock()  # makes "still current?" and starting playback one step
        self._generation = 0  # bumped by every say/stop/debrief; older pending lines are dropped
        self._debrief_llm = debrief_llm  # None: template debriefs only
        self._debrief: asyncio.Task | None = None  # the running debrief, guarded by _lock

    def say(self, req: SpeakRequest) -> bool:
        if self._clock() > req.deadline:
            return False
        while not self._holdmaps.empty():
            self._route_colour = self._holdmaps.get_nowait().colour
        values = slot_values(req.event.data, self._route_colour)
        line = self._lines.pick(req.event.kind, req.level, req.mood, values)
        if line is None:
            log.warning("no usable line for %s L%d %s", req.event.kind, req.level, req.mood)
            return False
        clips, text = stitch(line, values)
        generation = self._supersede()
        parts = [self._tts.cached(clip) for clip in clips]
        if any(part is None for part in parts):
            self._pool.submit(self._render_and_play, req, clips, text, generation)
            return False
        return self._play(req, parts, text, generation)

    def is_speaking(self) -> bool:
        """True for a whole debrief, gaps between sentences included, so a cue can't cut it."""
        return self._debrief is not None or self._speaker.is_speaking()

    def stop(self) -> None:
        with self._lock:
            self._next_generation()
            self._speaker.stop()

    async def debrief(
        self, history: list[AttemptSummary], habits: dict, crux_hold: int | None
    ) -> None:
        """Speak a debrief, returning when it ends or say()/stop() cuts it. Never raises."""
        payload = debrief_payload(history, habits, crux_hold)
        with self._lock:
            generation = self._next_generation()
            task = self._debrief = asyncio.create_task(self._run_debrief(payload, generation))
        try:
            await task
        except asyncio.CancelledError:
            if asyncio.current_task().cancelling():  # our caller gave up, not say() or stop()
                with self._lock:
                    if self._debrief is task:
                        self._next_generation()
                raise
            log.info("debrief cut")
        except Exception:
            log.exception("debrief failed")
        finally:
            with self._lock:
                if self._debrief is task:
                    self._debrief = None

    def warm_up(self) -> None:
        """Load the TTS engine on the worker, so the first on-demand line can make its deadline."""
        self._pool.submit(self._warm_up)

    def close(self) -> None:
        self.stop()
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._speaker.close()

    def _warm_up(self) -> None:
        try:
            self._tts.warmup()
        except Exception:  # cached lines still work without the engine
            log.exception("TTS warm-up failed")

    def _supersede(self) -> int:
        with self._lock:
            return self._next_generation()

    def _next_generation(self) -> int:
        """Invalidate everything pending and cut any debrief. The caller holds the lock."""
        self._generation += 1
        if self._debrief is not None:
            task, self._debrief = self._debrief, None
            self._speaker.stop()  # the line replacing it may still need rendering
            try:  # say() and stop() can come from any thread
                task.get_loop().call_soon_threadsafe(task.cancel)
            except RuntimeError:  # its loop has closed, so the task is gone anyway
                log.debug("debrief loop already closed")
        return self._generation

    async def _run_debrief(self, payload: dict, generation: int) -> None:
        speech: asyncio.Queue[str | None] = asyncio.Queue()
        if self._fillers:
            speech.put_nowait(self._lines.draw(("filler",), self._fillers, lambda _line: True))
        async with asyncio.TaskGroup() as group:
            group.create_task(self._speak_queue(speech, generation))
            await asyncio.sleep(0)  # let the filler start before the LLM call can race a cut
            if not await self._stream_llm(payload, speech):
                for sentence in template_sentences(payload):
                    speech.put_nowait(sentence)
            speech.put_nowait(None)

    async def _stream_llm(self, payload: dict, speech: asyncio.Queue[str | None]) -> int:
        if self._debrief_llm is None or not payload["attempts"]:
            return 0
        return await stream_debrief(self._debrief_llm, payload, speech.put_nowait)

    async def _speak_queue(self, speech: asyncio.Queue[str | None], generation: int) -> None:
        """Play sentences in order, after whatever is already playing, such as the fall line.

        Each is rendered while the one before it plays. Cached clips skip the worker, so the
        filler can't wait behind warm-up.
        """
        loop = asyncio.get_running_loop()
        while (sentence := await speech.get()) is not None:
            try:
                audio = self._tts.cached(sentence)
                if audio is None:
                    audio = await loop.run_in_executor(self._pool, self._tts.synth, sentence)
            except Exception:  # one lost sentence shouldn't lose the rest
                log.exception("debrief synthesis failed for %r", sentence)
                continue
            await self._until_quiet()
            if self._start(audio, NO_DEADLINE, generation) is None:
                return
            self._bus.publish("debrief_text", sentence)
        await self._until_quiet()

    async def _until_quiet(self) -> None:
        while self._speaker.is_speaking():
            await asyncio.sleep(SPEAKER_POLL_S)

    def _render_and_play(
        self, req: SpeakRequest, clips: list[str], text: str, generation: int
    ) -> None:
        try:
            parts = []
            for clip in clips:
                if generation != self._generation or self._clock() > req.deadline:
                    log.info("dropped %r: late or replaced before it was rendered", text)
                    return
                parts.append(self._tts.synth(clip))  # cached on disk even if we end up too late
            self._play(req, parts, text, generation)
        except Exception:  # nothing above this thread would report it
            log.exception("on-demand synthesis failed for %r", text)

    def _play(
        self, req: SpeakRequest, parts: Sequence[np.ndarray], text: str, generation: int
    ) -> bool:
        t_audio_start = self._start(concat(parts), req.deadline, generation)
        if t_audio_start is None:
            return False
        event = req.event
        self._bus.publish("spoken", SpokenLine(event.t, t_audio_start, event.kind, text))
        return True

    def _start(self, audio: np.ndarray, deadline: float, generation: int) -> float | None:
        """Start playback and return the audio start time, or None if it's late or replaced."""
        with self._lock:
            late = self._clock() + self._speaker.latency > deadline
            if generation != self._generation or late:
                return None
            return self._speaker.play(audio)


def make_voice(cfg: dict, bus: Bus, no_llm: bool = False, bank: dict | None = None) -> Voice:
    """Build a Voice from the full config, as run.py does. The phrase bank must already be built.

    Debriefs use the template if `no_llm` is set or GEMINI_API_KEY is missing.
    """
    load_dotenv()
    voice_cfg = cfg["voice"]
    persona = yaml.safe_load(Path(voice_cfg["persona"]).read_text(encoding="utf-8"))
    if bank is None:
        bank = json.loads(BANK_PATH.read_text(encoding="utf-8"))
    speaker = Speaker(device=voice_cfg.get("output_device"))
    voice = Voice(
        bank,
        make_tts(voice_cfg, persona),
        speaker,
        bus,
        fillers=persona["fillers"],
        debrief_llm=None if no_llm else make_debrief_llm(voice_cfg, persona),
    )
    voice.warm_up()
    return voice


def main() -> None:
    ap = argparse.ArgumentParser(description="Speak one line through the real Voice.")
    ap.add_argument("--kind", default="feet_low")
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--mood", default=NEUTRAL)
    ap.add_argument("--line", help="say this line instead of using data/phrasebank.json")
    ap.add_argument("--debrief", action="store_true", help="speak a sample debrief instead")
    ap.add_argument("--no-llm", action="store_true", help="use the template debrief")
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    bank = {args.kind: {str(args.level): {args.mood: [args.line]}}} if args.line else None
    bus = Bus()
    spoken = bus.subscribe("spoken")
    voice = make_voice(cfg, bus, no_llm=args.no_llm, bank=bank)

    if args.debrief:
        history = [
            AttemptSummary(1, 21.0, 4, "fall", [(4, 6.5)], ["feet_low"], []),
            AttemptSummary(2, 34.0, 6, "fall", [(6, 8.0)], ["feet_low"], ["feet_low"]),
        ]
        asyncio.run(
            voice.debrief(history, {"feet_low": {"attempts": 2, "total": 3}, "mood": args.mood}, 6)
        )
        voice.close()
        return

    now = time.monotonic()
    data = {"limb": "LH", "hold_id": 6, "dir": "up and left", "colour": "pink"}
    event = CoachEvent(t=now, kind=args.kind, priority=1, ttl_s=DEMO_DEADLINE_S, data=data)
    req = SpeakRequest(event, args.level, args.mood, now + DEMO_DEADLINE_S)
    log.info("say() returned %s after %.1f ms", voice.say(req), (time.monotonic() - now) * 1000)
    while spoken.empty() and time.monotonic() < req.deadline:
        time.sleep(0.02)
    while voice.is_speaking():
        time.sleep(0.02)
    if spoken.empty():
        log.warning("nothing was spoken")
    else:
        line = spoken.get_nowait()
        log.info("%r started %.1f ms after the event", line.text, (line.t_audio_start - now) * 1000)
    voice.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
