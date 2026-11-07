"""Latency report from a bus log, plus a naive "ask a vision model" baseline for the demo slide.

Owned by D.

  uv run python scripts/bench_latency.py data/logs/bus-<problem>-<stamp>.jsonl
  uv run python scripts/bench_latency.py --vlm-baseline frame.jpg [--model gemini-...]
  (both at once works too)

Bus log lines are {"t", "topic", "payload"}, where "t" is the monotonic time the item was
published (or written, for run.py's log). For each spoken cue, matched to its CoachEvent by
(capture time, kind):

  capture -> event         event published  - CoachEvent.t (frame capture time)
  event   -> audio start   SpokenLine.t_audio_start - event published
  total                    SpokenLine.t_audio_start - CoachEvent.t

Prints markdown tables to stdout; diagnostics go to the log on stderr.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

log = logging.getLogger("bench_latency")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "default.yaml"

VLM_CALLS = 5
VLM_PROMPT = (
    "This is a frame from a fixed camera watching someone bouldering. "
    "What should this climber do next? Answer in one short sentence."
)
LATENCY_TARGET_MS = 800  # from CLAUDE.md: cue audio within ~0.8 s
ANSWER_PREVIEW_CHARS = 70


# ---- Bus log analysis -------------------------------------------------------------------


@dataclass
class Cue:
    kind: str
    capture_to_event: float | None  # seconds; None if the event wasn't in the log
    event_to_audio: float | None
    total: float


@dataclass
class AttemptCounts:
    label: str
    events: int = 0
    spoken: int = 0
    suppressed: int = 0
    outcome: str = ""
    suppressed_reasons: dict[str, int] = field(default_factory=lambda: defaultdict(int))


def read_log(path: Path) -> list[dict]:
    records = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            records.append({"t": float(rec["t"]), "topic": rec["topic"], "payload": rec["payload"]})
        except (ValueError, KeyError, TypeError) as e:
            log.warning("%s:%d skipped (%s)", path, n, e)
    records.sort(key=lambda r: r["t"])
    return records


def _key(t: float, kind: str) -> tuple[float, str]:
    return (round(t, 6), kind)  # capture times round-trip through JSON; match on 1 us


def spoken_cues(records: list[dict]) -> list[Cue]:
    """One Cue per spoken line, matched to the event that triggered it."""
    published: dict[tuple[float, str], float] = {}
    cues = []
    for r in records:
        p = r["payload"]
        if r["topic"] == "event" and isinstance(p, dict) and "t" in p:
            published.setdefault(_key(p["t"], p.get("kind", "")), r["t"])
        elif r["topic"] == "spoken" and isinstance(p, dict):
            t_event, t_audio = float(p["t_event"]), float(p["t_audio_start"])
            pub = published.get(_key(t_event, p.get("kind", "")))
            cues.append(
                Cue(
                    kind=p.get("kind", "?"),
                    capture_to_event=None if pub is None else pub - t_event,
                    event_to_audio=None if pub is None else t_audio - pub,
                    total=t_audio - t_event,
                )
            )
    return cues


def per_attempt(records: list[dict]) -> list[AttemptCounts]:
    """Events detected and lines spoken, split at each attempt_start."""
    current = AttemptCounts("before first attempt")
    rows = [current]
    for r in records:
        p = r["payload"]
        topic = r["topic"]
        if topic == "event" and isinstance(p, dict):
            if p.get("kind") == "attempt_start":
                current = AttemptCounts(str(len(rows)))
                rows.append(current)
                continue
            current.events += 1
        elif topic == "spoken":
            current.spoken += 1
        elif topic == "suppressed" and isinstance(p, dict):
            current.suppressed += 1
            current.suppressed_reasons[p.get("reason", "?")] += 1
        elif topic == "attempt" and isinstance(p, dict):
            if p.get("attempt_no") is not None:
                current.label = str(p["attempt_no"])
            current.outcome = p.get("outcome", "")
    return [a for a in rows if a.events or a.spoken or a.suppressed or a.outcome]


# ---- Stats and tables -------------------------------------------------------------------


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, q in 0-100."""
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def ms(seconds: float) -> str:
    return f"{round(seconds * 1000) + 0:.0f}"  # + 0 turns -0 into 0


def table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def latency_table(cues: list[Cue]) -> str:
    rows = []
    for name, values in (
        ("capture → event", [c.capture_to_event for c in cues if c.capture_to_event is not None]),
        ("event → audio start", [c.event_to_audio for c in cues if c.event_to_audio is not None]),
        ("**total**", [c.total for c in cues]),
    ):
        if not values:
            rows.append([name, "–", "–", "–", "0"])
            continue
        rows.append(
            [
                name,
                ms(statistics.median(values)),
                ms(percentile(values, 95)),
                ms(max(values)),
                str(len(values)),
            ]
        )
    return table(["Stage (ms)", "median", "p95", "max", "n"], rows)


def kind_table(cues: list[Cue]) -> str:
    by_kind: dict[str, list[float]] = defaultdict(list)
    for c in cues:
        by_kind[c.kind].append(c.total)
    rows = [
        [kind, str(len(v)), ms(statistics.median(v)), ms(max(v))]
        for kind, v in sorted(by_kind.items(), key=lambda kv: -len(kv[1]))
    ]
    return table(["Cue kind", "spoken", "median total (ms)", "max total (ms)"], rows)


def attempt_table(attempts: list[AttemptCounts]) -> str:
    rows = []
    for a in attempts:
        reasons = ", ".join(f"{r} {n}" for r, n in sorted(a.suppressed_reasons.items())) or "–"
        rate = f"{a.spoken / a.events:.0%}" if a.events else "–"
        rows.append([a.label, a.outcome or "–", str(a.events), str(a.spoken), rate, reasons])
    return table(
        ["Attempt", "outcome", "events", "spoken", "spoken/events", "not said (reason)"], rows
    )


def report(path: Path) -> str:
    records = read_log(path)
    cues = spoken_cues(records)
    unmatched = sum(c.capture_to_event is None for c in cues)
    span = records[-1]["t"] - records[0]["t"] if records else 0.0
    parts = [
        f"## Latency: `{path.name}`",
        "",
        f"{len(records)} bus items over {span:.0f} s, {len(cues)} spoken cues.",
    ]
    if not cues:
        parts += ["", "No spoken cues in this log."]
    else:
        total = [c.total for c in cues]
        late = sum(t * 1000 > LATENCY_TARGET_MS for t in total)
        parts += [
            f"Target: audio within {LATENCY_TARGET_MS} ms of capture. "
            f"{len(cues) - late} of {len(cues)} cues on target.",
            "",
            latency_table(cues),
            "",
            kind_table(cues),
        ]
        if unmatched:
            parts += [
                "",
                f"{unmatched} spoken line(s) had no matching event in the log (total only).",
            ]
        backwards = sum(c.event_to_audio is not None and c.event_to_audio < 0 for c in cues)
        if backwards:
            parts += [
                "",
                f"**Warning:** for {backwards} cue(s) the event was logged after its audio "
                "started. This log stamps items when they were written, not when they were "
                "published, so the capture → event / event → audio split is unreliable. "
                "The total is still right (both ends come from the payloads).",
            ]
    parts += ["", "### Per attempt", "", attempt_table(per_attempt(records))]
    return "\n".join(parts)


# ---- Vision-model baseline --------------------------------------------------------------


def default_model(config: Path) -> str | None:
    try:
        model = yaml.safe_load(config.read_text())["voice"]["llm_model"]
    except (OSError, KeyError, TypeError):
        return None
    return None if not model or "CURRENT" in model else model  # placeholder until kickoff


def vlm_baseline(frame: Path, model: str) -> str:
    """Time VLM_CALLS sequential Gemini vision calls on one frame (default sampling settings)."""
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types

    load_dotenv(ROOT / ".env")
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY is not set (put it in .env)")
    client = genai.Client(api_key=key)
    image = types.Part.from_bytes(data=frame.read_bytes(), mime_type="image/jpeg")

    rows, times = [], []
    for i in range(1, VLM_CALLS + 1):
        t0 = time.perf_counter()
        resp = client.models.generate_content(model=model, contents=[image, VLM_PROMPT])
        dt = time.perf_counter() - t0
        times.append(dt)
        answer = " ".join((resp.text or "").split())
        if len(answer) > ANSWER_PREVIEW_CHARS:
            answer = answer[: ANSWER_PREVIEW_CHARS - 1] + "…"
        rows.append([str(i), ms(dt), answer.replace("|", "/")])
        log.info("call %d: %.2f s", i, dt)

    summary = table(
        ["", "median", "p95", "max", "n"],
        [
            [
                "round trip (ms)",
                ms(statistics.median(times)),
                ms(percentile(times, 95)),
                ms(max(times)),
                str(len(times)),
            ]
        ],
    )
    return "\n".join(
        [
            f"## Naive baseline: `{model}` on `{frame.name}`",
            "",
            f'Prompt: "{VLM_PROMPT}"',
            "",
            "Round trip per call, before any speech. The local fast path never waits on this.",
            "",
            summary,
            "",
            table(["Call", "ms", "answer"], rows),
        ]
    )


# ---- CLI --------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Cue latency report and VLM baseline.")
    p.add_argument("bus_log", nargs="?", type=Path, help="bus log JSONL")
    p.add_argument(
        "--vlm-baseline", type=Path, metavar="FRAME.jpg", help="time Gemini on one frame"
    )
    p.add_argument("--model", help="Gemini model id (default: voice.llm_model in the config)")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = p.parse_args(argv)
    if not args.bus_log and not args.vlm_baseline:
        p.error("give a bus log, --vlm-baseline FRAME.jpg, or both")
    for path in (args.bus_log, args.vlm_baseline):
        if path and not path.exists():
            p.error(f"no such file: {path}")

    sections = []
    if args.bus_log:
        sections.append(report(args.bus_log))
    if args.vlm_baseline:
        model = args.model or default_model(args.config)
        if not model:
            p.error(
                "no Gemini model set: pass --model (config voice.llm_model is still a placeholder)"
            )
        sections.append(vlm_baseline(args.vlm_baseline, model))
    print("\n\n".join(sections))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
