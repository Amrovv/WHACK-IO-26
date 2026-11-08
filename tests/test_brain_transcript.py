"""D's readable transcript of the live brain (coach/stubs.py Transcript, run.py --transcript)."""

import asyncio
from pathlib import Path

from coach import stubs
from coach.bus import Bus
from coach.contracts import (
    AttemptSummary,
    ClimberState,
    CoachEvent,
    Limb,
    SpokenLine,
    Suppressed,
)
from run import parse_args


def state(t, person=True, **contacts):
    holds = {Limb[k]: v for k, v in contacts.items()}
    return ClimberState(t, 0, person, {}, {}, holds, (0.5, 0.5))


async def run_transcript(path: Path, items, clock=lambda: 105.0) -> list[str]:
    """Publish (topic, payload) items, or ("hello", dict) to a Transcript; its lines."""
    bus = Bus()
    bus.bind_loop()
    transcript = stubs.Transcript(bus, path, clock=clock)
    task = asyncio.create_task(transcript.run())
    await asyncio.sleep(0)
    for topic, payload in items:
        if topic == "hello":
            transcript.on_hello(payload)
        else:
            bus.publish(topic, payload)
        await asyncio.sleep(0.005)  # through the bus before the next (seconds apart in real life)
    await asyncio.sleep(0.01)
    task.cancel()
    transcript.close()
    return path.read_text(encoding="utf-8").splitlines()


async def test_every_decision_of_the_live_brain_is_one_readable_line(tmp_path):
    feet = CoachEvent(t=102.5, kind="feet_low", priority=1, ttl_s=1.5, data={"limb": "LF"})
    stall = CoachEvent(t=103.0, kind="stalling", priority=1, ttl_s=1.5)
    lines = await run_transcript(
        tmp_path / "t.txt",
        [
            ("hello", {"session": 1, "spec": "file:videos/climb_2.mp4", "problem_id": "climb_2"}),
            ("state", state(100.0, person=False)),
            ("state", state(101.0, LH=None)),
            ("state", state(102.0, LH=3)),
            ("state", state(102.1, LH=3)),  # nothing changed: no line
            ("event", feet),
            ("suppressed", Suppressed(event=stall, reason="cooldown")),
            ("spoken", SpokenLine(102.5, 102.56, "feet_low", "Feet up, find hold 3.")),
            ("attempt", AttemptSummary(1, 4.2, 7, "fall", [(5, 3.1)], ["feet_low"], [])),
            ("debrief_text", "You fell at seven."),
        ],
    )
    assert lines == [
        "",
        "=== session 1: file:videos/climb_2.mp4 (problem climb_2) ===",
        "    0.00s  PERCEPTION  person lost",
        "    1.00s  PERCEPTION  person found",
        "    2.00s  PERCEPTION  left_hand: none -> 3",
        '    2.50s  EVENT       feet_low             p1  {"limb": "LF"}',
        "    3.00s  SUPPRESSED  stalling             cooldown",
        '    2.56s  SPOKEN      feet_low             "Feet up, find hold 3."  '
        "(60 ms after the event)",
        "    5.00s  ATTEMPT     #1 fall, high hold 7, 4.2 s; stalls: hold 5 3.1 s; "
        "cues given: feet_low; followed: none",
        '    5.00s  DEBRIEF     "You fell at seven."',
    ]


async def test_a_new_source_starts_a_block_and_a_looped_replay_isnt_written_twice(tmp_path):
    hello = {"session": 1, "spec": "file:a.mp4", "problem_id": "p"}
    event = CoachEvent(t=11.0, kind="attempt_start", priority=0, ttl_s=1.0)
    lines = await run_transcript(
        tmp_path / "t.txt",
        [
            ("hello", hello),
            ("state", state(10.0)),
            ("event", event),
            ("hello", {**hello, "pass": 2}),
            ("event", event),  # the replayed pass: already written
            ("hello", {**hello, "pass": 3}),
            ("hello", {"session": 2, "spec": "browser", "problem_id": "p"}),
            ("state", state(50.0)),
        ],
    )
    assert [line for line in lines if "EVENT" in line] == [
        "    1.00s  EVENT       attempt_start        p0"
    ]
    assert lines.count("--- the clip looped: later passes replay this one ---") == 1
    assert "=== session 2: browser (problem p) ===" in lines
    assert lines[-1] == "    0.00s  PERCEPTION  person found"  # the new session's own clock


def test_the_transcript_flag_takes_an_optional_path(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"")
    base = ["--source", f"file:{clip}", "--problem", "p"]
    assert parse_args(base).transcript is None
    assert parse_args([*base, "--transcript"]).transcript is True  # the default path
    assert parse_args([*base, "--transcript", "out.txt"]).transcript == Path("out.txt")
