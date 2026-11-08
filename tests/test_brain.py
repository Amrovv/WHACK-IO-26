from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from coach.brain import Brain
from coach.bus import Bus
from coach.contracts import HoldMap, SpokenLine
from tests.test_pipeline import FakeVoice, _synthetic_climb, holdmap


def _config() -> dict:
    return yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "config" / "default.yaml").read_text(
            encoding="utf-8"
        )
    )


async def _run_frames(
    brain: Brain, bus: Bus, states: list, *, publish_holdmap: bool = True
) -> None:
    queue: asyncio.Queue = asyncio.Queue()
    task = asyncio.create_task(brain.run(queue))
    if publish_holdmap:
        bus.publish("holdmap", holdmap())
    for state in states:
        await queue.put(state)
    for _ in range(len(states) + 2):
        await asyncio.sleep(0)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_brain_publishes_events_and_finished_attempt() -> None:
    bus = Bus()
    event_queue = bus.subscribe("event")
    attempt_queue = bus.subscribe("attempt")
    brain = Brain(_config(), bus, FakeVoice())

    await _run_frames(brain, bus, _synthetic_climb())

    events = []
    while not event_queue.empty():
        events.append(event_queue.get_nowait())
    assert any(event.kind == "attempt_start" for event in events)
    assert (await attempt_queue.get()).outcome == "top"


@pytest.mark.asyncio
async def test_brain_runs_without_a_holdmap(monkeypatch: pytest.MonkeyPatch) -> None:
    import coach.brain as brain_module

    monkeypatch.setattr(brain_module, "HOLDMAP_GRACE_S", 0.01)
    bus = Bus()
    brain = Brain(_config(), bus, FakeVoice())
    await _run_frames(brain, bus, _synthetic_climb()[:2], publish_holdmap=False)
    assert brain.attempts.holdmap is None


@pytest.mark.asyncio
async def test_brain_survives_one_bad_frame() -> None:
    # Regression (C's audit): Brain.run() called self._process_state(state) with no
    # guard, so one exception (a bad detector, Memory, or the arbiter) ended run()
    # for the rest of the session. A single bad frame must be logged and skipped,
    # not silence the coach permanently.
    bus = Bus()
    event_queue = bus.subscribe("event")
    brain = Brain(_config(), bus, FakeVoice())
    original = brain._process_state
    calls: list[int] = []

    def flaky(state):
        calls.append(state.frame_idx)
        if len(calls) == 1:
            raise RuntimeError("synthetic detector failure")
        original(state)

    brain._process_state = flaky

    await _run_frames(brain, bus, _synthetic_climb())

    assert len(calls) == len(_synthetic_climb())
    events = []
    while not event_queue.empty():
        events.append(event_queue.get_nowait())
    assert any(event.kind == "attempt_start" for event in events)


@pytest.mark.asyncio
async def test_spoken_topic_updates_memory() -> None:
    bus = Bus()
    brain = Brain(_config(), bus, FakeVoice())
    states = asyncio.Queue()
    task = asyncio.create_task(brain.run(states))
    bus.publish("holdmap", HoldMap("demo", "pink", [], (1, 1), "colour"))
    for state in _synthetic_climb():
        await states.put(state)
    for _ in range(len(_synthetic_climb()) + 2):
        await asyncio.sleep(0)
    bus.publish("spoken", SpokenLine(3.0, 3.0, "stalling", "move"))
    await asyncio.sleep(0)
    assert brain.memory._cues_given == 1
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
