import json
import threading

import pytest

from coach.bus import Bus, to_jsonable
from coach.contracts import (
    AttemptSummary,
    ClimberState,
    CoachEvent,
    Hold,
    HoldMap,
    Limb,
    SpeakRequest,
    SpokenLine,
    Suppressed,
)


async def test_publish_subscribe_same_loop():
    bus = Bus()
    bus.bind_loop()
    queue = bus.subscribe("event")
    event = CoachEvent(t=1.0, kind="stalling", priority=1, ttl_s=1.2)

    bus.publish("event", event)
    got = await asyncio_wait_for(queue)

    assert got is event


async def test_publish_from_other_thread_is_safe():
    bus = Bus()
    bus.bind_loop()
    queue = bus.subscribe("state")
    state = ClimberState(
        t=1.0,
        frame_idx=0,
        person_found=True,
        keypoints={},
        limbs={},
        contacts={},
        hip=(0.5, 0.5),
    )

    thread = threading.Thread(target=lambda: bus.publish("state", state))
    thread.start()
    thread.join()

    got = await asyncio_wait_for(queue)
    assert got is state


def test_publish_with_no_subscribers_is_a_noop():
    bus = Bus()
    bus.publish(
        "holdmap",
        HoldMap(problem_id="x", colour="pink", holds=[], image_size=(1920, 1080), mode="manual"),
    )


@pytest.mark.parametrize(
    "payload",
    [
        Hold(id=1, center=(0.1, 0.9), bbox=(0.05, 0.85, 0.15, 0.95), role="start"),
        HoldMap(problem_id="pink-v3", colour="pink", holds=[], image_size=(1920, 1080), mode="detector"),
        ClimberState(
            t=12.3,
            frame_idx=42,
            person_found=True,
            keypoints={"nose": (0.5, 0.2, 0.9)},
            limbs={Limb.LH: (0.3, 0.4)},
            contacts={Limb.LH: 3, Limb.RH: None},
            hip=(0.5, 0.5),
        ),
        CoachEvent(
            t=1.0,
            kind="hand_searching",
            priority=1,
            ttl_s=1.2,
            data={"limb": "LH", "hold_id": 7, "dir": "up and left"},
        ),
        SpeakRequest(
            event=CoachEvent(t=1.0, kind="top", priority=0, ttl_s=2.0),
            level=1,
            mood="impressed",
            deadline=2.0,
        ),
        AttemptSummary(
            attempt_no=1,
            duration_s=30.0,
            high_hold_id=6,
            outcome="fall",
            stalls=[(3, 4.5)],
            cues_given=["feet_low"],
            cues_followed=["feet_low"],
        ),
        SpokenLine(t_event=1.0, t_audio_start=1.05, kind="stalling", text="Move."),
        Suppressed(event=CoachEvent(t=1.0, kind="arms_bent", priority=1, ttl_s=1.2), reason="cooldown"),
    ],
)
def test_to_jsonable_round_trips_through_json(payload):
    decoded = json.loads(json.dumps(to_jsonable(payload)))
    assert isinstance(decoded, dict)


def test_to_jsonable_converts_enum_dict_keys_and_values():
    state = ClimberState(
        t=1.0,
        frame_idx=0,
        person_found=True,
        keypoints={},
        limbs={Limb.LH: (0.1, 0.2)},
        contacts={Limb.LH: 3, Limb.RH: None},
        hip=(0.5, 0.5),
    )

    encoded = to_jsonable(state)

    assert encoded["limbs"] == {"left_hand": [0.1, 0.2]}
    assert encoded["contacts"] == {"left_hand": 3, "right_hand": None}


def test_bus_log_appends_one_jsonl_record_per_publish(tmp_path):
    log_path = tmp_path / "bus.jsonl"
    bus = Bus(log_path=log_path)

    bus.publish("event", CoachEvent(t=1.0, kind="top", priority=0, ttl_s=2.0))
    bus.publish("event", CoachEvent(t=2.0, kind="fall", priority=0, ttl_s=2.0))
    bus.close()

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["topic"] == "event"
    assert first["payload"]["kind"] == "top"


async def asyncio_wait_for(queue, timeout: float = 1.0):
    import asyncio

    return await asyncio.wait_for(queue.get(), timeout=timeout)
