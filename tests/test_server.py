import asyncio
import base64
import json

import pytest
from fastapi.testclient import TestClient

from coach.bus import Bus
from coach.contracts import CoachEvent, HoldMap
from coach.server.app import (
    CLIENT_BACKLOG,
    STATE_HZ,
    Client,
    Hub,
    create_app,
    encode,
    mjpeg_part,
    read_bus_log,
    replay,
)


class FakeClock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def event(t: float = 1.0) -> CoachEvent:
    return CoachEvent(t=t, kind="stalling", priority=1, ttl_s=1.2, data={"hold_id": 6})


# ---- Message format -----------------------------------------------------------------------


def test_encode_uses_topic_t_payload():
    msg = json.loads(encode("event", 12.5, event()))
    assert msg == {
        "topic": "event",
        "t": 12.5,
        "payload": {
            "t": 1.0,
            "kind": "stalling",
            "priority": 1,
            "ttl_s": 1.2,
            "data": {"hold_id": 6},
        },
    }


def test_encode_frame_bytes_as_base64():
    msg = json.loads(encode("frame", 0.0, b"\xff\xd8jpeg"))
    assert base64.b64decode(msg["payload"]) == b"\xff\xd8jpeg"


def test_mjpeg_part_has_boundary_and_length():
    part = mjpeg_part(b"abc")
    assert part.startswith(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: 3\r\n\r\n")
    assert part.endswith(b"abc\r\n")


# ---- Per-client throttle and backlog --------------------------------------------------------


def test_state_is_throttled_per_client():
    clock = FakeClock()
    a, b = Client(clock), Client(clock)
    step = 1.0 / STATE_HZ
    for t in (0.0, step / 2, step, step * 1.5):
        clock.t = t
        a.offer("state", f"s{t}")
    clock.t = step * 1.5
    b.offer("state", "late joiner")  # b has its own throttle, so it gets this one
    assert a.queue.qsize() == 2
    assert b.queue.qsize() == 1


def test_other_topics_are_not_throttled():
    client = Client(FakeClock())
    for i in range(5):
        client.offer("event", f"e{i}")
    assert client.queue.qsize() == 5


def test_full_backlog_drops_the_oldest():
    client = Client(FakeClock())
    for i in range(CLIENT_BACKLOG + 3):
        client.offer("event", f"e{i}")
    assert client.dropped == 3
    assert client.queue.get_nowait() == "e3"


# ---- Hub --------------------------------------------------------------------------------


async def test_hub_forwards_bus_items_to_clients():
    bus = Bus()
    bus.bind_loop()
    hub = Hub(bus, clock=FakeClock(5.0))
    hub.start()
    client = hub.connect()
    bus.publish("event", event())
    msg = json.loads(await asyncio.wait_for(client.queue.get(), 1))
    await hub.stop()
    assert msg["topic"] == "event" and msg["t"] == 5.0 and msg["payload"]["kind"] == "stalling"


async def test_late_joiner_gets_hello_and_latest_holdmap():
    hub = Hub(Bus(), hello={"problem_id": "pink-v3"}, clock=FakeClock())
    holdmap = HoldMap(
        problem_id="pink-v3", colour="pink", holds=[], image_size=(1280, 720), mode="manual"
    )
    await hub.dispatch("holdmap", holdmap)
    client = hub.connect()
    topics = [json.loads(client.queue.get_nowait())["topic"] for _ in range(2)]
    assert topics == ["hello", "holdmap"]


async def test_frames_yields_new_jpegs_for_mjpeg():
    clock = FakeClock()
    hub = Hub(Bus(), clock=clock)
    frames = hub.frames()
    first = asyncio.ensure_future(anext(frames))
    await asyncio.sleep(0)
    await hub.dispatch("frame", b"jpeg-1")
    assert await asyncio.wait_for(first, 1) == b"jpeg-1"

    clock.t = 1.0
    nxt = asyncio.ensure_future(anext(frames))
    await asyncio.sleep(0)
    await hub.dispatch("frame", base64.b64encode(b"jpeg-2").decode())  # replayed frame
    assert await asyncio.wait_for(nxt, 1) == b"jpeg-2"


# ---- Replay -----------------------------------------------------------------------------


def write_log(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def test_read_bus_log_sorts_and_skips_bad_lines(tmp_path):
    path = tmp_path / "bus.jsonl"
    path.write_text(
        json.dumps({"t": 2.0, "topic": "event", "payload": {}}) + "\n"
        + "not json\n\n"
        + json.dumps({"t": 1.0, "topic": "state", "payload": {"t": 1.0}}) + "\n"
        + json.dumps({"topic": "event"}) + "\n"
    )  # fmt: skip
    assert [(t, topic) for t, topic, _ in read_bus_log(path)] == [(1.0, "state"), (2.0, "event")]


async def test_replay_publishes_in_order_at_original_spacing(tmp_path):
    path = tmp_path / "bus.jsonl"
    write_log(path, [
        {"t": 10.0, "topic": "state", "payload": {"t": 10.0}},
        {"t": 10.5, "topic": "event", "payload": {"kind": "stalling"}},
        {"t": 10.5, "topic": "spoken", "payload": {"text": "Go"}},
        {"t": 12.0, "topic": "event", "payload": {"kind": "top"}},
    ])  # fmt: skip
    bus = Bus()
    seen: list[str] = []
    sleeps: list[float] = []
    for topic in ("state", "event", "spoken"):
        queue = bus.subscribe(topic)
        queue.put_nowait = lambda item, topic=topic: seen.append(topic)

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    await replay(bus, read_bus_log(path), sleep=fake_sleep)
    assert seen == ["state", "event", "spoken", "event"]
    assert sleeps == pytest.approx([0.5, 1.5])


async def test_replay_loop_starts_again():
    bus = Bus()
    queue = bus.subscribe("event")
    records = [(0.0, "event", {"n": 0}), (1.0, "event", {"n": 1})]
    calls = 0

    async def stop_after_second_pass(s: float) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await replay(bus, records, loop=True, sleep=stop_after_second_pass)
    assert [queue.get_nowait()["n"] for _ in range(queue.qsize())] == [0, 1, 0]


# ---- HTTP and WebSocket -------------------------------------------------------------------


def test_index_serves_the_dashboard():
    with TestClient(create_app(Bus())) as client:
        r = client.get("/")
    assert r.status_code == 200
    assert "Climbing Coach" in r.text


def test_websocket_sends_hello_then_bus_items():
    bus = Bus()
    with TestClient(create_app(bus, hello={"problem_id": "demo"})) as client:
        with client.websocket_connect("/ws") as ws:
            hello = ws.receive_json()
            client.portal.call(bus.publish, "event", event())  # publish on the app's loop
            msg = ws.receive_json()
    assert hello["topic"] == "hello" and hello["payload"] == {"problem_id": "demo"}
    assert msg["topic"] == "event" and msg["payload"]["kind"] == "stalling"
    assert isinstance(msg["t"], float)
