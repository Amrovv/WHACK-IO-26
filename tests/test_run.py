import asyncio
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import run
from coach import stubs
from coach.bus import Bus
from coach.contracts import ClimberState, HoldMap
from run import build_hello, parse_args


def hello_for(argv, probe=lambda path: {"frames": 900, "fps": 30.0}):
    return build_hello(parse_args(argv), probe=probe)


def test_webcam_is_live():
    hello = hello_for(["--source", "cam:0", "--problem", "pink-v3"])
    assert hello["mode"] == "live"
    assert hello["source"] == {"kind": "camera", "index": 0}


def test_video_file_is_offline_with_clip_length(tmp_path: Path):
    clip = tmp_path / "attempt1.mp4"
    clip.write_bytes(b"")
    hello = hello_for(["--source", f"file:{clip}", "--problem", "pink-v3"])
    assert hello["mode"] == "offline"
    assert hello["source"] == {"kind": "file", "name": "attempt1.mp4", "frames": 900, "fps": 30.0}


def test_pose_replay_is_offline(tmp_path: Path):
    poses = tmp_path / "clip.jsonl"
    poses.write_text("")
    hello = hello_for(["--replay-pose", str(poses), "--problem", "pink-v3", "--no-llm"])
    assert hello["mode"] == "offline"
    assert hello["source"] == {"kind": "poses", "name": "clip.jsonl"}
    assert hello["no_llm"] is True


@pytest.mark.parametrize("probe_result", [{}, {"frames": 0, "fps": 0}])
def test_unreadable_clip_still_says_offline(tmp_path: Path, probe_result):
    clip = tmp_path / "broken.mp4"
    clip.write_bytes(b"")
    hello = hello_for(["--source", f"file:{clip}", "--problem", "x"], probe=lambda p: probe_result)
    assert hello["mode"] == "offline"
    assert hello["source"]["name"] == "broken.mp4"


# ---- Wiring: the perception -> brain hand-off -------------------------------------------


class FakeFrameQueue:
    """Stands in for A's thread -> thread LatestQueue."""

    def __init__(self) -> None:
        self.items, self.closed, self.cond = [], False, threading.Condition()

    def put(self, item) -> None:
        with self.cond:
            self.items = [item]
            self.cond.notify()

    def get(self, timeout=None):
        with self.cond:
            self.cond.wait_for(lambda: self.items or self.closed, timeout)
            return self.items.pop() if self.items else None

    def close(self) -> None:
        with self.cond:
            self.closed = True
            self.cond.notify_all()


class FakeSource:
    def __init__(self, n: int) -> None:
        self.n, self.closed = n, False

    def __iter__(self):
        for i in range(self.n):
            yield 100.0 + i * 0.2, i, np.zeros((4, 6, 3), dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


class FakeTracker:
    def __init__(self) -> None:
        self.closed = False

    def process(self, t, idx, img) -> ClimberState:
        return ClimberState(t, idx, True, {}, {}, {}, (0.5, 0.5))

    def close(self) -> None:
        self.closed = True


def fake_parts(n_frames=5, holdmap=None):
    tracker = FakeTracker()

    def load_holdmap(problem_id):
        if holdmap is None:
            raise FileNotFoundError(problem_id)
        return holdmap, {}

    return tracker, SimpleNamespace(
        open_source=lambda spec, **kw: FakeSource(n_frames),
        make_tracker=lambda hm, shapes: tracker,
        load_holdmap=load_holdmap,
        frame_queue=FakeFrameQueue,
    )


async def test_pose_threads_feed_brain_queue_and_bus_in_fast_mode():
    bus = Bus()
    bus.bind_loop()
    published = []
    bus.publish = lambda topic, payload: published.append(topic)  # record, don't deliver
    states = stubs.AsyncLatestQueue(asyncio.get_running_loop())
    holdmap = HoldMap("p1", "pink", [], (6, 4), "manual")
    tracker, parts = fake_parts(n_frames=5, holdmap=holdmap)
    perception = run.PoseThreads("clip.mp4", "p1", {}, bus, states, fast=True, parts=parts)

    perception.start()
    got = [await asyncio.wait_for(states.get(), 2) for _ in range(5)]  # fast mode drops nothing
    await asyncio.to_thread(perception.done.wait, 2)
    perception.stop()

    assert [s.frame_idx for s in got] == [0, 1, 2, 3, 4]
    assert published[0] == "holdmap"
    assert published.count("state") == 5
    assert published.count("frame") == 5  # 0.2 s apart, so none throttled at 10 fps
    assert tracker.closed


async def test_pose_threads_live_mode_runs_and_stops_cleanly():
    bus = Bus()
    bus.bind_loop()
    states = stubs.AsyncLatestQueue(asyncio.get_running_loop())
    tracker, parts = fake_parts(n_frames=50)
    perception = run.PoseThreads(0, "p1", {}, bus, states, fast=False, parts=parts)  # camera

    perception.start()
    first = await asyncio.wait_for(states.get(), 2)
    await asyncio.to_thread(perception.done.wait, 2)  # camera source ran out
    perception.stop()

    assert isinstance(first, ClimberState)
    assert perception.spec == "cam:0"
    assert tracker.closed


def test_build_perception_falls_back_to_video_only_without_a(monkeypatch, tmp_path):
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"")

    def missing():
        raise ImportError("No module named 'coach.perception.pose'")

    monkeypatch.setattr(run, "a_parts", missing)
    args = parse_args(["--source", f"file:{clip}", "--problem", "x"])
    workers = run.build_perception(args, {}, Bus(), object())
    assert [type(w) for w in workers] == [stubs.VideoPerception]


def test_build_perception_falls_back_when_pose_model_fails(tmp_path):
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"")
    _, parts = fake_parts()
    parts.make_tracker = lambda hm, shapes: (_ for _ in ()).throw(RuntimeError("no model file"))
    args = parse_args(["--source", f"file:{clip}", "--problem", "x"])
    workers = run.build_perception(args, {}, Bus(), object(), parts=parts)
    assert [type(w) for w in workers] == [stubs.VideoPerception]


async def test_fast_pose_replay_runs_end_to_end(monkeypatch, tmp_path):
    """run.py with whatever brain and voice this branch has (B's real Brain once merged)."""
    poses = tmp_path / "poses.jsonl"
    lines = [
        {"t": i * 0.1, "frame_idx": i, "person_found": True, "keypoints": {},
         "limbs": {}, "contacts": {}, "hip": [0.5, 0.6]}
        for i in range(30)
    ]  # fmt: skip
    poses.write_text("".join(json.dumps(x) + "\n" for x in lines))
    monkeypatch.setattr(run, "LOG_DIR", tmp_path / "logs")

    code = await run.main(parse_args(["--replay-pose", str(poses), "--problem", "x", "--fast"]))

    assert code == 0
    (log_file,) = (tmp_path / "logs").glob("*.jsonl")
    topics = [json.loads(line)["topic"] for line in log_file.read_text().splitlines()]
    assert topics.count("state") == 30


# ---- Switching the source -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "maps", "expected"),
    [
        ("climb_2_05_hand_searching.mp4", [], "climb_2"),
        ("climb_2_05_hand_searching.mp4", ["climb2"], "climb2"),  # A's hold map naming
        ("climb_6_12-1_new_high_point.mp4", [], "climb_6"),
        ("pink-v3_07_stall.mp4", [], "pink-v3"),
        ("Vidéo 07-11-2026 19 03 59.mp4", [], "fallback"),
    ],
)
def test_problem_for_reads_the_clip_name(tmp_path: Path, name, maps, expected):
    for m in maps:
        (tmp_path / f"{m}.json").write_text("{}")
    assert run.problem_for(Path(name), "fallback", holdmap_dir=tmp_path) == expected


def test_list_clips_is_natural_order_and_skips_other_files(tmp_path: Path):
    (tmp_path / "mp4").mkdir()
    for n in ["climb_10_00.mp4", "climb_2_00.mp4", "climb_2_00.mov", "notes.txt"]:
        (tmp_path / "mp4" / n).write_bytes(b"")
    assert [p.name for p in run.list_clips(tmp_path)] == ["climb_2_00.mp4", "climb_10_00.mp4"]


def make_session(tmp_path: Path, monkeypatch, parts):
    videos = tmp_path / "videos"
    videos.mkdir()
    for n in ["climb_1_02_send.mp4", "climb_3_06_feet_cut.mp4"]:
        (videos / n).write_bytes(b"")
    monkeypatch.setattr(run, "ROOT", tmp_path)
    args = parse_args(["--source", f"file:{videos / 'climb_1_02_send.mp4'}", "--problem", "p1"])

    class Voice:
        stops = 0

        def stop(self) -> None:
            Voice.stops += 1

    class Brain:
        def __init__(self, cfg, bus, voice) -> None:
            pass

        async def run(self, states) -> None:
            await asyncio.Event().wait()

    bus = Bus()
    bus.bind_loop()
    return run.Session(args, {}, bus, Voice(), Brain, parts=parts, video_dir=videos), Voice


async def test_session_offers_clips_and_camera_and_rejects_anything_else(tmp_path, monkeypatch):
    _, parts = fake_parts()
    session, _ = make_session(tmp_path, monkeypatch, parts)
    specs = [o["spec"] for o in session.options()]
    assert specs == [
        "file:videos/climb_1_02_send.mp4",
        "file:videos/climb_3_06_feet_cut.mp4",
        "browser",
    ]
    with pytest.raises(ValueError):
        session.args_for("file:/etc/passwd")
    clip = session.args_for("file:videos/climb_3_06_feet_cut.mp4")
    assert clip.problem == "climb_3" and clip.source.endswith("climb_3_06_feet_cut.mp4")
    assert isinstance(session.args_for("browser").source, run.BrowserCamera)
    assert session.args_for(specs[0]) is session.initial  # back to the start keeps --problem


async def test_session_switch_rebuilds_and_announces_the_new_source(tmp_path, monkeypatch):
    _, parts = fake_parts(n_frames=3)
    session, voice = make_session(tmp_path, monkeypatch, parts)
    hellos = []
    session.listen(hellos.append)
    await session.start(session.initial)
    first_workers, first_brain = session.workers, session.brain_task

    hello = await session.switch("browser")

    assert [h["session"] for h in hellos] == [1, 2]
    assert hello["spec"] == "browser" and hello["mode"] == "live"
    assert hello["source"] == {"kind": "browser_camera"}
    assert first_brain.cancelled() and session.workers is not first_workers
    assert voice.stops == 1
    assert session.feed_camera(b"jpeg") is True
    assert await session.switch("browser") is hello  # already on it: nothing restarts
    await session.stop()
    assert session.feed_camera(b"jpeg") is False


async def test_pose_threads_read_the_browser_camera_and_stop_cleanly():
    import cv2

    bus = Bus()
    bus.bind_loop()
    states = stubs.AsyncLatestQueue(asyncio.get_running_loop())
    tracker, parts = fake_parts()
    cam = run.BrowserCamera()
    perception = run.PoseThreads(cam, "p1", {}, bus, states, fast=False, parts=parts)
    ok, buf = cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))

    perception.start()
    cam.put(buf.tobytes())
    state = await asyncio.wait_for(states.get(), 2)
    await asyncio.to_thread(perception.stop)  # must not hang waiting for another frame

    assert perception.spec == "browser" and not perception.is_file
    assert state.frame_idx == 0
    assert tracker.closed


async def test_a_switch_during_startup_waits_for_it(tmp_path, monkeypatch):
    _, parts = fake_parts(n_frames=3)
    session, _ = make_session(tmp_path, monkeypatch, parts)
    build = run.build_perception

    def slow_for_files(args, *rest):  # like loading the pose model for the first source
        if isinstance(args.source, str):
            time.sleep(0.2)
        return build(args, *rest)

    monkeypatch.setattr(run, "build_perception", slow_for_files)
    starting = asyncio.create_task(session.start(session.initial))
    await asyncio.sleep(0)  # startup is now off the loop, building perception
    hello = await session.switch("browser")
    await starting

    assert hello["session"] == 2 and session.hello is hello
    assert isinstance(session.args.source, run.BrowserCamera)  # startup didn't overwrite it
    assert [w.custom for w in session.workers] == [session.camera]
    await session.stop()
