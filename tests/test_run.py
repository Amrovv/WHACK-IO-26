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
from coach.contracts import ClimberState, CoachEvent, HoldMap, SpeakRequest
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


FAKE_FRAME_S = 0.001


class FakeSource:
    def __init__(self, n: int) -> None:
        self.n, self.closed = n, False

    def __iter__(self):
        for i in range(self.n):
            time.sleep(FAKE_FRAME_S)  # a little pacing, so a looping clip doesn't flood the loop
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
    ("name", "expected"),
    [
        ("climb_2_05_hand_searching.mp4", "climb_2"),  # never A's "climb2": other footage
        ("climb_6_12-1_new_high_point.mp4", "climb_6"),
        ("pink-v3_07_stall.mp4", "pink-v3"),
        ("Vidéo 07-11-2026 19 03 59.mp4", "fallback"),
    ],
)
def test_problem_for_reads_the_clip_name(name, expected):
    assert run.problem_for(Path(name), "fallback") == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("climb_9_00_empty_wall.mp4", True),
        ("climb_9_00_empty_wall_out_of_frame.mp4", True),
        ("climb_9_18_rest_between_attempts.mp4", False),
        ("Vidéo 07-11-2026 19 03 59.mp4", False),
    ],
)
def test_clip_00_is_the_empty_wall(name, expected):
    assert run.is_empty_wall(Path(name)) is expected


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

    assert [h["session"] for h in hellos if "pass" not in h] == [1, 2]  # the fake clip loops
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


# ---- Hold maps (coach/server/holds.py drives these) ---------------------------------------


def write_clip(path: Path, colour=(0, 200, 255), frames=40) -> None:
    """A 2 s clip at 20 fps. With colour=None each frame's brightness is 5x its index, so a
    frame that's read back tells which moment it came from."""
    import cv2

    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (32, 48))
    for i in range(frames):
        out.write(np.full((48, 32, 3), colour if colour is not None else 5 * i, np.uint8))
    out.release()


async def test_stills_offer_the_problems_empty_wall_then_the_current_frame(tmp_path, monkeypatch):
    _, parts = fake_parts()
    session, _ = make_session(tmp_path, monkeypatch, parts)
    videos = tmp_path / "videos"
    write_clip(videos / "climb_3_00_empty_wall.mp4", colour=None)
    (videos / "climb_3_00_empty_wall (1).mp4").write_bytes(b"")  # a copy: offered last
    (videos / "climb_1_00_empty_wall.mp4").write_bytes(b"")  # another problem's wall
    session.args = session.args_for("file:videos/climb_3_06_feet_cut.mp4")

    assert session.problem_id == "climb_3"
    stills = session.stills()
    assert stills[0].pop("duration") == pytest.approx(2.0)  # a video: any moment can be used
    assert stills == [
        {
            "key": "file:videos/climb_3_00_empty_wall.mp4",
            "name": "Empty wall · climb_3_00_empty_wall.mp4",
            "at": 1.0,  # the default moment
        },
        {  # unreadable: no length, so no moments
            "key": "file:videos/climb_3_00_empty_wall (1).mp4",
            "name": "Empty wall · climb_3_00_empty_wall (1).mp4",
        },
        {  # the clip being analysed
            "key": "file:videos/climb_3_06_feet_cut.mp4",
            "name": "This video · climb_3_06_feet_cut.mp4",
            "live": True,
        },
    ]
    img = session.read_still("file:videos/climb_3_00_empty_wall.mp4")
    assert img.shape == (48, 32, 3) and abs(int(img.mean()) - 100) <= 6  # frame 20, at 1 s
    wall = "file:videos/climb_3_00_empty_wall.mp4"
    assert abs(int(session.read_still(wall, at=0.5).mean()) - 50) <= 6  # frame 10
    assert abs(int(session.read_still(wall, at=60).mean()) - 190) <= 6  # kept inside: frame 38
    with pytest.raises(ValueError):
        session.read_still("file:videos/climb_1_00_empty_wall.mp4")  # not this problem's
    with pytest.raises(ValueError):
        session.read_still("frame")  # nothing has been analysed yet

    session.workers = [SimpleNamespace(latest_frame=np.zeros((4, 6, 3), np.uint8))]
    assert session.stills()[-1] == {"key": "frame", "name": "Current frame", "live": True}
    frame = session.read_still("frame")
    frame[:] = 9
    assert session.workers[0].latest_frame.max() == 0  # the editor gets its own copy


async def test_reload_restarts_analysis_so_the_brain_gets_the_new_map(tmp_path, monkeypatch):
    _, parts = fake_parts(n_frames=3)
    session, voice = make_session(tmp_path, monkeypatch, parts)
    await session.start(session.initial)
    first_brain = session.brain_task

    hello = await session.reload()

    assert hello["session"] == 2 and hello["spec"] == "file:videos/climb_1_02_send.mp4"
    assert first_brain.cancelled() and session.brain_task is not first_brain
    await session.stop()


async def test_pause_stops_analysis_but_keeps_the_last_frame_until_resume(tmp_path, monkeypatch):
    _, parts = fake_parts(n_frames=3)
    session, voice = make_session(tmp_path, monkeypatch, parts)
    hellos = []
    session.listen(hellos.append)
    await session.start(session.initial)
    first_brain = session.brain_task
    seen = SimpleNamespace(latest_frame=np.ones((4, 6, 3), np.uint8), stop=lambda: None)
    session.workers.insert(0, seen)  # the fake clip loops: pin the frame the pause will keep

    await session.pause()
    await session.pause()  # the editor may open twice (another picture): still one pause

    assert session.paused and session.workers == [] and session.brain_task is None
    assert first_brain.cancelled() and voice.stops == 1  # the coach stops talking
    assert session.latest_frame().max() == 1  # the editor can still build from the last frame
    assert session.stills()[-1]["key"] == "frame"

    hello = await session.resume()

    assert not session.paused and hello["session"] == 2
    assert [h["session"] for h in hellos if "pass" not in h] == [1, 2]  # the fake clip loops
    assert session.brain_task is not None and session._paused_frame is None  # fresh source
    assert await session.resume() is hello  # not paused: nothing restarts
    await session.stop()


async def test_a_switch_while_paused_starts_the_new_source(tmp_path, monkeypatch):
    _, parts = fake_parts(n_frames=3)
    session, _ = make_session(tmp_path, monkeypatch, parts)
    await session.start(session.initial)
    await session.pause()

    hello = await session.switch("file:videos/climb_3_06_feet_cut.mp4")

    assert not session.paused and hello["session"] == 2 and session.workers
    await session.stop()


async def test_reload_on_the_browser_camera_opens_a_fresh_camera(tmp_path, monkeypatch):
    _, parts = fake_parts(n_frames=3)
    session, _ = make_session(tmp_path, monkeypatch, parts)
    await session.start(session.initial)
    await session.switch("browser")
    old = session.camera

    await session.reload()

    assert session.camera is not old and isinstance(session.camera, run.BrowserCamera)
    assert session.feed_camera(b"jpeg") is True
    await session.stop()


async def test_pose_threads_keep_the_newest_frame():
    bus = Bus()
    bus.bind_loop()
    states = stubs.AsyncLatestQueue(asyncio.get_running_loop())
    _, parts = fake_parts(n_frames=3)
    perception = run.PoseThreads("clip.mp4", "p1", {}, bus, states, fast=True, parts=parts)
    assert perception.latest_frame is None
    perception.start()
    for _ in range(3):
        await asyncio.wait_for(states.get(), 2)
    await asyncio.to_thread(perception.done.wait, 2)
    perception.stop()
    assert perception.latest_frame.shape == (4, 6, 3)


# ---- A looping clip: analysed live once, replayed after ------------------------------------


def state(t, idx):
    return ClimberState(t, idx, True, {}, {}, {}, (0.5, 0.5))


def test_shift_times_moves_every_monotonic_time_and_the_frame_number():
    event = CoachEvent(t=10.0, kind="feet_low", priority=1, ttl_s=1.5, data={"limb": "LF"})
    req = SpeakRequest(event=event, level=1, mood="neutral", deadline=11.0)
    moved = run.shift_times(req, 100.0)
    assert (moved.event.t, moved.deadline, moved.event.ttl_s) == (110.0, 111.0, 1.5)
    assert req.event.t == 10.0  # a copy: the recording isn't changed
    assert run.shift_times(state(5.0, 3), 1.0, d_idx=40).frame_idx == 43
    assert run.shift_times({"t": 1.0}, 5.0) == {"t": 1.0}  # not a dataclass: as it is


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class InlineLoop:
    """Runs what's handed to call_soon_threadsafe straight away."""

    def call_soon_threadsafe(self, fn, *args) -> None:
        fn(*args)


class RecordingVoice:
    def __init__(self) -> None:
        self.calls = []

    def say(self, req) -> bool:
        self.calls.append(("say", req))
        return True

    def stop(self) -> None:
        self.calls.append(("stop", None))

    def debrief(self, history, habits, crux_hold):
        self.calls.append(("debrief", (history, habits, crux_hold)))

    def is_speaking(self) -> bool:
        return False


def test_the_first_pass_is_recorded_and_played_back_in_step_on_later_passes():
    clock, published, voice = FakeClock(), [], RecordingVoice()
    bus = SimpleNamespace(publish=lambda topic, p: published.append((topic, p)))
    passes = []
    replay = run.LoopReplay(bus, voice, InlineLoop(), clock=clock, on_pass=passes.append)
    tap_bus, tap_voice = run.TapBus(bus, replay), run.TapVoice(voice, replay)

    # Pass 1, live: perception and the brain publish, the brain talks to the voice.
    replay.start_pass(0)
    clock.now = 1001.0
    tap_bus.publish("state", state(1001.0, 30))
    tap_bus.publish("frame", b"jpeg")  # the video is live on every pass: not recorded
    clock.now = 1002.0
    event = CoachEvent(t=1002.0, kind="feet_low", priority=1, ttl_s=1.5)
    tap_bus.publish("event", event)
    assert tap_voice.say(SpeakRequest(event, 1, "neutral", deadline=1002.8)) is True
    clock.now = 1003.0
    tap_voice.debrief(["attempt 1"], {"mood": "neutral"}, 4)
    assert not replay.replaying and len(replay.items) == 4

    # Pass 2 starts 10 s later: perception rests, the recording plays as the clip does.
    published.clear()
    voice.calls.clear()
    clock.now = 1010.0
    replay.start_pass(90)
    assert replay.replaying and not replay.recording
    tap_bus.publish("state", state(1010.0, 90))  # a late live item: not recorded any more
    published.clear()
    clock.now = 1010.5
    replay.advance()
    assert published == []  # nothing was due yet
    clock.now = 1011.0
    replay.advance()
    ((topic, s),) = published
    assert topic == "state" and s.t == 1011.0 and s.frame_idx == 120  # now, in this pass
    clock.now = 1012.1
    replay.advance()
    assert published[-1][0] == "event" and published[-1][1].t == 1012.0
    kind, req = voice.calls[0]
    assert kind == "say" and req.event.t == pytest.approx(1012.1)
    assert req.deadline == pytest.approx(1012.9)  # the same 0.8 s of slack as live
    assert voice.calls[1:] == []  # the debrief isn't due yet

    # Pass 3 starts before pass 2 reached the debrief: it's played first, then pass 3 begins.
    clock.now = 1020.0
    replay.start_pass(180)
    assert voice.calls[-1] == ("debrief", (["attempt 1"], {"mood": "neutral"}, 4))
    published.clear()
    clock.now = 1021.0
    replay.advance()
    assert [p.frame_idx for _, p in published] == [210]
    assert passes == [2, 3]  # dashboards were told to start each replayed pass fresh


async def wait_for(condition, timeout_s=3.0) -> bool:
    for _ in range(int(timeout_s / 0.01)):
        if condition():
            return True
        await asyncio.sleep(0.01)
    return condition()


async def test_a_clip_plays_once_then_replays_its_live_pass_on_request():
    bus = Bus()
    bus.bind_loop()
    published = []  # (topic, while replaying)
    bus.publish = lambda topic, payload: published.append((topic, replay.replaying))
    states = stubs.AsyncLatestQueue(asyncio.get_running_loop())
    tracker, parts = fake_parts(n_frames=5)

    class ClipPass(FakeSource):  # like a real clip: capture times keep rising across passes
        def __iter__(self):
            for i in range(self.n):
                time.sleep(0.03)  # ~30 fps
                yield time.monotonic(), i, np.zeros((4, 6, 3), dtype=np.uint8)

    parts.open_source = lambda spec, **kw: ClipPass(5)
    seen = []
    analyse = tracker.process
    tracker.process = lambda t, idx, img: seen.append(idx) or analyse(t, idx, img)
    replay = run.LoopReplay(bus)
    perception = run.PoseThreads(
        "clip.mp4", "p1", {}, run.TapBus(bus, replay), states, parts=parts, replay=replay
    )

    ends = []
    perception.on_ended = lambda: ends.append(replay.passes)
    assert perception.request_replay() is False  # nothing has ended yet

    perception.start()
    assert await wait_for(perception.ended.is_set)  # a pass takes ~0.15 s
    await asyncio.sleep(0.2)
    shown = len(published)
    await asyncio.sleep(0.2)
    assert len(published) == shown and replay.passes == 1  # it stopped: no loop

    for n in (2, 3):  # the Replay button, twice
        assert perception.request_replay() is True
        assert await wait_for(lambda n=n: replay.passes == n and perception.ended.is_set())
    perception.stop()

    assert ends == [1, 2, 3]  # the end was reported after every pass
    assert all(idx < 5 for idx in seen)  # only the first pass was analysed
    assert ("frame", True) in published  # the video played on the replayed passes
    assert ("state", True) in published  # and the recorded poses played back with it


async def test_a_clip_session_records_through_the_tap_and_a_camera_session_does_not(
    tmp_path, monkeypatch
):
    _, parts = fake_parts(n_frames=3)
    session, _ = make_session(tmp_path, monkeypatch, parts)
    built = []
    session.make_brain = lambda cfg, bus, voice: (
        built.append((bus, voice)) or SimpleNamespace(run=lambda states: asyncio.Event().wait())
    )

    await session.start(session.initial)
    bus, voice = built[-1]
    assert isinstance(bus, run.TapBus) and isinstance(voice, run.TapVoice)
    assert session.replay is not None and session.workers[0].replay is session.replay

    await session.switch("browser")
    bus, voice = built[-1]
    assert bus is session.bus and voice is session.voice and session.replay is None
    await session.stop()


async def test_a_replayed_pass_starts_dashboards_fresh_with_the_hold_map(tmp_path, monkeypatch):
    # Set up by hand: a running fake clip loops non-stop and would announce passes of its own.
    holdmap = HoldMap("p1", "pink", [], (6, 4), "manual")
    _, parts = fake_parts()
    session, _ = make_session(tmp_path, monkeypatch, parts)
    hellos, maps = [], []
    session.listen(hellos.append)
    session.bus.publish = lambda topic, payload: (
        maps.append(payload) if topic == "holdmap" else None
    )
    session._loop = asyncio.get_running_loop()
    session.hello = {"session": 1, "spec": "file:videos/climb_1_02_send.mp4"}
    session.replay = run.LoopReplay(session.bus, on_pass=session._replayed_pass)
    session.workers = [SimpleNamespace(holdmap=holdmap)]

    session._replayed_pass(2)  # from perception's reader thread in real life
    await asyncio.sleep(0)

    assert hellos == [{"session": 1, "spec": "file:videos/climb_1_02_send.mp4", "pass": 2}]
    assert maps == [holdmap]  # after the hello, which made dashboards forget it
    stale, session.replay = session.replay, run.LoopReplay(session.bus)  # the source changed
    session._announce_pass(stale, 3)
    assert len(hellos) == 1  # an old source's pass: ignored


async def test_an_ended_clip_offers_replay_and_replaying_clears_it(tmp_path, monkeypatch):
    holdmap = HoldMap("p1", "pink", [], (6, 4), "manual")
    _, parts = fake_parts()
    session, _ = make_session(tmp_path, monkeypatch, parts)
    hellos, maps, asked = [], [], []
    session.listen(hellos.append)
    session.bus.publish = lambda topic, payload: (
        maps.append(payload) if topic == "holdmap" else None
    )
    session._loop = asyncio.get_running_loop()
    session.hello = {"session": 1}
    session.replay = run.LoopReplay(session.bus)
    worker = SimpleNamespace(holdmap=holdmap, request_replay=lambda: asked.append(1) or True)
    session.workers = [worker]

    session._clip_ended()  # from perception's reader thread in real life
    await asyncio.sleep(0)
    assert hellos[-1] == {"session": 1, "ended": True} and maps == [holdmap]

    assert await session.replay_clip() == {"session": 1, "ended": True} and asked == [1]
    session._announce_pass(session.replay, 2)  # the replay's first frame
    assert hellos[-1] == {"session": 1, "pass": 2}  # Replay goes away; the page starts fresh

    worker.request_replay = lambda: False  # playing: nothing to replay
    with pytest.raises(ValueError):
        await session.replay_clip()


def test_the_replay_endpoint_asks_the_session(tmp_path):
    from fastapi.testclient import TestClient

    from coach.server.app import create_app

    calls = []

    class Control:
        def listen(self, on_hello) -> None:
            pass

        async def replay_clip(self) -> dict:
            calls.append(1)
            if len(calls) > 1:
                raise ValueError("there's no ended clip to replay")
            return {"session": 1, "pass": 2}

    with TestClient(create_app(Bus(), control=Control())) as client:
        assert client.post("/api/replay").json() == {"hello": {"session": 1, "pass": 2}}
        assert client.post("/api/replay").status_code == 409
    with TestClient(create_app(Bus())) as client:
        assert client.post("/api/replay").status_code == 404  # no control: can't replay
