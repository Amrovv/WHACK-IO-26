import threading

import cv2
import numpy as np
import pytest

from coach.sources import FileSource, LatestQueue, parse_source


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24))
    for i in range(20):
        w.write(np.full((24, 32, 3), i * 10, np.uint8))
    w.release()
    return path


class FakeClock:
    def __init__(self):
        self.now = 100.0
        self.slept = 0.0

    def __call__(self):
        return self.now

    def sleep(self, s):
        self.slept += s
        self.now += s


def test_fast_mode_yields_all_frames_without_sleeping(clip):
    clk = FakeClock()
    src = FileSource(clip, fast=True, clock=clk, sleep=clk.sleep)
    frames = list(src)
    assert [f[1] for f in frames] == list(range(20))
    assert clk.slept == 0
    assert frames[0][2].shape == (24, 32, 3)


def test_paced_playback_runs_at_file_fps(clip):
    clk = FakeClock()
    src = FileSource(clip, clock=clk, sleep=clk.sleep)
    times = [t for t, _, _ in src]
    assert len(times) == 20
    assert times == pytest.approx([100.0 + i / 10 for i in range(20)])
    assert src.skipped == 0


def test_slow_consumer_skips_frames_instead_of_slowing(clip):
    clk = FakeClock()
    src = FileSource(clip, clock=clk, sleep=clk.sleep)
    got = []
    for _, idx, _ in src:
        got.append(idx)
        clk.now += 0.35  # consumer takes 3.5 frame periods per frame
    assert src.skipped > 0
    assert got == sorted(got) and len(got) < 20
    assert len(got) + src.skipped == 20


def test_parse_source_rejects_bad_specs():
    with pytest.raises(ValueError):
        parse_source("bogus")
    with pytest.raises(ValueError):
        parse_source("cam:x")
    with pytest.raises(FileNotFoundError):
        parse_source("file:/nope.mp4")


def test_latest_queue_drops_oldest():
    q = LatestQueue()
    q.put(1)
    q.put(2)
    q.put(3)
    assert q.get(timeout=0.1) == 3
    assert q.dropped == 2
    assert q.get(timeout=0.01) is None


def test_latest_queue_blocks_until_put_and_close_unblocks():
    q = LatestQueue()
    threading.Timer(0.05, q.put, args=("x",)).start()
    assert q.get(timeout=2) == "x"
    threading.Timer(0.05, q.close).start()
    assert q.get(timeout=2) is None
