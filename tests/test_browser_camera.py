import threading

import cv2
import numpy as np

from coach.server.camera import BrowserCamera


def jpeg(shade: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((24, 32, 3), shade, np.uint8))
    assert ok
    return buf.tobytes()


def test_keeps_only_the_newest_frame_and_stamps_arrival_time():
    times = iter([1.0, 2.0])
    cam = BrowserCamera(clock=lambda: next(times))
    cam.put(jpeg(10))
    cam.put(jpeg(200))  # arrives before perception took the first: replaces it
    frames = iter(cam)
    t, idx, img = next(frames)
    cam.close()
    assert (t, idx) == (2.0, 0)
    assert img.shape == (24, 32, 3) and img.mean() > 150
    assert cam.dropped == 1 and cam.received == 2
    assert list(frames) == []


def test_skips_frames_that_arent_jpegs():
    cam = BrowserCamera(clock=lambda: 5.0)
    cam.put(b"not a jpeg")
    got = []

    def read() -> None:
        got.extend(cam)

    reader = threading.Thread(target=read)
    reader.start()
    cam.put(jpeg(50))
    while not got and reader.is_alive():
        reader.join(0.01)
    cam.close()
    reader.join(1)
    assert not reader.is_alive()
    assert [idx for _, idx, _ in got] == [0]


def test_close_wakes_a_waiting_reader():
    cam = BrowserCamera()
    reader = threading.Thread(target=lambda: list(cam))
    reader.start()
    cam.close()
    reader.join(1)
    assert not reader.is_alive()
