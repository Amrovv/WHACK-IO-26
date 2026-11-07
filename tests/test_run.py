from pathlib import Path

import pytest

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
