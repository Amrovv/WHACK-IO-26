"""Smoke-test for scripts/replay_brain.py.

Creates a synthetic pose-cache JSONL fixture (a simplified version of the
climb used in test_pipeline.py) and runs the replay script end-to-end,
checking that:

- it exits cleanly (return code 0),
- it emits at least one [CUE] log line (the arbiter spoke),
- it emits at least one [ATTEMPT] log line (an attempt was finalised),
- it correctly replaces the old _replay_reported hack (AttemptTracker has no
  such attribute after the run).
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Import the script module from its file path so we don't need to install it.
# ---------------------------------------------------------------------------

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "replay_brain.py"
_spec = importlib.util.spec_from_file_location("replay_brain", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

PrintVoice = _mod.PrintVoice
_state_fn = _mod._state
_config_fn = _mod._config
main = _mod.main


# ---------------------------------------------------------------------------
# Config fixture (loads the real default.yaml once per session).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def cfg() -> dict:
    return yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "config" / "default.yaml").read_text(
            encoding="utf-8"
        )
    )


# ---------------------------------------------------------------------------
# Helpers for building synthetic ClimberState dicts.
# ---------------------------------------------------------------------------


def _frame_dict(
    t: float,
    *,
    hip_y: float,
    contacts: dict[str, int | None] | None = None,
) -> dict:
    """Return a pose-cache line dict (the JSON representation of a ClimberState)."""
    contacts = contacts or {}
    kp = {
        "left_shoulder": [0.45, hip_y - 0.25, 1.0],
        "right_shoulder": [0.55, hip_y - 0.25, 1.0],
        "left_elbow": [0.42, hip_y - 0.15, 1.0],
        "right_elbow": [0.58, hip_y - 0.15, 1.0],
        "left_wrist": [0.46, hip_y - 0.30, 1.0],
        "right_wrist": [0.54, hip_y - 0.30, 1.0],
        "left_hip": [0.47, hip_y, 1.0],
        "right_hip": [0.53, hip_y, 1.0],
        "left_knee": [0.47, hip_y + 0.12, 1.0],
        "right_knee": [0.53, hip_y + 0.12, 1.0],
        "left_ankle": [0.47, hip_y + 0.24, 1.0],
        "right_ankle": [0.53, hip_y + 0.24, 1.0],
    }
    limbs = {
        "left_hand": [0.46, hip_y - 0.30],
        "right_hand": [0.54, hip_y - 0.30],
        "left_foot": [0.47, hip_y + 0.24],
        "right_foot": [0.53, hip_y + 0.24],
    }
    # All four limbs must be present; fill missing ones with None.
    all_contacts: dict[str, int | None] = {
        "left_hand": None,
        "right_hand": None,
        "left_foot": None,
        "right_foot": None,
    }
    all_contacts.update(contacts)
    return {
        "t": t,
        "frame_idx": int(t * 10),
        "person_found": True,
        "keypoints": kp,
        "limbs": limbs,
        "contacts": all_contacts,
        "hip": [0.5, hip_y],
    }


def _synthetic_pose_cache() -> list[dict]:
    """Minimal synthetic climb: ground → on-wall stall → top."""
    frames: list[dict] = []
    # Ground: stand still.
    for t in (0.0, 0.5, 1.0):
        frames.append(_frame_dict(t, hip_y=0.85))
    # Both feet on start hold → triggers attempt_start.
    frames.append(_frame_dict(1.5, hip_y=0.85, contacts={"left_foot": 1, "right_foot": 1}))
    # Pull on; hands on mid hold, hip rises.
    on_wall = {
        "left_foot": 1,
        "right_foot": 1,
        "left_hand": 3,
        "right_hand": 3,
    }
    frames.append(_frame_dict(2.0, hip_y=0.55, contacts=on_wall))
    # Stall for enough frames to trigger the stalling detector.
    for t in (2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0):
        frames.append(_frame_dict(t, hip_y=0.55, contacts=on_wall))
    # Reach the top hold → triggers top event.
    top = {
        "left_foot": 3,
        "right_foot": 3,
        "left_hand": 5,
        "right_hand": 5,
    }
    for t in (7.5, 8.0, 8.5, 9.0):
        frames.append(_frame_dict(t, hip_y=0.25, contacts=top))
    return frames


# ---------------------------------------------------------------------------
# A minimal hold-map JSON file for the synthetic climb.
# ---------------------------------------------------------------------------


def _holdmap_dict() -> dict:
    return {
        "problem_id": "test_replay",
        "colour": "pink",
        "image_size": [1920, 1080],
        "mode": "manual",
        "holds": [
            {"id": 1, "center": [0.5, 0.85], "bbox": [0, 0, 0, 0], "role": "start"},
            {"id": 3, "center": [0.5, 0.55], "bbox": [0, 0, 0, 0], "role": "mid"},
            {"id": 5, "center": [0.5, 0.20], "bbox": [0, 0, 0, 0], "role": "top"},
        ],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPrintVoice:
    """Unit tests for the PrintVoice helper class."""

    def test_say_returns_true_and_logs(self, caplog):
        voice = PrintVoice()
        from coach.contracts import CoachEvent, SpeakRequest

        req = SpeakRequest(
            event=CoachEvent(t=1.0, kind="stalling", priority=1, ttl_s=2.0),
            level=0,
            mood="neutral",
            deadline=1.5,
        )
        with caplog.at_level(logging.INFO):
            result = voice.say(req)
        assert result is True
        assert any("stalling" in record.message for record in caplog.records)

    def test_is_speaking_always_false(self):
        assert PrintVoice().is_speaking() is False

    def test_stop_is_noop(self):
        PrintVoice().stop()  # must not raise

    def test_debrief_logs_summary(self, caplog):
        from coach.contracts import AttemptSummary

        voice = PrintVoice()
        summary = AttemptSummary(
            attempt_no=1,
            duration_s=9.0,
            high_hold_id=5,
            outcome="top",
            stalls=[],
            cues_given=[],
            cues_followed=[],
        )
        with caplog.at_level(logging.INFO):
            voice.debrief([summary], {}, crux_hold=None)
        assert any("DEBRIEF" in record.message for record in caplog.records)


class TestReplayBrainMain:
    """Integration smoke-test: run main() on a synthetic pose-cache file."""

    @pytest.fixture()
    def pose_file(self, tmp_path: Path) -> Path:
        p = tmp_path / "climb.jsonl"
        lines = [json.dumps(f) for f in _synthetic_pose_cache()]
        p.write_text("\n".join(lines), encoding="utf-8")
        return p

    @pytest.fixture()
    def holdmap_file(self, tmp_path: Path) -> Path:
        p = tmp_path / "holdmap.json"
        p.write_text(json.dumps(_holdmap_dict()), encoding="utf-8")
        return p

    def test_exit_code_zero(self, pose_file: Path, holdmap_file: Path, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "replay_brain",
                "--pose",
                str(pose_file),
                "--holdmap",
                str(holdmap_file),
                "--voice",
                "print",
            ],
        )
        rc = main()
        assert rc == 0

    def test_missing_pose_file_returns_one(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["replay_brain", "--pose", str(tmp_path / "nope.jsonl"), "--voice", "print"],
        )
        rc = main()
        assert rc == 1

    def test_attempt_logged_and_no_replay_reported_hack(
        self, pose_file: Path, holdmap_file: Path, monkeypatch, caplog
    ):
        """The script must log the attempt summary and NOT monkey-patch AttemptTracker.

        We spy on AttemptTracker via monkeypatching the module the script
        imports from (coach.brain.attempts).  Because the script module was
        loaded with a non-standard spec name it is not in sys.modules, so we
        cannot use importlib.reload.  Instead we patch the already-imported
        AttemptTracker *name* inside the script module itself so that the main()
        call uses our spy class.
        """
        import coach.brain.attempts as _at_mod

        original_class = _at_mod.AttemptTracker
        seen_instances: list = []

        class _Spy(original_class):  # type: ignore[misc]
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                seen_instances.append(self)

        # Patch both the source module and the name bound inside the script module.
        monkeypatch.setattr(_at_mod, "AttemptTracker", _Spy)
        monkeypatch.setattr(_mod, "AttemptTracker", _Spy)

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "replay_brain",
                "--pose",
                str(pose_file),
                "--holdmap",
                str(holdmap_file),
                "--voice",
                "print",
            ],
        )
        with caplog.at_level(logging.INFO):
            rc = _mod.main()

        assert rc == 0
        # At least one attempt should be logged.
        assert any("[ATTEMPT]" in r.message for r in caplog.records), (
            "Expected at least one [ATTEMPT] log line; got:\n"
            + "\n".join(r.message for r in caplog.records)
        )
        # The old hack must be gone: no instance should have _replay_reported.
        for inst in seen_instances:
            assert not hasattr(inst, "_replay_reported"), (
                "AttemptTracker instance still has _replay_reported; remove the hack."
            )

    def test_cue_spoken(self, pose_file: Path, holdmap_file: Path, monkeypatch, caplog):
        """The arbiter must actually speak at least one cue."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "replay_brain",
                "--pose",
                str(pose_file),
                "--holdmap",
                str(holdmap_file),
                "--voice",
                "print",
            ],
        )
        with caplog.at_level(logging.INFO):
            rc = _mod.main()
        assert rc == 0
        assert any("[SPOKEN]" in r.message for r in caplog.records), (
            "Expected at least one [SPOKEN] log line; got:\n"
            + "\n".join(r.message for r in caplog.records)
        )

    def test_no_holdmap_still_runs(self, pose_file: Path, monkeypatch):
        """Running without --holdmap must not crash (hold-dependent detectors self-disable)."""
        monkeypatch.setattr(
            sys, "argv", ["replay_brain", "--pose", str(pose_file), "--voice", "print"]
        )
        rc = _mod.main()
        assert rc == 0
