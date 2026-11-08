from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from coach.contracts import ClimberState, Hold, HoldMap, Limb
from coach.perception.config import load_config
from coach.perception.holds import load_holdmap_with_shapes
from coach.perception.serialize import state_from_line
from coach.perception.transcript import TranscriptEngine, draw_panel, run_over_states


def _state(t: float, idx: int, hip_y: float, contacts=None) -> ClimberState:
    return ClimberState(
        t,
        idx,
        True,
        {},
        {limb: (0.5, hip_y) for limb in Limb},
        contacts or {limb: None for limb in Limb},
        (0.5, hip_y),
    )


def test_incremental_and_precomputed_are_identical() -> None:
    config = load_config()
    holdmap = HoldMap(
        "synthetic",
        "blue",
        [Hold(1, (0.4, 0.8), (0.3, 0.7, 0.5, 0.9), "start")],
        (100, 100),
        "manual",
    )
    states = [
        _state(
            i / 30,
            i,
            0.8 if i < 5 else 0.65,
            {
                **{limb: None for limb in Limb},
                Limb.LF: 1 if i >= 8 else None,
                Limb.RF: 1 if i >= 8 else None,
            },
        )
        for i in range(15)
    ]
    engine = TranscriptEngine(config, holdmap)
    incremental = [entry for state in states for entry in engine.update(state)]
    whole = run_over_states(states, config, holdmap)
    assert [asdict(entry) for entry in incremental] == [asdict(entry) for entry in whole]
    assert any(entry.source == "perception" for entry in whole)
    assert any(entry.kind == "attempt_start" for entry in whole)


def test_attempt_can_start_from_hip_rise() -> None:
    config = load_config()
    keypoints = {
        "left_hip": (0.45, 0.7, 1.0),
        "right_hip": (0.55, 0.7, 1.0),
        "left_knee": (0.45, 0.8, 1.0),
        "right_knee": (0.55, 0.8, 1.0),
        "left_ankle": (0.45, 0.9, 1.0),
        "right_ankle": (0.55, 0.9, 1.0),
        "left_shoulder": (0.45, 0.5, 1.0),
        "right_shoulder": (0.55, 0.5, 1.0),
        "left_elbow": (0.4, 0.45, 1.0),
        "right_elbow": (0.6, 0.45, 1.0),
        "left_wrist": (0.4, 0.4, 1.0),
        "right_wrist": (0.6, 0.4, 1.0),
    }
    states = []
    for idx in range(25):
        y = 0.7 if idx < 5 else 0.65
        kp = {**keypoints, "left_hip": (0.45, y, 1.0), "right_hip": (0.55, y, 1.0)}
        states.append(
            ClimberState(
                idx / 30,
                idx,
                True,
                kp,
                {limb: (0.5, y) for limb in Limb},
                {limb: None for limb in Limb},
                (0.5, y),
            )
        )
    entries = run_over_states(states, config, None)
    assert any(entry.kind == "attempt_start" for entry in entries)


def test_draw_panel_empty_and_long_entries() -> None:
    assert draw_panel([], 0, 240, 360).shape == (240, 360, 3)
    engine = TranscriptEngine(load_config(), None)
    engine.entries.append(
        __import__("coach.perception.transcript", fromlist=["TranscriptEntry"]).TranscriptEntry(
            0, "detector", "long", "A very long transcript line " * 40
        )
    )
    assert draw_panel(engine.entries, 0, 240, 360).shape == (240, 360, 3)


def test_real_cache_transcript_regression() -> None:
    cache = Path("data/pose_cache/climb_1_02_clean_send_steady_pace.jsonl")
    holdmap_path = Path("data/holdmaps/climb_1.json")
    if not cache.exists() or not holdmap_path.exists():
        return
    states = [
        state_from_line(line)
        for line in cache.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    holdmap, _ = load_holdmap_with_shapes("climb_1")
    entries = run_over_states(states, load_config(), holdmap)
    assert any(entry.kind == "attempt_start" for entry in entries)
    spoken = {entry.summary for entry in entries if entry.source == "voice"}
    for phrase in ("Open up.", "Breathe out.", "Bring feet up.", "Lock it down."):
        assert phrase in spoken
    assert "Dont stop." in spoken or "Don't stop." in spoken
    assert any(
        entry.source == "arbiter" and entry.kind == "rejected" and entry.details.get("reason")
        for entry in entries
    )
