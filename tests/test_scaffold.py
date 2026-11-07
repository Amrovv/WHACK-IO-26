"""Smoke tests: contracts serialise and config loads. Delete once real tests exist."""

import dataclasses
from pathlib import Path

import yaml

from coach.contracts import EVENT_KINDS, ClimberState, CoachEvent, Limb, Suppressed


def test_config_has_all_sections():
    cfg = yaml.safe_load(Path("config/default.yaml").read_text())
    assert set(cfg) == {"perception", "holds", "brain", "arbiter", "voice"}


def test_contracts_roundtrip_to_dict():
    state = ClimberState(
        t=1.0,
        frame_idx=0,
        person_found=True,
        keypoints={},
        limbs={Limb.LH: (0.4, 0.5)},
        contacts={Limb.LH: 3},
        hip=(0.5, 0.7),
    )
    assert dataclasses.asdict(state)["contacts"] == {Limb.LH: 3}

    event = CoachEvent(t=1.0, kind="stalling", priority=1, ttl_s=1.2)
    assert event.kind in EVENT_KINDS
    assert dataclasses.asdict(Suppressed(event, "cooldown"))["event"]["kind"] == "stalling"
