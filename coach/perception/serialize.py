"""ClimberState <-> JSON lines (the pose-cache format shared with B).

One state per line. `Limb` keys become their string values ("left_hand", ...).
Tuples become JSON arrays. `contacts` values are hold ids or null.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from coach.contracts import ClimberState, Limb


def state_to_dict(state: ClimberState) -> dict:
    d = asdict(state)
    d["limbs"] = {Limb(k).value: list(v) for k, v in state.limbs.items()}
    d["contacts"] = {Limb(k).value: v for k, v in state.contacts.items()}
    d["keypoints"] = {k: list(v) for k, v in state.keypoints.items()}
    d["hip"] = list(state.hip)
    return d


def state_from_dict(d: dict) -> ClimberState:
    return ClimberState(
        t=d["t"],
        frame_idx=d["frame_idx"],
        person_found=d["person_found"],
        keypoints={k: (v[0], v[1], v[2]) for k, v in d["keypoints"].items()},
        limbs={Limb(k): (v[0], v[1]) for k, v in d["limbs"].items()},
        contacts={Limb(k): v for k, v in d["contacts"].items()},
        hip=(d["hip"][0], d["hip"][1]),
    )


def state_to_line(state: ClimberState) -> str:
    return json.dumps(state_to_dict(state), separators=(",", ":"))


def state_from_line(line: str) -> ClimberState:
    return state_from_dict(json.loads(line))
