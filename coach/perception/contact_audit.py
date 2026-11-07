"""Audit cached contacts against hold geometry and raw pose visibility.

Run with ``python -m coach.perception.contact_audit --cache ... --holdmap ...``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from coach.contracts import ClimberState, Limb
from coach.perception.contacts import LIMB_LANDMARKS
from coach.perception.serialize import state_from_line

log = logging.getLogger(__name__)


def audit(cache: Path, holdmap: Path, stale_s: float = 1.0, resume_visibility: float = 0.7) -> None:
    """Log frame timeline, implausible spans, and long contacts with suspect pose evidence."""
    data = json.loads(holdmap.read_text(encoding="utf-8"))
    holds = {h["id"]: h for h in data["holds"]}
    states = [
        state_from_line(line) for line in cache.read_text(encoding="utf-8").splitlines() if line
    ]
    for state in states:
        log.info(
            "f=%d t=%.3f " + " ".join(f"{limb.value}={state.contacts.get(limb)}" for limb in Limb),
            state.frame_idx,
            state.t,
        )
    for state in states:
        attached = [
            (limb, state.contacts.get(limb)) for limb in Limb if state.contacts.get(limb) in holds
        ]
        for i, (a, aid) in enumerate(attached):
            for b, bid in attached[i + 1 :]:
                if aid != bid and _centre_distance(
                    holds[aid]["center"], holds[bid]["center"]
                ) > 2.2 * _arm_length(state):
                    log.warning(
                        "implausible f=%d t=%.3f %s=%d %s=%d distance=%.3f",
                        state.frame_idx,
                        state.t,
                        a.value,
                        aid,
                        b.value,
                        bid,
                        _centre_distance(holds[aid]["center"], holds[bid]["center"]),
                    )
    for limb in Limb:
        run: list[ClimberState] = []
        run_id = None
        for state in [*states, None]:
            hold_id = state.contacts.get(limb) if state else None
            if hold_id != run_id:
                if len(run) > 1 and run[-1].t - run[0].t >= stale_s:
                    below = all(_visibility(s, limb) < resume_visibility for s in run)
                    still = _still_while_hip_moves(run, limb)
                    if below or still:
                        log.warning(
                            "stale %s hold=%s f=%d..%d t=%.3f..%.3f below_resume=%s frozen_like=%s",
                            limb.value,
                            run_id,
                            run[0].frame_idx,
                            run[-1].frame_idx,
                            run[0].t,
                            run[-1].t,
                            below,
                            still,
                        )
                run, run_id = [], hold_id
            if state and hold_id is not None:
                run.append(state)


def _visibility(state: ClimberState, limb: Limb) -> float:
    return min(
        (state.keypoints.get(n, (0.0, 0.0, 0.0))[2] for n in LIMB_LANDMARKS[limb]), default=0.0
    )


def _arm_length(state: ClimberState) -> float:
    pairs = (("left_shoulder", "left_wrist"), ("right_shoulder", "right_wrist"))
    lengths = [
        (
            (state.keypoints[a][0] - state.keypoints[b][0]) ** 2
            + (state.keypoints[a][1] - state.keypoints[b][1]) ** 2
        )
        ** 0.5
        for a, b in pairs
        if a in state.keypoints and b in state.keypoints
    ]
    return sum(lengths) / len(lengths) if lengths else 0.25


def _centre_distance(a: list[float], b: list[float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _still_while_hip_moves(run: list[ClimberState], limb: Limb) -> bool:
    if len(run) < 2:
        return False
    points = [s.limbs.get(limb) for s in run]
    if any(p is None for p in points):
        return False
    point_motion = _centre_distance(points[0], points[-1])
    hip_motion = _centre_distance(run[0].hip, run[-1].hip)
    return point_motion < 0.015 and hip_motion > 0.08


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--holdmap", type=Path, required=True)
    parser.add_argument("--stale-s", type=float, default=1.0)
    parser.add_argument("--resume-visibility", type=float, default=0.7)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    audit(args.cache, args.holdmap, args.stale_s, args.resume_visibility)


if __name__ == "__main__":
    main()
