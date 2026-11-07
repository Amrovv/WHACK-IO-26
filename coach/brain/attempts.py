"""Attempt state machine for the local coach brain."""

from __future__ import annotations

from typing import Any

from coach.brain.config import cfg
from coach.brain.features import Features, height_above
from coach.contracts import AttemptSummary, ClimberState, CoachEvent, HoldMap, Limb


def make_event(kind: str, t: float, data: dict, config: dict[str, Any]) -> CoachEvent:
    priority = int(cfg(config, "brain", "priorities")[kind])
    ttl = float(cfg(config, "arbiter", "ttl_s")[f"p{priority}"])
    return CoachEvent(t=t, kind=kind, priority=priority, ttl_s=ttl, data=data)


class AttemptTracker:
    """Drive GROUND -> ON_WALL -> GROUND.

    Completed summaries are retained in ``finished_attempts``; ``update`` returns
    only events so callers can publish events without special tuple handling.
    """

    def __init__(self, holdmap: HoldMap | None, config: dict[str, Any]) -> None:
        self.holdmap = holdmap
        self.config = config
        self.phase = "GROUND"
        self.finished_attempts: list[AttemptSummary] = []
        self._standing_hips: list[float] = []
        self._raised_since: float | None = None
        self._ground_since: float | None = None
        self._top_since: float | None = None
        self._fall_seen = False
        self._start_t = 0.0
        self._attempt_no = 0
        self._attempt_high: int | None = None
        self._best_hold = 0
        self._last_record_emitted = 0

    def _standing_y(self) -> float | None:
        return sum(self._standing_hips) / len(self._standing_hips) if self._standing_hips else None

    def _raised(self, state: ClimberState, features: Features) -> bool:
        standing = self._standing_y()
        return (
            standing is not None
            and features.leg_length > 0
            and height_above(standing, state.hip[1])
            >= float(cfg(self.config, "brain", "on_wall_margin")) * features.leg_length
        )

    def _grounded(self, state: ClimberState, features: Features) -> bool:
        feet_off = state.contacts.get(Limb.LF) is None and state.contacts.get(Limb.RF) is None
        return feet_off and not self._raised(state, features)

    def _is_top(self, state: ClimberState) -> bool:
        hand_ids = [state.contacts.get(Limb.LH), state.contacts.get(Limb.RH)]
        if any(value is None for value in hand_ids):
            return False
        if self.holdmap is None or not self.holdmap.holds:
            top_id = max(value for value in hand_ids if value is not None)
            return hand_ids[0] == hand_ids[1] == top_id
        top_ids = {hold.id for hold in self.holdmap.holds if hold.role == "top"}
        if not top_ids:
            top_ids = {max(hold.id for hold in self.holdmap.holds)}
        return all(value in top_ids for value in hand_ids)

    def _enter_wall(self, state: ClimberState) -> CoachEvent:
        self.phase = "ON_WALL"
        self._start_t = state.t
        self._attempt_no += 1
        self._attempt_high = None
        self._ground_since = None
        self._top_since = None
        self._fall_seen = False
        return make_event("attempt_start", state.t, {"attempt_no": self._attempt_no}, self.config)

    def _finish(self, state: ClimberState, outcome: str) -> AttemptSummary:
        summary = AttemptSummary(
            self._attempt_no, state.t - self._start_t, self._attempt_high, outcome, [], [], []
        )
        self.finished_attempts.append(summary)
        self.phase = "GROUND"
        self._ground_since = None
        self._raised_since = None
        if self._attempt_high is not None:
            self._best_hold = max(self._best_hold, self._attempt_high)
        return summary

    def update(self, state: ClimberState, features: Features) -> list[CoachEvent]:
        events: list[CoachEvent] = []
        if not state.person_found:
            return events
        if self.phase == "GROUND":
            features.add_calibration_sample(state)
            if not self._standing_hips or not self._raised(state, features):
                self._standing_hips.append(state.hip[1])
            if state.contacts.get(Limb.LF) is not None and state.contacts.get(Limb.RF) is not None:
                events.append(self._enter_wall(state))
            elif self._raised(state, features):
                self._raised_since = self._raised_since or state.t
                if state.t - self._raised_since >= float(
                    cfg(self.config, "brain", "on_wall_settle_s")
                ):
                    events.append(self._enter_wall(state))
            else:
                self._raised_since = None
            return events

        current_high = max(
            (hold for hold in state.contacts.values() if hold is not None), default=None
        )
        if current_high is not None and (
            self._attempt_high is None or current_high > self._attempt_high
        ):
            self._attempt_high = current_high
            if current_high > self._best_hold and current_high > self._last_record_emitted:
                self._last_record_emitted = current_high
                events.append(
                    make_event("high_point", state.t, {"hold_id": current_high}, self.config)
                )

        if self._is_top(state):
            self._top_since = self._top_since or state.t
            if state.t - self._top_since >= float(cfg(self.config, "brain", "top_hold_s")):
                events.append(
                    make_event("top", state.t, {"attempt_no": self._attempt_no}, self.config)
                )
                self._finish(state, "top")
                return events
        else:
            self._top_since = None

        if features.hip_velocity_y > float(cfg(self.config, "brain", "fall_speed")):
            self._fall_seen = True
        if self._grounded(state, features):
            self._ground_since = self._ground_since or state.t
            if state.t - self._ground_since >= float(cfg(self.config, "brain", "ground_settle_s")):
                outcome = "fall" if self._fall_seen else "downclimb"
                # Only "fall" is a spoken event kind; "downclimb" just closes the attempt
                # quietly (it is an AttemptSummary outcome, not one of EVENT_KINDS).
                if outcome == "fall":
                    events.append(
                        make_event("fall", state.t, {"attempt_no": self._attempt_no}, self.config)
                    )
                self._finish(state, outcome)
        else:
            self._ground_since = None
        return events


if __name__ == "__main__":
    print("AttemptTracker ready; provide ClimberState and Features frames to update().")
