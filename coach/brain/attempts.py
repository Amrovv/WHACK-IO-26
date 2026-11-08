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
        self._await_ground = False

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

    def _top_ids(self) -> set[int]:
        if self.holdmap is None or not self.holdmap.holds:
            return set()
        top_ids = {hold.id for hold in self.holdmap.holds if hold.role == "top"}
        return top_ids or {max(hold.id for hold in self.holdmap.holds)}

    def _is_top(self, state: ClimberState) -> bool:
        hand_ids = [state.contacts.get(Limb.LH), state.contacts.get(Limb.RH)]
        if any(value is None for value in hand_ids):
            return False
        top_ids = self._top_ids()
        if not top_ids:
            top_id = max(value for value in hand_ids if value is not None)
            return hand_ids[0] == hand_ids[1] == top_id
        return all(value in top_ids for value in hand_ids)

    def _hands_match_top(self, state: ClimberState) -> bool:
        """Allow a top match when pose points straddle a detector's tight box.

        Contact tracking is deliberately conservative for ordinary coaching cues.  At the
        finish, however, a hand can be visibly on a large top hold while its smoothed point
        lands just outside Roboflow's axis-aligned box.  Treat both hands as matching when
        they are inside the configured expanded top-hold region, but require one confirmed
        top contact so a climber hovering near the top cannot send from the second-to-last
        hold.
        """
        top_ids = self._top_ids()
        if not top_ids or self.holdmap is None:
            return False
        hand_contacts = (state.contacts.get(Limb.LH), state.contacts.get(Limb.RH))
        if any(value is not None and value not in top_ids for value in hand_contacts):
            return False
        margin = float(cfg(self.config, "brain", "top_hand_margin"))
        top_boxes = [hold.bbox for hold in self.holdmap.holds if hold.id in top_ids]
        for limb in (Limb.LH, Limb.RH):
            point = state.limbs.get(limb)
            if point is None:
                return False
            x, y = point
            if not any(
                x0 - margin <= x <= x1 + margin and y0 - margin <= y <= y1 + margin
                for x0, y0, x1, y1 in top_boxes
            ):
                return False
        return True

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
        self._await_ground = True
        if self._attempt_high is not None:
            self._best_hold = max(self._best_hold, self._attempt_high)
        return summary

    def update(self, state: ClimberState, features: Features) -> list[CoachEvent]:
        events: list[CoachEvent] = []
        if not state.person_found:
            return events
        if self.phase == "GROUND":
            if self._await_ground:
                if not self._grounded(state, features):
                    return events
                self._await_ground = False
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
            # Only a prior *finished* attempt gives a real best to beat; on the very
            # first attempt every new contact is trivially "further than ever" and
            # would spam a high_point on almost every hold grabbed.
            if (
                self.finished_attempts
                and current_high > self._best_hold
                and current_high > self._last_record_emitted
            ):
                self._last_record_emitted = current_high
                events.append(
                    make_event("high_point", state.t, {"hold_id": current_high}, self.config)
                )

        fall_speed = float(cfg(self.config, "brain", "fall_speed"))
        # A two-hand match on the top hold is an immediate send.  Do not wait for a
        # dwell timer: on the demo footage the climber matches, then jumps down before
        # the next frame can satisfy a one-second timer.  A frozen/hidden-limb contact
        # can still report both hands after release, so keep the velocity guard.
        immediate_top = float(cfg(self.config, "brain", "top_hold_s")) <= 0
        if (self._is_top(state) or (immediate_top and self._hands_match_top(state))) and (
            features.hip_velocity_y <= fall_speed
        ):
            self._top_since = self._top_since or state.t
            if state.t - self._top_since >= float(cfg(self.config, "brain", "top_hold_s")):
                events.append(
                    make_event("top", state.t, {"attempt_no": self._attempt_no}, self.config)
                )
                self._finish(state, "top")
                return events
        else:
            self._top_since = None

        if self._grounded(state, features):
            self._ground_since = self._ground_since or state.t
            # Only the velocity during the actual trip to the ground decides "fall" vs
            # "downclimb" — a fast move made earlier while still gripping the wall (e.g.
            # a dynamic bump) must not poison a controlled downclimb at the end.
            if features.hip_velocity_y > fall_speed:
                self._fall_seen = True
            if state.t - self._ground_since >= float(cfg(self.config, "brain", "ground_settle_s")):
                outcome = "fall" if self._fall_seen else "downclimb"
                # Only "fall" is a spoken event kind; "downclimb" just closes the attempt
                # quietly (it is an AttemptSummary outcome, not one of EVENT_KINDS).
                if outcome == "fall":
                    events.append(
                        make_event(
                            "fall",
                            state.t,
                            {"attempt_no": self._attempt_no, "hold_id": self._attempt_high},
                            self.config,
                        )
                    )
                self._finish(state, outcome)
                # Fall and downclimb both finish on a grounded frame, so the
                # post-top wall latch is already satisfied for these outcomes.
                self._await_ground = False
        else:
            self._ground_since = None
            self._fall_seen = False
        return events


if __name__ == "__main__":
    print("AttemptTracker ready; provide ClimberState and Features frames to update().")
