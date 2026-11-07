"""Fast, local MVP cue detectors for the coach brain."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import atan2, degrees, hypot
from typing import Any

from coach.brain.attempts import make_event
from coach.brain.config import cfg
from coach.brain.features import Features, height_above
from coach.contracts import ClimberState, CoachEvent, HoldMap, Limb

logger = logging.getLogger(__name__)


@dataclass
class DetectorContext:
    config: dict[str, Any]
    holdmap: HoldMap | None
    on_wall: bool
    last_fall_t: float | None
    crux_hold: int | None = None  # memory.crux_hold(); used by CruxWarning


def _brain(ctx: DetectorContext, name: str) -> Any:
    return cfg(ctx.config, "brain", name)


def _highest_contact(state: ClimberState) -> int | None:
    return max(
        (hold_id for hold_id in state.contacts.values() if hold_id is not None), default=None
    )


class Detector:
    """Base class with the common wall/mute gate and debounce machinery.

    A condition starts a pending interval on its first true frame and may fire
    only after ``debounce_ms`` of capture time.  Subclasses provide a release
    condition: an active cue remains active while that condition is true, and
    only releases after a clear excursion beyond the trigger threshold.  This
    prevents threshold chatter from producing repeated cues.
    """

    # Most detectors are live only while ON_WALL. GROUND-only detectors (e.g.
    # RoutePreview, spoken before the climb starts) set this False.
    requires_on_wall: bool = True

    def __init__(self) -> None:
        self._pending_since: float | None = None
        self._active = False
        self._last_valid = False

    def update(
        self, features: Features, state: ClimberState, ctx: DetectorContext
    ) -> list[CoachEvent]:
        if not self._allowed(features, state, ctx):
            self._pending_since = None
            return []
        return self._update(features, state, ctx)

    def _allowed(self, features: Features, state: ClimberState, ctx: DetectorContext) -> bool:
        if self.requires_on_wall != ctx.on_wall:
            return False
        mute_after = float(_brain(ctx, "cue_mute_after_fall_s"))
        if ctx.last_fall_t is not None and state.t - ctx.last_fall_t < mute_after:
            return False
        return features.hip_speed <= float(_brain(ctx, "cue_mute_hip_speed"))

    def _debounced(self, condition: bool, t: float, debounce_s: float) -> bool:
        if not condition:
            self._pending_since = None
            return False
        if self._pending_since is None:
            self._pending_since = t
        return t - self._pending_since + 1e-9 >= debounce_s

    def _one_shot(self, condition: bool, release: bool, t: float, debounce_s: float) -> bool:
        if not release:
            self._active = False
            self._pending_since = None
        if self._active:
            return False
        if self._debounced(condition, t, debounce_s):
            self._active = True
            self._pending_since = None
            return True
        return False

    def _update(
        self, features: Features, state: ClimberState, ctx: DetectorContext
    ) -> list[CoachEvent]:
        raise NotImplementedError

    def still_valid(self, event: CoachEvent, state: ClimberState) -> bool:
        return self._last_valid


class Stalling(Detector):
    def __init__(self) -> None:
        super().__init__()
        self._stage = 0
        self._episode = False

    def _update(
        self, features: Features, state: ClimberState, ctx: DetectorContext
    ) -> list[CoachEvent]:
        thresholds = [float(value) for value in _brain(ctx, "stall_s")]
        static = features.hip_static_s > 0
        self._last_valid = static
        if not static:
            self._stage = 0
            self._episode = False
            self._pending_since = None
            return []
        if not self._episode:
            self._episode = True
            self._stage = 0
        events: list[CoachEvent] = []
        debounce = float(_brain(ctx, "debounce_ms")) / 1000.0
        while self._stage < len(thresholds) and features.hip_static_s >= thresholds[self._stage]:
            # ``hip_static_s`` already records how long the raw condition has
            # held, so the threshold crossing is the beginning of debounce.
            if self._pending_since is None:
                self._pending_since = state.t - (features.hip_static_s - thresholds[self._stage])
            if self._debounced(True, state.t, debounce):
                stage = self._stage
                events.append(
                    make_event(
                        "stalling",
                        state.t,
                        {
                            "stage": stage,
                            "seconds": features.hip_static_s,
                            "hold_id": _highest_contact(state),
                        },
                        ctx.config,
                    )
                )
                self._stage += 1
                self._pending_since = None
            else:
                break
        return events


class ArmsBent(Detector):
    def _update(
        self, features: Features, state: ClimberState, ctx: DetectorContext
    ) -> list[CoachEvent]:
        limit = float(_brain(ctx, "arms_bent_deg"))
        eps = float(_brain(ctx, "move_speed_eps"))
        angles = features.elbow_angle
        bent = all(
            angles.get(side) is not None and angles[side] < limit for side in ("left", "right")
        )
        static = features.hip_speed <= eps and all(
            value <= eps for value in features.limb_speed.values()
        )
        # A 5% release band is deliberately relative to the configured trigger.
        release = all(
            value is not None and value < limit * 1.05
            for value in (angles.get("left"), angles.get("right"))
        )
        self._last_valid = bent and static
        fired = self._one_shot(
            bent and static, release and static, state.t, float(_brain(ctx, "debounce_ms")) / 1000
        )
        return [make_event("arms_bent", state.t, {}, ctx.config)] if fired else []


class FeetLow(Detector):
    def _condition(
        self, features: Features, state: ClimberState, ctx: DetectorContext
    ) -> tuple[bool, bool]:
        feet = [state.limbs.get(Limb.LF), state.limbs.get(Limb.RF)]
        shoulders = [state.keypoints.get("left_shoulder"), state.keypoints.get("right_shoulder")]
        wrists = [state.keypoints.get("left_wrist"), state.keypoints.get("right_wrist")]
        if any(point is None for point in feet) or any(
            point is None for point in shoulders + wrists
        ):
            return False, False
        if features.leg_length <= 0:
            return False, False
        lowest_foot_y = max(point[1] for point in feet if point is not None)
        low = height_above(lowest_foot_y, state.hip[1]) / features.leg_length > float(
            _brain(ctx, "feet_low_ratio")
        )
        hands_high = all(
            height_above(shoulder[1], wrist[1]) > 0
            for shoulder, wrist in zip(shoulders, wrists, strict=True)
        )
        return low and hands_high, low and hands_high

    def _update(
        self, features: Features, state: ClimberState, ctx: DetectorContext
    ) -> list[CoachEvent]:
        condition, release = self._condition(features, state, ctx)
        self._last_valid = condition
        # Release at 95% of the configured ratio, while retaining the hand gate.
        fired = self._one_shot(
            condition, release, state.t, float(_brain(ctx, "debounce_ms")) / 1000
        )
        if not fired:
            return []
        feet = {Limb.LF: state.limbs.get(Limb.LF), Limb.RF: state.limbs.get(Limb.RF)}
        trailing_foot = max(
            (limb for limb, point in feet.items() if point is not None),
            key=lambda limb: feet[limb][1],
        )
        return [
            make_event(
                "feet_low",
                state.t,
                {"hold_id": _highest_contact(state), "limb": trailing_foot.value},
                ctx.config,
            )
        ]


def direction_phrase(dx: float, dy: float) -> str:
    """Map a screen-coordinate vector to one of the eight spoken directions."""
    angle = (degrees(atan2(-dy, dx)) + 360) % 360
    phrases = (
        "right",
        "up and right",
        "up",
        "up and left",
        "left",
        "down and left",
        "down",
        "down and right",
    )
    return phrases[int((angle + 22.5) // 45) % 8]


class HandSearching(Detector):
    def _candidate(
        self, hand: Limb, features: Features, state: ClimberState, ctx: DetectorContext
    ) -> tuple[int, str] | None:
        if ctx.holdmap is None or features.arm_length <= 0:
            return None
        hand_point = state.limbs.get(hand)
        shoulder_name = "left_shoulder" if hand == Limb.LH else "right_shoulder"
        shoulder = state.keypoints.get(shoulder_name)
        if hand_point is None or shoulder is None or shoulder[2] <= 0:
            return None
        occupied = {value for value in state.contacts.values() if value is not None}
        penalty = float(_brain(ctx, "next_hold_distance_penalty"))
        choices: list[tuple[float, int, str]] = []
        for hold in ctx.holdmap.holds:
            if hold.id in occupied or hold.center[1] >= hand_point[1]:
                continue
            reach = hypot(shoulder[0] - hold.center[0], shoulder[1] - hold.center[1])
            if reach > features.arm_length:
                continue
            distance = hypot(hand_point[0] - hold.center[0], hand_point[1] - hold.center[1])
            score = height_above(hand_point[1], hold.center[1]) - penalty * distance
            choices.append(
                (
                    score,
                    hold.id,
                    direction_phrase(
                        hold.center[0] - hand_point[0], hold.center[1] - hand_point[1]
                    ),
                )
            )
        if not choices:
            return None
        _, hold_id, direction = max(choices, key=lambda item: (item[0], item[1]))
        return hold_id, direction

    def _update(
        self, features: Features, state: ClimberState, ctx: DetectorContext
    ) -> list[CoachEvent]:
        if ctx.holdmap is None:
            return []
        debounce = max(float(_brain(ctx, "debounce_ms")), float(_brain(ctx, "search_ms"))) / 1000
        events: list[CoachEvent] = []
        for hand in (Limb.LH, Limb.RH):
            searching = (
                state.contacts.get(hand) is None
                and features.hand_moving_no_contact_s.get(hand, 0.0)
                >= float(_brain(ctx, "search_ms")) / 1000
            )
            candidate = self._candidate(hand, features, state, ctx) if searching else None
            key_active = getattr(self, "_active_hand", None)
            if searching and candidate is not None and key_active != hand:
                if self._pending_since is None:
                    self._pending_since = state.t - features.hand_moving_no_contact_s.get(hand, 0.0)
                if self._debounced(True, state.t, debounce):
                    self._active_hand = hand
                    self._pending_since = None
                    events.append(
                        make_event(
                            "hand_searching",
                            state.t,
                            {"limb": hand.value, "hold_id": candidate[0], "dir": candidate[1]},
                            ctx.config,
                        )
                    )
            elif not searching or candidate is None:
                if key_active == hand:
                    self._active_hand = None
                self._pending_since = None
        self._last_valid = any(state.contacts.get(hand) is None for hand in (Limb.LH, Limb.RH))
        return events

    def still_valid(self, event: CoachEvent, state: ClimberState) -> bool:
        limb = event.data.get("limb")
        return state.contacts.get(Limb(limb)) is None if limb else False


def mvp_detectors(config: Any) -> list[Detector]:
    """Return detectors in the standard arbitration order."""
    return [Stalling(), ArmsBent(), FeetLow(), HandSearching()]


class CruxWarning(Detector):
    requires_on_wall: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._warned = False

    def _update(
        self, features: Features, state: ClimberState, ctx: DetectorContext
    ) -> list[CoachEvent]:
        if ctx.crux_hold is None:
            return []

        target = ctx.crux_hold - 1
        current_contacts = [hold_id for hold_id in state.contacts.values() if hold_id is not None]
        highest_contact = max(current_contacts) if current_contacts else None

        if highest_contact is not None and highest_contact >= target:
            if not self._warned:
                self._warned = True
                self._last_valid = True
                return [
                    make_event(
                        "crux_warning",
                        state.t,
                        {"hold_id": ctx.crux_hold, "crux_hold": ctx.crux_hold},
                        ctx.config,
                    )
                ]
        elif highest_contact is None or highest_contact < target:
            self._warned = False
            self._last_valid = False

        return []

    def still_valid(self, event: CoachEvent, state: ClimberState) -> bool:
        current_contacts = [hold_id for hold_id in state.contacts.values() if hold_id is not None]
        highest_contact = max(current_contacts) if current_contacts else None
        crux_hold = event.data.get("crux_hold") or event.data.get("hold_id")
        if crux_hold is None:
            return self._last_valid
        return highest_contact is not None and highest_contact >= crux_hold - 1


class RoutePreview(Detector):
    requires_on_wall: bool = False

    def __init__(self) -> None:
        super().__init__()
        self._fired = False

    def update(
        self, features: Features, state: ClimberState, ctx: DetectorContext
    ) -> list[CoachEvent]:
        if ctx.on_wall:
            self._fired = False
        return super().update(features, state, ctx)

    def _update(
        self, features: Features, state: ClimberState, ctx: DetectorContext
    ) -> list[CoachEvent]:
        if self._fired:
            return []
        if ctx.holdmap is None or len(ctx.holdmap.holds) < 2:
            return []
        if features.arm_length <= 0:
            return []

        ratio = float(_brain(ctx, "route_preview_ratio"))
        threshold = ratio * features.arm_length
        sorted_holds = sorted(ctx.holdmap.holds, key=lambda h: h.id)

        max_gap = -1.0
        far_hold_id: int | None = None
        for h1, h2 in zip(sorted_holds, sorted_holds[1:], strict=False):
            gap = hypot(h2.center[0] - h1.center[0], h2.center[1] - h1.center[1])
            if gap > max_gap:
                max_gap = gap
                far_hold_id = h2.id

        if max_gap > threshold and far_hold_id is not None:
            self._fired = True
            self._last_valid = True
            return [
                make_event(
                    "route_preview",
                    state.t,
                    {"hold_id": far_hold_id, "gap": max_gap},
                    ctx.config,
                )
            ]
        return []


class QuietFeet(Detector):
    requires_on_wall: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._last_hold: dict[Limb, int | None] = {Limb.LF: None, Limb.RF: None}
        self._left_t: dict[Limb, float | None] = {Limb.LF: None, Limb.RF: None}
        self._prev_contacts: dict[Limb, int | None] = {}

    def _update(
        self, features: Features, state: ClimberState, ctx: DetectorContext
    ) -> list[CoachEvent]:
        window_s = float(_brain(ctx, "quiet_feet_window_s"))
        events: list[CoachEvent] = []

        for foot in (Limb.LF, Limb.RF):
            current = state.contacts.get(foot)
            prev = self._prev_contacts.get(foot)

            if prev is not None and current != prev:
                self._last_hold[foot] = prev
                self._left_t[foot] = state.t

            if current is not None and current != prev:
                left_t = self._left_t[foot]
                if (
                    self._last_hold[foot] == current
                    and left_t is not None
                    and (state.t - left_t) <= window_s
                ):
                    events.append(
                        make_event(
                            "quiet_feet",
                            state.t,
                            {"limb": foot.value, "hold_id": current},
                            ctx.config,
                        )
                    )
                self._last_hold[foot] = None
                self._left_t[foot] = None

            self._prev_contacts[foot] = current

        self._last_valid = len(events) > 0
        return events

    def still_valid(self, event: CoachEvent, state: ClimberState) -> bool:
        limb_val = event.data.get("limb")
        hold_id = event.data.get("hold_id")
        if not limb_val:
            return False
        return state.contacts.get(Limb(limb_val)) == hold_id


class FeetCut(Detector):
    requires_on_wall: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._cut_since: float | None = None
        self._fired = False

    def _update(
        self, features: Features, state: ClimberState, ctx: DetectorContext
    ) -> list[CoachEvent]:
        feet_off = state.contacts.get(Limb.LF) is None and state.contacts.get(Limb.RF) is None
        hands_on = (
            state.contacts.get(Limb.LH) is not None and state.contacts.get(Limb.RH) is not None
        )

        if not feet_off:
            self._cut_since = None
            self._fired = False
            self._last_valid = False
            return []

        if hands_on:
            if self._cut_since is None:
                self._cut_since = state.t
            feet_cut_s = float(_brain(ctx, "feet_cut_ms")) / 1000.0
            if (state.t - self._cut_since + 1e-9) >= feet_cut_s and not self._fired:
                self._fired = True
                self._last_valid = True
                return [
                    make_event(
                        "feet_cut",
                        state.t,
                        {"hold_id": _highest_contact(state)},
                        ctx.config,
                    )
                ]
        else:
            self._cut_since = None

        self._last_valid = self._fired and feet_off
        return []

    def still_valid(self, event: CoachEvent, state: ClimberState) -> bool:
        return state.contacts.get(Limb.LF) is None and state.contacts.get(Limb.RF) is None


class WeightShift(Detector):
    requires_on_wall: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._active_foot: Limb | None = None

    def _update(
        self, features: Features, state: ClimberState, ctx: DetectorContext
    ) -> list[CoachEvent]:
        eps = float(_brain(ctx, "move_speed_eps"))
        static = features.hip_speed <= eps
        if not static or features.leg_length <= 0:
            self._one_shot(False, False, state.t, 0)
            self._active_foot = None
            self._last_valid = False
            return []

        lf_on = state.contacts.get(Limb.LF) is not None
        rf_on = state.contacts.get(Limb.RF) is not None
        if lf_on == rf_on:
            self._one_shot(False, False, state.t, 0)
            self._active_foot = None
            self._last_valid = False
            return []

        supporting_foot = Limb.LF if lf_on else Limb.RF
        foot_point = state.limbs.get(supporting_foot)
        if foot_point is None:
            self._one_shot(False, False, state.t, 0)
            self._active_foot = None
            self._last_valid = False
            return []

        if self._active_foot != supporting_foot:
            self._active = False
            self._pending_since = None
            self._active_foot = supporting_foot

        dx = abs(state.hip[0] - foot_point[0])
        ratio = float(_brain(ctx, "weight_shift_ratio"))
        threshold = ratio * features.leg_length

        condition = dx > threshold
        release = dx > (threshold * 0.9)

        debounce = float(_brain(ctx, "debounce_ms")) / 1000.0
        self._last_valid = condition and static
        fired = self._one_shot(condition and static, release and static, state.t, debounce)
        if fired:
            return [
                make_event(
                    "weight_shift",
                    state.t,
                    {
                        "foot": supporting_foot.value,
                        "hold_id": state.contacts.get(supporting_foot),
                    },
                    ctx.config,
                )
            ]
        return []

    def still_valid(self, event: CoachEvent, state: ClimberState) -> bool:
        foot_str = event.data.get("foot")
        if not foot_str:
            return False
        return state.contacts.get(Limb(foot_str)) is not None


def good_tier_detectors(config: Any) -> list[Detector]:
    """Return the good-tier detector set."""
    return [CruxWarning(), RoutePreview(), QuietFeet(), FeetCut(), WeightShift()]


if __name__ == "__main__":
    logger.info(
        "detectors ready: mvp=%s good=%s",
        [type(d).__name__ for d in mvp_detectors({})],
        [type(d).__name__ for d in good_tier_detectors({})],
    )
