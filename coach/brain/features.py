"""Fast, local geometric features derived from :class:`ClimberState`."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import atan2, degrees, hypot
from statistics import median

from coach.contracts import ClimberState, Limb


def height_above(lower_y: float, upper_y: float) -> float:
    """Return vertical height from ``lower_y`` to ``upper_y`` (y increases down)."""
    return lower_y - upper_y


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def _keypoint(state: ClimberState, name: str) -> tuple[float, float] | None:
    value = state.keypoints.get(name)
    if value is None or len(value) < 3 or value[2] < 0.5:
        return None
    return value[0], value[1]


def _chain_length(state: ClimberState, names: tuple[str, ...]) -> float | None:
    points = [_keypoint(state, name) for name in names]
    if any(point is None for point in points):
        return None
    return sum(_distance(a, b) for a, b in zip(points, points[1:], strict=False))  # type: ignore[arg-type]


def _elbow_angle(state: ClimberState, side: str) -> float | None:
    points = [_keypoint(state, f"{side}_{part}") for part in ("shoulder", "elbow", "wrist")]
    if any(point is None for point in points):
        return None
    shoulder, elbow, wrist = points  # type: ignore[misc]
    first = (shoulder[0] - elbow[0], shoulder[1] - elbow[1])
    second = (wrist[0] - elbow[0], wrist[1] - elbow[1])
    return abs(
        degrees(
            atan2(
                first[0] * second[1] - first[1] * second[0],
                first[0] * second[0] + first[1] * second[1],
            )
        )
    )


@dataclass
class Features:
    limb_speed: dict[Limb, float] = field(default_factory=dict)
    elbow_angle: dict[str, float | None] = field(default_factory=dict)
    hip_velocity_y: float = 0.0
    hip_speed: float = 0.0
    hip_static_s: float = 0.0
    arm_length: float = 0.0
    leg_length: float = 0.0
    highest_hold_id: int | None = None
    hand_moving_no_contact_s: dict[Limb, float] = field(default_factory=dict)
    # Contacts that have held for `contact_stable_s` (a one-frame flicker doesn't change them).
    # Empty when the tracker has none yet: detectors then use the state's raw contacts.
    stable_contacts: dict[Limb, int | None] = field(default_factory=dict)
    _calibrate: Callable[[ClimberState], None] | None = field(default=None, repr=False)

    def contacts(self, state: ClimberState) -> dict[Limb, int | None]:
        """The steady contacts when known, else the state's raw ones."""
        return self.stable_contacts or state.contacts

    def add_calibration_sample(self, state: ClimberState) -> None:
        """Forward a standing sample to the tracker that produced this snapshot."""
        if self._calibrate is not None:
            self._calibrate(state)


class FeatureTracker:
    """Keep the small amount of history needed by the attempt and cue logic.

    The cue detectors need signals that are steady over time, like attempt detection's:
      * `hip_static_s`: how long the hip has stayed within `static_radius_ratio` x leg length of
        where it settled. A frame-to-frame speed test reset on every pixel of landmark jitter, so
        a 12 s stall never added up and arms_bent / weight_shift (which need a still hip) rarely
        fired;
      * `stable_contacts`: a contact change counts once it has held for `contact_stable_s`;
      * `hand_moving_no_contact_s`: how long a hand has been off any hold and up above its
        shoulder (reaching), not reset by a hand pausing as it pats around for the next hold.
    """

    def __init__(
        self,
        move_speed_eps: float = 0.02,
        static_radius_ratio: float = 0.12,
        contact_stable_s: float = 0.25,
        leg_length_default: float = 0.3,
    ) -> None:
        self.move_speed_eps = float(move_speed_eps)
        self.static_radius_ratio = float(static_radius_ratio)
        self.contact_stable_s = float(contact_stable_s)
        self.leg_length_default = float(leg_length_default)
        self._previous: ClimberState | None = None
        self._hip_static_s = 0.0
        self._hip_anchor: tuple[float, float] | None = None
        self._stable: dict[Limb, int | None] | None = None
        self._pending_contact: dict[Limb, tuple[int | None, float]] = {}
        self._hand_search_s = {Limb.LH: 0.0, Limb.RH: 0.0}
        self._highest_hold_id: int | None = None
        self._arm_samples: list[float] = []
        self._leg_samples: list[float] = []

    def add_calibration_sample(self, state: ClimberState) -> None:
        arm = _chain_length(state, ("left_shoulder", "left_elbow", "left_wrist"))
        if arm is None:
            arm = _chain_length(state, ("right_shoulder", "right_elbow", "right_wrist"))
        leg = _chain_length(state, ("left_hip", "left_knee", "left_ankle"))
        if leg is None:
            leg = _chain_length(state, ("right_hip", "right_knee", "right_ankle"))
        if arm is not None:
            self._arm_samples.append(arm)
        if leg is not None:
            self._leg_samples.append(leg)

    def reset_highest_hold(self) -> None:
        """Reset the highest hold tracker at the start of a new attempt."""
        self._highest_hold_id = None

    def update(self, state: ClimberState) -> Features:
        previous = self._previous
        dt = max(0.0, state.t - previous.t) if previous is not None else 0.0
        speeds = {limb: 0.0 for limb in Limb}
        if previous is not None and dt > 0:
            for limb, point in state.limbs.items():
                old = previous.limbs.get(limb)
                if old is not None:
                    speeds[limb] = _distance(point, old) / dt

        # A person reappearing after a dropout (person_found False -> True) is not
        # continuous motion: pose.py freezes the hip at its last known point while
        # nobody is found, so the next real detection (possibly a different person
        # entirely) can jump far away. Treating that jump as hip velocity/speed has
        # mistaken a person just entering the frame for a fall.
        continuous = previous is not None and previous.person_found and state.person_found
        hip_velocity = 0.0
        if continuous and dt > 0:
            hip_velocity = (state.hip[1] - previous.hip[1]) / dt
        hip_speed = 0.0
        if continuous and dt > 0:
            hip_speed = _distance(state.hip, previous.hip) / dt

        current_leg = _chain_length(state, ("left_hip", "left_knee", "left_ankle"))
        if current_leg is None:
            current_leg = _chain_length(state, ("right_hip", "right_knee", "right_ankle"))
        leg = median(self._leg_samples) if self._leg_samples else (current_leg or 0.0)

        # Still while the hip stays near where it settled (jitter and slow drift stay inside).
        radius = self.static_radius_ratio * (leg or self.leg_length_default)
        if not continuous or self._hip_anchor is None:
            self._hip_anchor, self._hip_static_s = state.hip, 0.0
        elif _distance(state.hip, self._hip_anchor) > radius:
            self._hip_anchor, self._hip_static_s = state.hip, 0.0
        elif dt > 0:
            self._hip_static_s += dt

        stable = self._steady_contacts(state)
        for hand in (Limb.LH, Limb.RH):
            side = "left" if hand == Limb.LH else "right"
            point, shoulder = state.limbs.get(hand), _keypoint(state, f"{side}_shoulder")
            reaching = point is not None and shoulder is not None and point[1] < shoulder[1]
            if previous is not None and dt > 0 and stable.get(hand) is None and reaching:
                self._hand_search_s[hand] += dt
            else:
                self._hand_search_s[hand] = 0.0

        for hold_id in state.contacts.values():
            if hold_id is not None and (
                self._highest_hold_id is None or hold_id > self._highest_hold_id
            ):
                self._highest_hold_id = hold_id

        current_arm = _chain_length(state, ("left_shoulder", "left_elbow", "left_wrist"))
        if current_arm is None:
            current_arm = _chain_length(state, ("right_shoulder", "right_elbow", "right_wrist"))
        self._previous = state
        return Features(
            limb_speed=speeds,
            elbow_angle={
                "left": _elbow_angle(state, "left"),
                "right": _elbow_angle(state, "right"),
            },
            hip_velocity_y=hip_velocity,
            hip_speed=hip_speed,
            hip_static_s=self._hip_static_s,
            arm_length=median(self._arm_samples) if self._arm_samples else (current_arm or 0.0),
            leg_length=leg,
            highest_hold_id=self._highest_hold_id,
            hand_moving_no_contact_s=dict(self._hand_search_s),
            stable_contacts=dict(stable),
            _calibrate=self.add_calibration_sample,
        )

    def _steady_contacts(self, state: ClimberState) -> dict[Limb, int | None]:
        """Each limb's contact, changing only once a new value has held `contact_stable_s`."""
        if self._stable is None:  # the first state: nothing to be steady against yet
            self._stable = {limb: state.contacts.get(limb) for limb in Limb}
            return self._stable
        for limb in Limb:
            raw = state.contacts.get(limb)
            if raw == self._stable[limb]:
                self._pending_contact.pop(limb, None)
                continue
            pending = self._pending_contact.get(limb)
            if pending is None or pending[0] != raw:
                self._pending_contact[limb] = (raw, state.t)
            elif state.t - pending[1] >= self.contact_stable_s:
                self._stable[limb] = raw
                self._pending_contact.pop(limb)
        return self._stable


if __name__ == "__main__":
    print("FeatureTracker ready; provide ClimberState frames to update().")
