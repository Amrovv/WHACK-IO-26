"""Which hold is each hand/foot on?

A limb is ON a hold when its point is inside the hold's shape expanded by
`contact_margin x arm_length` AND the limb has been nearly still for `contact_settle_ms`.
The shape is the hold's polygon if it has one (diagonal holds), otherwise its box. It is
released only once the point leaves a bigger zone (`contact_release_ratio` x that margin), so
jitter at the edge doesn't flicker. If two holds qualify, the one the point is closest to wins
(0 if it is on the shape itself), then the nearest centre. A limb whose landmarks are not
visible enough has no contact.

All coordinates are normalised 0-1 (y down); times come from `state.t`, never the wall clock.
"""

from __future__ import annotations

import logging
from collections import deque
from statistics import median

from coach.contracts import ClimberState, Hold, HoldMap, Limb, Point
from coach.perception.shapes import Polygon, distance_to_polygon

log = logging.getLogger(__name__)

# Landmarks whose visibility decides whether a limb point can be trusted.
LIMB_LANDMARKS = {
    Limb.LH: ("left_wrist", "left_index"),
    Limb.RH: ("right_wrist", "right_index"),
    Limb.LF: ("left_heel", "left_foot_index"),
    Limb.RF: ("right_heel", "right_foot_index"),
}
ARMS = (
    ("left_shoulder", "left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow", "right_wrist"),
)


def _dist(a: Point, b: Point) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _inside(p: Point, bbox: tuple[float, float, float, float], margin: float) -> bool:
    x0, y0, x1, y1 = bbox
    return x0 - margin <= p[0] <= x1 + margin and y0 - margin <= p[1] <= y1 + margin


def _distance_to_box(p: Point, bbox: tuple[float, float, float, float]) -> float:
    """0 if the point is inside the box, otherwise the distance to it."""
    x0, y0, x1, y1 = bbox
    return _dist(p, (min(max(p[0], x0), x1), min(max(p[1], y0), y1)))


class ContactTracker:
    def __init__(
        self, cfg: dict, holdmap: HoldMap, shapes: dict[int, Polygon] | None = None
    ) -> None:
        self._cfg = cfg
        self._shapes = shapes or {}  # hold id -> polygon (diagonal holds only)
        self._holds: list[Hold] = holdmap.holds
        self._by_id = {h.id: h for h in self._holds}
        self._arm_samples: deque[float] = deque(maxlen=cfg["arm_length_frames"])
        self._lost_frames = 0
        self.reset()

    def reset(self) -> None:
        self._on: dict[Limb, int | None] = {limb: None for limb in Limb}
        self._prev: dict[Limb, tuple[float, Point]] = {}
        self._still_since: dict[Limb, float | None] = {limb: None for limb in Limb}
        self._low_vis: dict[Limb, int] = {limb: 0 for limb in Limb}  # consecutive low-vis frames

    # -- arm length ---------------------------------------------------------------------------
    def _update_arm_length(self, kp: dict) -> None:
        min_vis = self._cfg["min_visibility"]
        for shoulder, elbow, wrist in ARMS:
            if all(k in kp and kp[k][2] >= min_vis for k in (shoulder, elbow, wrist)):
                self._arm_samples.append(
                    _dist(kp[shoulder], kp[elbow]) + _dist(kp[elbow], kp[wrist])
                )

    @property
    def arm_length(self) -> float:
        return median(self._arm_samples) if self._arm_samples else self._cfg["arm_length_default"]

    # -- per frame ----------------------------------------------------------------------------
    def update(self, state: ClimberState) -> dict[Limb, int | None]:
        if not state.person_found:
            # A brief detection gap keeps the hold memory (so a limb that is still inside its hold
            # needn't settle again); a long one clears it. Contacts are None while nobody is found.
            self._lost_frames += 1
            if self._lost_frames > self._cfg["lost_memory_frames"]:
                self.reset()
            return {limb: None for limb in Limb}
        self._lost_frames = 0
        self._update_arm_length(state.keypoints)
        margin = self._cfg["contact_margin"] * self.arm_length
        release_margin = margin * self._cfg["contact_release_ratio"]
        settle_s = self._cfg["contact_settle_ms"] / 1000

        contacts: dict[Limb, int | None] = {}
        for limb in Limb:
            point = state.limbs.get(limb)
            if point is None:
                # PoseTracker uses None after its short frozen-point grace period. Unlike a
                # raw visibility dip, this is an explicit unusable-point signal: do not extend
                # the contact-dip grace a second time.
                self._on[limb], self._still_since[limb] = None, None
                self._low_vis[limb] = 0
                self._prev.pop(limb, None)
                contacts[limb] = None
                continue
            if not self._visible(state.keypoints, limb):
                # No contact is reported while the limb is hidden, but a brief dip keeps the hold
                # memory so the limb needn't settle again; a long one clears it.
                self._low_vis[limb] += 1
                if self._low_vis[limb] > self._cfg["lost_memory_frames"]:
                    self._on[limb], self._still_since[limb] = None, None
                    self._prev.pop(limb, None)
                # A one-or-two-frame visibility dip keeps the contact the limb already had (the
                # point is frozen meanwhile); a longer one reports no contact.
                dip = self._low_vis[limb] <= self._cfg.get("contact_dip_frames", 0)
                contacts[limb] = self._on[limb] if dip else None
                continue
            self._low_vis[limb] = 0
            self._update_stillness(limb, state.t, point)
            held = self._by_id.get(self._on[limb]) if self._on[limb] is not None else None
            if held is not None and self._reach(held, point, release_margin) is not None:
                # hysteresis: stay on until clearly outside, unless the point is now ON another
                # hold while only near this one (an overlapping neighbour must not keep it)
                if self._reach(held, point, 0.0) is None:
                    inside = self._holds_containing(point)
                    if inside is not None:
                        self._on[limb] = contacts[limb] = inside
                        continue
                contacts[limb] = held.id
                continue
            self._on[limb] = None
            since = self._still_since[limb]
            if since is not None and state.t - since >= settle_s:
                self._on[limb] = self._nearest_hold(point, margin)
            contacts[limb] = self._on[limb]
        return contacts

    def _visible(self, kp: dict, limb: Limb) -> bool:
        min_vis = self._cfg["min_visibility"]
        return all(name in kp and kp[name][2] >= min_vis for name in LIMB_LANDMARKS[limb])

    def _update_stillness(self, limb: Limb, t: float, point: Point) -> None:
        prev = self._prev.get(limb)
        self._prev[limb] = (t, point)
        if prev is None:
            self._still_since[limb] = t  # first sighting: start the settle timer
            return
        dt = t - prev[0]
        if dt <= 0:
            return
        if _dist(point, prev[1]) / dt >= self._cfg["contact_still_speed"]:
            self._still_since[limb] = None  # moving: restart
        elif self._still_since[limb] is None:
            self._still_since[limb] = t

    def _reach(self, hold: Hold, point: Point, margin: float) -> float | None:
        """How far the point is from the hold's shape (0 = on it), or None if it is further than
        `margin`. A polygon is measured with real distance; a plain box keeps the square margin."""
        shape = self._shapes.get(hold.id)
        if shape is not None:
            d = distance_to_polygon(point, shape)
            return d if d <= margin else None
        if not _inside(point, hold.bbox, margin):
            return None
        return _distance_to_box(point, hold.bbox)

    def _holds_containing(self, point: Point) -> int | None:
        """The polygon hold the point is on (nearest centre if several), else None. Only polygons
        count: a plain box is a loose approximation, so being inside one does not take a limb away
        from the hold it is already on."""
        on = [
            (_dist(point, h.center), h.id)
            for h in self._holds
            if h.id in self._shapes and self._reach(h, point, 0.0) == 0.0
        ]
        return min(on)[1] if on else None

    def _nearest_hold(self, point: Point, margin: float) -> int | None:
        reached = []
        for hold in self._holds:
            d = self._reach(hold, point, margin)
            if d is not None:
                reached.append((d, _dist(point, hold.center), hold.id))
        return min(reached)[2] if reached else None
