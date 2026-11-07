"""One Euro filter (Casiez et al.): low lag when moving, strong smoothing when still."""

from __future__ import annotations

import math


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self._t: float | None = None
        self._x = 0.0
        self._dx = 0.0

    def __call__(self, t: float, x: float) -> float:
        if self._t is None:
            self._t, self._x = t, x
            return x
        dt = t - self._t
        if dt <= 0:  # duplicate or out-of-order timestamp: keep the last output
            return self._x
        dx = (x - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * abs(self._dx)
        a = _alpha(cutoff, dt)
        self._x = a * x + (1 - a) * self._x
        self._t = t
        return self._x


class OneEuroPoint:
    """Smooths an (x, y) point."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self._fx = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self._fy = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def reset(self) -> None:
        self._fx.reset()
        self._fy.reset()

    def __call__(self, t: float, p: tuple[float, float]) -> tuple[float, float]:
        return self._fx(t, p[0]), self._fy(t, p[1])
