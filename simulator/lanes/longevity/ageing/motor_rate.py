"""Count-preserving moving rates for the slow power and fast steering paths."""
from collections import deque

import numpy as np


class MotorRateWindow:
    def __init__(self, window_ms, tick_ms):
        ratio = float(window_ms) / float(tick_ms)
        if not np.isfinite(ratio) or ratio < 1 or abs(ratio - round(ratio)) > 1e-8:
            raise ValueError("Rate window must be a positive integer number of ticks")
        self.limit, self.tick_s = round(ratio), float(tick_ms) * .001
        self.queue, self.total = deque(), None

    def push(self, counts):
        counts = np.asarray(counts, dtype=float).copy()
        if self.total is None:
            self.total = np.zeros_like(counts)
        elif counts.shape != self.total.shape:
            raise ValueError("Motor count shape changed")
        if len(self.queue) == self.limit:
            self.total -= self.queue.popleft()
        self.queue.append(counts)
        self.total += counts
        return self.total / (len(self.queue) * self.tick_s)


class MotorReference:
    """Freeze a pre-stimulus operating point, then express motor deviations.

    HYPOTHESIS, following FL3c's existing adapter: steering is a deviation
    from tonic activity; each power pool is scaled to the 5 Hz calibration.
    Raw neuronal rates remain separate from these effective muscle inputs.
    """
    def __init__(self, interval_ms, tick_ms, power_mask, sides, anchor_hz=5.):
        self.start, self.end = map(float, interval_ms)
        if not 0 <= self.start < self.end < 600:
            raise ValueError("Reference must finish before the 600 ms stimulus")
        self.tick_s, self.anchor = float(tick_ms)*.001, float(anchor_hz)
        self.power = np.asarray(power_mask, dtype=bool)
        self.sides = np.asarray(sides)
        self.total = np.zeros(len(self.power))
        self.ticks, self.baseline, self.scale = 0, None, np.ones(len(self.power))

    def observe(self, time_ms, counts):
        if self.baseline is not None:
            return
        if self.start <= time_ms < self.end:
            self.total += np.asarray(counts)
            self.ticks += 1
        elif time_ms >= self.end:
            if not self.ticks:
                raise ValueError("No motor reference samples")
            self.baseline = self.total/(self.ticks*self.tick_s)
            for side in ("left", "right"):
                mask = self.power & (self.sides == side)
                if not mask.any() or self.baseline[mask].mean() <= 0:
                    raise ValueError("Power reference must contain spikes on each side")
                self.scale[mask] = self.anchor/self.baseline[mask].mean()

    def apply(self, rates):
        if self.baseline is None:
            raise ValueError("Motor reference is not ready")
        rates = np.asarray(rates, dtype=float)
        return np.where(self.power, rates*self.scale, rates-self.baseline)

    def describe(self):
        return {"interval_ms": [self.start, self.end], "ticks": self.ticks,
                "baseline_hz_by_cell": None if self.baseline is None else self.baseline.tolist(),
                "power_scale_by_cell": self.scale.tolist(),
                "scope": "hypothesized operating-point calibration; does not correct raw motor firing"}
