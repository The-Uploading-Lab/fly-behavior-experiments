"""Bound the final wing stroke by a shared cubic amplitude budget.

At fixed frequency, sum(amplitude**3) is a power proxy, not computed
aerodynamic work. Gordon & Dickinson 2006, doi:10.1073/pnas.0510109103,
motivates the exponent; the bilateral allocation here is a hypothesis.
"""
from __future__ import annotations

import math


def limit_stroke_amplitudes(requested, available):
    """Only reduce excess amplitude, preserving the positive left/right ratio."""
    if len(requested) != 2 or len(available) != 2:
        raise ValueError("Expected left and right wing amplitudes")
    if any(not math.isfinite(value) for value in (*requested, *available)):
        raise ValueError("Wing amplitudes must be finite")
    raw = tuple(max(0., float(value)) for value in requested)
    supply = tuple(max(0., float(value)) for value in available)
    demand = sum(value**3 for value in raw)
    budget = sum(value**3 for value in supply)
    scale = (budget / demand)**(1./3.) if demand > budget else 1.
    result = tuple(value * scale for value in raw)
    return result, {"requested_amplitude": list(requested),
                    "available_amplitude": list(supply), "scale": scale,
                    "requested_cubic_sum": demand, "budget_cubic_sum": budget}
