"""Fixed antennal input for paired full-CNS puff and sham protocols.

The JO-A/JO-B population is anatomical. The 60 Hz low-input and delivered
450 Hz pulses last 30 ms. Both are model hypotheses, not calibrated
conversions from hand velocity.
Both protocols return the same row list, including zero-rate JO rows, so
turning the pulse off does not change the random-vector length.
"""
from __future__ import annotations

import numpy as np


class AntennalPulse:
    def __init__(self, meta, *, sham=False, onset_ms=600.0, rate_hz=60.0):
        if rate_hz not in (60.0, 450.0):
            raise ValueError("Use a declared 60 or 450 Hz input")
        types = meta["cell_type"].astype(str).to_numpy()
        self.rows = np.flatnonzero(np.isin(types, ("JO-A", "JO-B")))
        if not len(self.rows):
            raise ValueError("No JO-A/JO-B cells in the model")
        selected = meta.iloc[self.rows]
        if not selected["banc_888_id"].is_unique:
            raise ValueError("Duplicate antennal neuron identifiers")
        if not selected["super_class"].isin(("sensory", "sensory_ascending")).all():
            raise ValueError("Antennal input contains a non-sensory cell")
        self.membership = np.zeros(len(meta), dtype=bool)
        self.membership[self.rows] = True
        self.onset_ms, self.duration_ms = float(onset_ms), 30.0
        self.rate_hz = 0.0 if sham else float(rate_hz)
        self.sham = bool(sham)
        self.ids = selected["banc_888_id"].astype(str).tolist()
        self.populations = {kind: int((types[self.rows] == kind).sum())
                            for kind in ("JO-A", "JO-B")}

    def apply(self, t_ms, idx, rates):
        idx = np.asarray(idx, dtype=np.int64)
        rates = np.asarray(rates, dtype=float)
        keep = ~self.membership[idx]
        on = self.onset_ms <= t_ms < self.onset_ms + self.duration_ms
        return (np.concatenate((idx[keep], self.rows)),
                np.concatenate((rates[keep], np.full(len(self.rows), self.rate_hz if on else 0.0))))

    def describe(self):
        return {"kind": "antennal_puff", "sham": self.sham,
                "onset_ms": self.onset_ms, "duration_ms": self.duration_ms,
                "rate_hz": self.rate_hz, "populations": self.populations,
                "banc_888_ids": self.ids,
                "scope": "hypothesized antennal sensory pulse; no GF forcing; no physical hand-to-rate calibration"}
