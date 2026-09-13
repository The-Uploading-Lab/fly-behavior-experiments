"""Keep sensory entry identities fixed across visual, mechanical and no-input trials.

The native renewal engine stores its clocks by input entry and recreates them
when the vector length changes. Zero-rate entries keep that layout stable.
Other input entries, including duplicates, retain their original order.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np


def _encode(value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return {"object_array": _encode(value.tolist()), "shape": list(value.shape)}
        data = np.ascontiguousarray(value)
        return {"array_dtype": data.dtype.str, "shape": list(data.shape),
                "sha256": hashlib.sha256(data.tobytes()).hexdigest()}
    if isinstance(value, np.generic):
        return _encode(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"float": repr(value)}
    if isinstance(value, (list, tuple)):
        return [_encode(v) for v in value]
    if isinstance(value, dict):
        pairs = [[_encode(k), _encode(v)] for k, v in value.items()]
        return {"mapping": sorted(pairs, key=lambda pair: json.dumps(pair[0], sort_keys=True))}
    raise TypeError(f"Unsupported runtime parameter type: {type(value).__name__}")


def parameter_fingerprint(params):
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
    fields = {key: digest(_encode(value)) for key, value in vars(params).items()}
    return {"sha256": digest(fields), "field_sha256": fields,
            "scope": "actual SimParams values at CNS.run entry, including every per-neuron input regularity value; graph identity is recorded separately"}


class FixedSensoryInputs:
    def __init__(self, meta):
        types = meta["cell_type"].astype(str)
        selected = (types.isin(("JO-A", "JO-B", "LC4", "LPLC2"))
                    | types.str.startswith("LC4_") | types.str.startswith("LPLC2_"))
        population = meta.loc[selected]
        if (population.empty or not population["banc_888_id"].is_unique
                or not population["super_class"].isin(("sensory", "sensory_ascending", "visual_projection")).all()):
            raise ValueError("Expected unique canonical antennal and visual sensory populations")
        self.rows = np.flatnonzero(selected)
        self.lookup = np.full(len(meta), -1, dtype=np.int64)
        self.lookup[self.rows] = np.arange(len(self.rows))
        self.ids = meta["banc_888_id"].astype(str).to_numpy()
        self.types = types.to_numpy()
        self.base_rows = self.output_rows = None
        self.calls = 0
        self.original_lengths = set()

    def regularity(self, original, scalar):
        result = (np.full(len(self.lookup), float(scalar)) if original is None
                  else np.asarray(original, dtype=float).copy())
        if result.shape != self.lookup.shape:
            raise ValueError("Input regularity vector has the wrong graph shape")
        result[self.rows] = 1.
        return result

    def apply(self, indices, rates):
        indices, rates = np.asarray(indices, dtype=np.int64), np.asarray(rates, dtype=float)
        if indices.ndim != 1 or rates.shape != indices.shape:
            raise ValueError("Input indices and rates must be equal-length vectors")
        if indices.size and (indices.min() < 0 or indices.max() >= len(self.lookup)):
            raise ValueError("Input row is outside the canonical graph")
        positions = self.lookup[indices]
        sensory = positions >= 0
        selected = positions[sensory]
        if len(np.unique(selected)) != len(selected):
            raise ValueError("Duplicate sensory input entries require an explicit union-preserving layout")
        if not np.isfinite(rates[sensory]).all() or np.any(rates[sensory] < 0):
            raise ValueError("Sensory rates must be finite and nonnegative")
        base = indices[~sensory]
        if self.base_rows is None:
            self.base_rows = base.copy()
            self.output_rows = np.concatenate((self.base_rows, self.rows))
        elif not np.array_equal(base, self.base_rows):
            raise ValueError("Nonvisual/nonantennal input identities changed during the trial")
        values = np.zeros(len(self.rows))
        values[selected] = rates[sensory]
        self.calls += 1
        self.original_lengths.add(len(indices))
        return self.output_rows, np.concatenate((rates[~sensory], values))

    def describe(self):
        rows = self.output_rows
        return {"scope": "fixed sensory layout and shared Poisson regularity across protocols; other input entry identities retained",
                "sensory_populations": {kind: int(np.sum(self.types[self.rows] == kind)) for kind in sorted(set(self.types[self.rows]))},
                "sensory_ids": self.ids[self.rows].tolist(),
                "calls": self.calls, "original_vector_lengths": sorted(self.original_lengths),
                "fixed_vector_length": None if rows is None else len(rows),
                "input_id_sequence_sha256": None if rows is None else hashlib.sha256(
                    json.dumps(self.ids[rows].tolist()).encode()).hexdigest(),
                "sensory_regularity_shape": 1.}
