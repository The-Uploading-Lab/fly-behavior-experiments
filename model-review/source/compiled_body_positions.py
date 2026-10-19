"""Read FlyGym body positions without indexing uncompiled body IDs.

FlyGym 2.1 keeps ``c_head`` in ``Fly.get_bodysegs_order()`` after MuJoCo
fuses that static body from the compiled model.  Its internal body ID is -1;
using the complete ID vector as a NumPy index therefore returns the last body,
``rh_tarsus5``, twice.  This module is the single checked indexing path for
project code that reads MuJoCo body positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


FUSED_BODY_NAMES = frozenset({"BodySegment(name='c_head')"})


class CompiledBodyIDError(ValueError):
    """A requested body has no valid row in the compiled MuJoCo model."""


def checked_body_position_rows(
    xpos: Any,
    body_ids: Sequence[int] | np.ndarray,
    body_names: Sequence[str],
) -> np.ndarray:
    """Return ``xpos`` rows after validating every compiled body ID.

    The negative-ID check deliberately precedes conversion or indexing of
    ``xpos``.  This ordering prevents NumPy's negative-index semantics from
    turning a missing compiled body into an alias of the last physical body.
    """

    ids = np.asarray(body_ids, dtype=np.int64)
    names = tuple(str(name) for name in body_names)
    if ids.ndim != 1 or ids.shape != (len(names),):
        raise CompiledBodyIDError(
            f"body ID/name shape mismatch: {ids.shape} versus {len(names)}"
        )
    negative = np.flatnonzero(ids < 0)
    if len(negative):
        details = [f"{names[index]}={int(ids[index])}" for index in negative]
        raise CompiledBodyIDError(
            "negative compiled body IDs rejected before position indexing: "
            + ", ".join(details)
        )
    if len(np.unique(ids)) != len(ids):
        raise CompiledBodyIDError("duplicate compiled body IDs would double-count a body")

    values = np.asarray(xpos)
    if values.ndim != 2 or values.shape[1] != 3:
        raise CompiledBodyIDError(f"MuJoCo xpos shape is not (n, 3): {values.shape}")
    if len(ids) and int(ids.max()) >= values.shape[0]:
        raise CompiledBodyIDError(
            f"compiled body ID {int(ids.max())} exceeds xpos rows {values.shape[0]}"
        )
    return np.asarray(values[ids, :]).copy()


@dataclass(frozen=True)
class CompiledBodyPositionSelector:
    """The one-to-one compiled subset of a FlyGym body-segment order."""

    body_names: tuple[str, ...]
    body_ids: np.ndarray
    omitted_body_names: tuple[str, ...]

    @classmethod
    def from_simulation(
        cls, simulation: Any, fly: Any
    ) -> "CompiledBodyPositionSelector":
        all_names = tuple(str(body) for body in fly.get_bodysegs_order())
        all_ids = np.asarray(
            simulation._internal_bodyids_by_fly[fly.name], dtype=np.int64
        )
        if all_ids.ndim != 1 or all_ids.shape != (len(all_names),):
            raise CompiledBodyIDError(
                f"body ID/name shape mismatch: {all_ids.shape} versus {len(all_names)}"
            )

        missing_rows = np.flatnonzero(all_ids < 0)
        missing_names = tuple(all_names[index] for index in missing_rows)
        unexpected = [name for name in missing_names if name not in FUSED_BODY_NAMES]
        if unexpected:
            details = [
                f"{all_names[index]}={int(all_ids[index])}"
                for index in missing_rows
                if all_names[index] in unexpected
            ]
            raise CompiledBodyIDError(
                "unexpected negative compiled body IDs rejected before position "
                "indexing: " + ", ".join(details)
            )

        retained = all_ids >= 0
        body_names = tuple(
            name for name, keep in zip(all_names, retained, strict=True) if keep
        )
        body_ids = np.asarray(all_ids[retained], dtype=np.int64)
        # This shared guard remains the final operation before every xpos read.
        # The empty array has the same row count contract without touching a
        # simulation or indexing a MuJoCo array during selector construction.
        checked_body_position_rows(
            np.empty((int(body_ids.max()) + 1 if len(body_ids) else 0, 3)),
            body_ids,
            body_names,
        )
        body_ids.setflags(write=False)
        return cls(body_names, body_ids, missing_names)

    def read(self, xpos: Any) -> np.ndarray:
        """Read the retained physical bodies through the checked index path."""

        return checked_body_position_rows(xpos, self.body_ids, self.body_names)
