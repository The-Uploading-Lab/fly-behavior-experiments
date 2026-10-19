"""The four default-off controls in accepted study 4b4112bf.

Controller state uses the time of the recorded sample, after its physics tick.
The reductor intervention changes only named matrix signs; stiffness eligibility
comes from the original matrix. These are model interventions, not anatomy.
"""
from __future__ import annotations

import hashlib

import numpy as np

DRIVE_LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
REDUCTOR_NEGATIVE_IDS = (
    "720575941469052535", "720575941508205185", "720575941563973174",
    "720575941637140597", "720575941637141109", "720575941632094880",
    "720575941502759767", "720575941649909489", "720575941566924135",
    "720575941553597502", "720575941415175196", "720575941559314989",
    "720575941513385306",
)


class AdhesionState:
    """One hysteresis, minimum-dwell, or actuator-lag intervention per run."""

    def __init__(self, *, release_ratio=1.0, dwell_ms=0.0, tau_ms=0.0):
        values = np.asarray([release_ratio, dwell_ms, tau_ms], dtype=float)
        if not np.isfinite(values).all() or release_ratio < 1 or min(dwell_ms, tau_ms) < 0:
            raise ValueError("invalid adhesion ratio, dwell or time constant")
        if sum((release_ratio != 1, dwell_ms != 0, tau_ms != 0)) > 1:
            raise ValueError("the fall study permits one adhesion intervention per arm")
        self.release_ratio, self.dwell_ms, self.tau_ms = values
        self.state = None
        self.entered = np.zeros(6)
        self.command = None

    def gate(self, mean_flex, thresholds, row_ms):
        raw = np.asarray(mean_flex) <= thresholds
        if self.state is None:
            self.state = raw.copy()
        if self.release_ratio != 1:
            self.state = np.asarray(mean_flex) <= np.where(
                self.state, self.release_ratio * thresholds, thresholds)
        elif self.dwell_ms:
            switch = (raw != self.state) & (row_ms - self.entered >= self.dwell_ms)
            self.state[switch] = raw[switch]
            self.entered[switch] = row_ms
        else:
            self.state = raw.copy()
        return self.state.copy()

    def apply(self, requested, tick_ms):
        if not self.tau_ms:
            return requested
        if self.command is None:
            self.command = np.asarray(requested, dtype=float).copy()
        else:
            alpha = 1.0 - np.exp(-tick_ms / self.tau_ms)
            self.command += (requested - self.command) * alpha
        return self.command.copy()


def split_reductors(matrix, mn_ids, negative_ids):
    """Copy and sign-flip the 13 registered effective CTr-roll entries only."""
    import motormap as MM
    import muscle_split as MS

    requested = [str(x) for x in negative_ids]
    if len(requested) != 13 or set(requested) != set(REDUCTOR_NEGATIVE_IDS):
        raise ValueError("reductor_split_ids differs from accepted design f84249e5")
    ids = [str(x) for x in mn_ids]
    if len(set(ids)) != len(ids) or matrix.shape != (391, 42):
        raise ValueError("the registered reductor map is 391 unique MNs by 42 DOFs")
    columns = [MM.DOF_INDEX[f"{leg}_CTr_roll"] for leg in DRIVE_LEGS]
    if np.any(matrix[:, columns] < 0):
        raise ValueError("the wm1 reference CTr-roll map must be single-signed")
    counts = np.count_nonzero(matrix[:, columns], axis=0)
    if counts.tolist() != [6, 4, 5, 6, 2, 5]:
        raise ValueError(f"effective reductor counts differ: {counts.tolist()}")
    changed = np.zeros(matrix.shape, dtype=bool)
    for cell in requested:
        if cell not in ids:
            raise ValueError(f"missing reductor neuron {cell}")
        i = ids.index(cell)
        js = [j for j in columns if matrix[i, j] > 0]
        if len(js) != 1:
            raise ValueError(f"{cell} does not have exactly one effective CTr-roll entry")
        changed[i, js[0]] = True
    arm = matrix.copy()
    arm[changed] *= -1
    _, _, baseline_mask = MS.split_map(matrix)
    _, _, changed_mask = MS.split_map(arm)
    assert not baseline_mask[columns].any()
    assert np.array_equal(np.abs(arm), np.abs(matrix))
    assert np.array_equal(arm[~changed], matrix[~changed])
    assert np.array_equal(arm != matrix, changed)
    receipt = {
        "matrix_shape": list(matrix.shape), "effective_counts": counts.tolist(),
        "matrix_dtype": str(matrix.dtype), "mn_ids": ids,
        "base_matrix": matrix.tolist(), "arm_matrix": arm.tolist(),
        "base_matrix_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
        "arm_matrix_sha256": hashlib.sha256(arm.tobytes()).hexdigest(),
        "magnitudes_equal": True, "unnamed_entries_equal": True,
        "changed_entries": [
            {"banc_888_id": ids[i], "dof": MM.DOF_NAMES[j],
             "before": float(matrix[i, j]), "after": float(arm[i, j])}
            for i, j in zip(*np.where(changed))],
        "baseline_both_mask": baseline_mask.tolist(),
        "arm_unpinned_both_mask": changed_mask.tolist(),
        "coefficient_sums": {
            leg: {"base": float(matrix[:, j].sum()), "signed": float(arm[:, j].sum()),
                  "positive": float(np.maximum(arm[:, j], 0).sum()),
                  "negative": float(np.maximum(-arm[:, j], 0).sum())}
            for leg, j in zip(DRIVE_LEGS, columns)},
    }
    return arm, baseline_mask, receipt


def position_order(model, actuator_ids):
    """Resolve capture-12 coordinates by their actual MuJoCo joint names."""
    rows = []
    for index, aid in enumerate(actuator_ids):
        jid = int(model.actuator_trnid[aid, 0])
        joint_name = model.joint(jid).name or ""
        parent, child, axis = joint_name.split("/")[-1].split("-")
        joint = ("ThC" if parent == "c_thorax" else "CTr" if child.endswith("trochanterfemur")
                 else "FTi" if child.endswith("tibia") else "TiTa" if child.endswith("tarsus1") else None)
        rows.append({"index": index, "actuator_id": int(aid),
                     "actuator_name": model.actuator(aid).name, "joint_name": joint_name,
                     "qpos_address": int(model.jnt_qposadr[jid]),
                     "dof": f"{child.split('_')[0]}_{joint}_{axis}" if joint else None})
    return rows


def stiffness_receipt(model, stiff, actuator_ids, baseline_mask, theta):
    import motormap as MM

    order = position_order(model, actuator_ids)
    aligned = [r["dof"] for r in order] == MM.DOF_NAMES
    # Resolve by name for the gain check even if the range-reading order fails.
    ctr = [next(r["actuator_id"] for r in order if r["dof"] == f"{leg}_CTr_roll")
           for leg in DRIVE_LEGS]
    # self.mask contains indices, not the Boolean mask passed to the controller.
    mask_equal = np.array_equal(stiff.mask, np.flatnonzero(baseline_mask))
    if not mask_equal or set(ctr) & set(stiff.ids.tolist()):
        raise ValueError("reductor intervention changed stiffness eligibility")
    return {"position_order": order, "position_order_aligned": aligned,
            "baseline_both_mask": baseline_mask.tolist(),
            "padded_mask_indices": stiff.mask.tolist(), "baseline_mask_equal": mask_equal,
            "ctr_actuator_ids": ctr, "leg_of_dof": list(stiff.leg_of_dof),
            "k_min": stiff.k_min, "k_gain": stiff.k_gain, "F_ref_k": stiff.F_ref_k,
            "cocon_scale": theta.get("cocon_scale", 1.0),
            "row_ms": [], "ctr_gainprm_0": [], "ctr_biasprm_1": []}
