"""The co-contraction -> joint stiffness mapping. Read MUSCLE-SPLIT.md first;
this file implements exactly its spec and nothing beyond it.

WHY: the signed-sum muscle collapse makes joint stiffness structurally zero --
a real fly stiffens a joint by activating both antagonists at once, and the
measured physiology says standing REQUIRES that: passive fly joint stiffness
is ~70x too small to support the body, and a posture model needs 40-80x over
passive to stand (Wang et al. 2025). frontier.png shows every fast candidate
in this model class dies by falling.

WHAT IS MEASURED vs SEARCHED here:
  - which DOFs get a second channel: ANATOMY (BANC sign census, 30 of 42 DOFs
    carry both flexor- and extensor-signed MNs; the 11 single-signed DOFs and
    rm_ThC_pitch keep the stock fixed servo -- we do not invent a channel).
  - the stiffness DRIVE: the tau_act-filtered antagonist activation sum from
    body.Body.step2 -- measured filter, never raw spikes (a rhythmic k(t) is
    a time-varying second-order system that can ring with no connectome in
    it; the repo's resonance rule applies).
  - k_min, k_gain: SEARCHED. Fly co-contraction and active joint stiffness
    are unmeasured (no recording exists), so these are explicitly-swept
    unknowns per the parameter rule. The 40-80x-passive posture bound is a
    sanity anchor, not a fit target.

MuJoCo mechanics (verified live on this stack, mujoco 3.9.0 / flygym 2.1.0):
flygym position actuators are gaintype=FIXED gainprm[0]=kp, biastype=AFFINE
biasprm=[0,-kp,-kv], force = kp*ctrl - kp*q - kv*qdot clamped to forcerange.
Writing gainprm/biasprm mid-rollout takes effect immediately; k=200 (4.4x the
stock kp 45) is verified stable at dt=1e-4 Euler. THE CONSISTENCY TRAP:
gainprm[0] and biasprm[1] MUST be written together or the equilibrium silently
moves off the commanded target (0.40 rad error measured); apply() is the one
place both are written. kv stays 0 by default -- actuator damping is EXPLICIT
under Euler and blows up before high k does.
"""
import numpy as np


def split_map(M):
    """Signed motor map -> (M_flex, M_ext, both_mask).

    M is the (391, 42) normalised map used by the walking code, flexor entries
    negative. Returns non-negative flexor and extensor maps whose difference
    is exactly M, plus the per-DOF mask of where BOTH signs exist (the only
    DOFs whose stiffness may be modulated).
    """
    M = np.asarray(M, float)
    M_ext = np.maximum(M, 0.0)
    M_flex = np.maximum(-M, 0.0)
    both = (M_ext.sum(axis=0) > 0) & (M_flex.sum(axis=0) > 0)
    return M_flex, M_ext, both


class StiffnessController:
    """Per-tick writes of k = k_min + k_gain * sat(cocon / drive_scale) onto
    the 42 position actuators, two-channel DOFs only.

    sat = tanh keeps k bounded in [k_min, k_min + k_gain] whatever the network
    does, mirroring the theta path's own saturation. Single-signed DOFs keep
    the stock kp forever (anatomy provides no antagonist there).
    """

    def __init__(self, sim, fly, both_mask, k_min, k_gain, drive_scale,
                 tau_d_ms=0.0):
        from flygym.compose.fly.base_fly import ActuatorType
        ids = np.asarray(sim._intern_actuatorids_by_type_by_fly[
            ActuatorType.POSITION][fly.name])
        # actuator ids are in DOF order (validate_motormap.py certifies the
        # M-columns <-> position-actuator order), so one mask indexes both
        assert len(ids) == len(both_mask), (len(ids), len(both_mask))
        self.mask = np.flatnonzero(np.asarray(both_mask, bool))
        self.ids = ids[self.mask]
        self.m = sim.mj_model
        self.k_min = float(k_min)
        self.k_gain = float(k_gain)
        self.drive_scale = float(drive_scale)
        self.kv_of_k = float(tau_d_ms) / 1000.0
        self.k_stock = float(self.m.actuator_gainprm[ids[0], 0])  # 45.0

    def apply(self, cocon):
        """cocon: the 42-vector body.cocon (tau_act-filtered, >= 0)."""
        k = self.k_min + self.k_gain * np.tanh(
            np.asarray(cocon, float)[self.mask] / self.drive_scale)
        self._write(k)

    def set_leg_map(self, dof_names, F_ref_k):
        """Enable LOAD-CLOSED mode (MUSCLE-SPLIT.md outcome section): stiffness
        follows each leg's own measured contact load, the only quantified
        regulation of co-contraction in a walking insect (Guenzel, Schmitz &
        Duerr 2022, stick insect, load-dependent, d = 1.2 -- mechanism FORM
        transplanted, gains searched; species flagged). A tilting fly loads
        one side more; stiffening the loaded side's joints resists the tip.
        """
        legs = [n[:2] for n in dof_names]
        self.leg_of_dof = [legs[i] for i in self.mask]
        self.F_ref_k = float(F_ref_k)

    def apply_load(self, loads):
        """loads: dict leg-code -> contact force (previous control tick; the
        5 ms lag is below real reflex latency, noted not hidden)."""
        L = np.array([loads.get(l, 0.0) for l in self.leg_of_dof])
        k = self.k_min + self.k_gain * np.clip(L / self.F_ref_k, 0.0, 1.0)
        self._write(k)

    def _write(self, k):
        # gainprm[0] and biasprm[1] together, always (the consistency trap)
        self.m.actuator_gainprm[self.ids, 0] = k
        self.m.actuator_biasprm[self.ids, 1] = -k
        if self.kv_of_k:
            self.m.actuator_biasprm[self.ids, 2] = -k * self.kv_of_k
