"""The missing layer: joint drive -> joint angles -> proprioceptor firing.

This is what closes the sensorimotor loop. Until now the proprioceptors have
been driven by unstructured Poisson noise, because nothing converted motor
output back into sensory input. In a walking fly the leg's own movement is what
drives the proprioceptors, and three regimes have now been searched for a motor
rhythm without one -- so the open loop is the leading explanation.

⚠️ READ MUSCLE.md. It exists because this layer is the only component that is
both fitted AND in the causal path, so freedom here does not explore a
hypothesis, it launders a bad result into a good score.

THE DISCIPLINE APPLIED HERE: NOTHING IS FITTED.

Every constant below is a measured anchor, a published encoding, or an
explicitly-swept unknown. There is no optimiser anywhere in this file. That is
deliberate -- if closing the loop with zero fitted parameters produces a
rhythm, nothing was laundered to get it, and MUSCLE.md's alarm (does the
real-vs-shuffled gap shrink once muscles are added?) cannot fire because there
is no fit to shrink it.

    quantity                what it is                           source
    tau_act = 12.3 ms       muscle activation, first order       Azevedo 2020:
                            (8.5 ms to half max => tau/ln2)      eLife 56754
    joint working ranges    per-DOF min/max over a step cycle    flygym's own
                                                                 CPG walker
    FeCO encoding           claw = angle, hook = signed          Mamiya, Gurung
                            velocity, club = |velocity|          & Tuthill 2018
    per-MN gain             ALL EQUAL                            not fitted; the
                                                                 391-parameter
                                                                 block of
                                                                 MUSCLE.md set
                                                                 to a constant
    tau_joint, drive_scale  SWEPT, never fitted                  see below

⚠️ THE JOINT IS FIRST-ORDER BY CONSTRUCTION, AND THAT IS A CONTROL, NOT A
SHORTCUT. A second-order joint (mass-spring-damper) can RESONATE, and a
resonant limb would produce a rhythm with no connectome involvement at all --
we would be measuring the body and calling it a brain. Overdamped first-order
dynamics cannot oscillate on their own, so any rhythm that appears has to come
from the loop. Physically this is the overdamped limit of

    b dtheta/dt = torque_muscle(a) - k (theta - theta_0)

rewritten as a relaxation toward an activation-dependent equilibrium with time
constant tau_joint = b/k.

⚠️ THIS IS NOT NeuroMechFly. It is 42 independent joints with no inertia, no
contact, no gravity and no body. It is enough to ask whether proprioceptive
feedback organises a rhythm, because that question is about loop TIMING. It is
NOT enough for a behavioural score -- that has to come from the real body
model, and any number from here must never be fed to flyscore/.

TWO PARAMETERS ARE SWEPT RATHER THAN SET, and for a reason. `tau_joint` is the
loop's dominant lag, and delay_law.py already established that the lag is what
sets a feedback loop's frequency -- so fixing it would be assuming the answer.
`drive_scale` is the loop gain, which sets whether the loop oscillates at all.
Sweeping them and reporting the relationship is the experiment; fitting them
to produce 14 Hz would be circular.
"""
import json

import numpy as np

import motormap as MM

TAU_ACT_MS = 8.5 / np.log(2.0)      # 12.26 ms, from Azevedo's 8.5 ms half-max

# Which DOF each leg's femoral chordotonal organ reports. NOT a guess: the
# FeCO sits in the femur and monitors the femur-tibia joint. Anatomy, not fit.
FECO_DOF = "FTi_pitch"
# CD1-KL: the seven per-leg DOF, in motormap.DOF_NAMES order. Used only
# when th["hair_dof"] == "spread"; the default path never reads this.
HAIR_SPREAD_DOF = ("ThC_yaw", "ThC_pitch", "ThC_roll", "CTr_pitch",
                   "CTr_roll", "FTi_pitch", "TiTa_pitch")
LEG_CODES = ["lf", "lm", "lh", "rf", "rm", "rh"]


def dallmann_u_min(coeffs, offset_deg):
    """Amendment 23cbec94 to design 2284c922: (a_min, p_min), the least value
    of p(a - offset) over a in [0, 180] degrees and the angle where it sits,
    found analytically from the real roots of p' inside the range and the two
    ends. With measurement 662e8678's coefficients and offset 80: 78.09 and
    -0.0362."""
    c = np.asarray(coeffs, dtype=np.float64)
    lo, hi = -float(offset_deg), 180.0 - float(offset_deg)
    xs = [lo, hi] + [float(x.real) for x in np.roots(np.polyder(c))
                     if abs(x.imag) < 1e-9 and lo <= x.real <= hi]
    p = np.polyval(c, np.asarray(xs))
    k = int(np.argmin(p))
    return xs[k] + float(offset_deg), float(p[k])


def dallmann_u(a_deg, coeffs, offset_deg, p_min=None):
    """Design 2284c922 B1 as amendment 23cbec94 corrects it: the claw's U,
    u(a) = (p(a - offset) - p_min) / (p(0 - offset) - p_min), p the polynomial
    `coeffs` (highest power first, as measurement 662e8678 prints them), a the
    femur-tibia angle in degrees (0 fully flexed), offset the authors' shift
    (their code's x = a - 80), p_min from dallmann_u_min (computed when not
    given). 1.0 at full flexion, 0 at the minimum, at or above 0 from 0 to
    180 degrees."""
    c = np.asarray(coeffs, dtype=np.float64)
    if p_min is None:
        p_min = dallmann_u_min(c, offset_deg)[1]
    p0 = np.polyval(c, -float(offset_deg))
    return ((np.polyval(c, np.asarray(a_deg, dtype=np.float64) - offset_deg)
             - p_min) / (p0 - p_min))


# BANC's body_part_sensory / nerve -> our leg code
# Marin et al. 2024's exact FeCO types and the modality each one is, as
# body.py's claw and hook branches already cite them: SNpp50 and SNpp51 are
# the flexion- and extension-tuned claw (position) cells, SNpp41 and SNpp39
# the flexion- and extension-tuned hook (signed velocity) cells.
EXACT_FECO_MODALITY = {"SNpp50": "claw", "SNpp51": "claw",
                       "SNpp41": "hook", "SNpp39": "hook"}

NERVE_LEG = {("left", "prothoracic"): "lf", ("left", "mesothoracic"): "lm",
             ("left", "metathoracic"): "lh", ("right", "prothoracic"): "rf",
             ("right", "mesothoracic"): "rm", ("right", "metathoracic"): "rh"}


def load_ranges(path="data/dof_ranges.json"):
    d = json.load(open(path))
    lo = np.array([d[n][0] for n in MM.DOF_NAMES])
    hi = np.array([d[n][1] for n in MM.DOF_NAMES])
    return lo, hi


# Measured walking ROM per joint, degrees (Haustein et al. 2024, tethered on
# ball at ~15 mm/s, 12 flies, 2,250 steps). The battery measured the champion
# using 100.0% of the CPG-borrowed ranges at every FTi/CTr joint -- the
# stride is range-capped, and the borrowed widths are ~half the measured ones
# at the propulsion joints (and WIDER than measured at mid CTr, the one ROM
# conflict). Replacing borrowed widths with measured ones is an anchor swap,
# not a fit. Centers keep the CPG posture; ThC/TiTa/CTr_roll keep CPG widths
# (Haustein's ThC axes cannot be paired to ours -- see motormap.py).
HAUSTEIN_ROM_DEG = {
    "FTi_pitch": {"f": 96.6, "m": 21.5, "h": 84.1},
    "CTr_pitch": {"f": 91.2, "m": 22.5, "h": 56.3},
}


def measured_ranges(path="data/dof_ranges.json", rom_override=None):
    """CD2-AV (lane A, 2026-09-05): rom_override merges over HAUSTEIN_ROM_DEG,
    shaped the same way ({joint suffix: {leg letter: degrees}}), and every joint
    it does not name keeps its current value. Absent/None = bit-identical.

    The reason it exists: HAUSTEIN_ROM_DEG is measured walking RANGE OF MOTION
    at about 15 mm/s (see the constant's own comment) and this function uses it
    as the joint's LIMIT, which the body then saturates against, so no theta can
    command a stride wider than that recording's. On the middle knee the cap is
    a fifth of the front's while the animal's own tracking puts that ratio near
    one (CD2-AK, 28,059 sequences; CD2-AU).
    """
    lo, hi = load_ranges(path)
    rom_table = HAUSTEIN_ROM_DEG
    if rom_override:
        if not isinstance(rom_override, dict):
            raise ValueError(f"rom_override must be a dict, got {rom_override!r}")
        rom_table = {k: dict(v) for k, v in HAUSTEIN_ROM_DEG.items()}
        for _joint, _legs in rom_override.items():
            if not isinstance(_legs, dict):
                raise ValueError(f"rom_override[{_joint!r}] must be a dict of leg -> degrees")
            for _leg, _deg in _legs.items():
                if _leg not in ("f", "m", "h"):
                    raise ValueError(f"rom_override leg must be f, m or h, got {_leg!r}")
                _deg = float(_deg)
                if not 0.0 < _deg <= 360.0:
                    raise ValueError(f"rom_override degrees must be in (0, 360], got {_deg}")
                rom_table.setdefault(_joint, {})[_leg] = _deg
    for j, name in enumerate(MM.DOF_NAMES):
        rom = rom_table.get(name[3:], {}).get(name[1])
        if rom is not None:
            mid = 0.5 * (lo[j] + hi[j])
            half = 0.5 * np.radians(rom)
            lo[j], hi[j] = mid - half, mid + half
    return lo, hi


class Body:
    """42 independent overdamped joints driven by first-order muscle activation.

    step() advances by dt_ms and returns (theta, vel) in radians and rad/s.
    Normalised position is 0 at the low end of the DOF's working range and 1 at
    the high end, which is what the proprioceptor encodings consume.
    """

    def __init__(self, tau_joint_ms=15.0, drive_scale=200.0,
                 tau_act_ms=TAU_ACT_MS, ranges=None, f0_hz=None, zeta=0.8,
                 exact_filters=False, joint_bias_rad=None,
                 drive_scale_vec=None, joint_integrator="euler",
                 drive_offset_vec=None, tau_deact_ms=None):
        # exact_filters: use the exact ZOH coefficient 1-exp(-dt/tau) for
        # the first-order filters instead of forward-Euler dt/tau. The
        # Euler coefficient is NOT tick-invariant (tau_act 12.26 ms runs
        # with an EFFECTIVE tau of ~9.5 ms at dt=5 but ~11.0 ms at
        # dt=2.5), which the 2026-08-14 tick pilot exposed: same theta,
        # tick 5 -> 4.18 Hz stepping, tick 2.5 -> 1.64 Hz. Default off =
        # bit-identical; every 5 ms-era champion's searched taus mean
        # Euler-effective values, so flipping this on breaks them by
        # design. (The second-order branch keeps semi-implicit Euler in
        # both modes; its pole error is a few percent, the first-order
        # coefficient was the dominant non-invariance.)
        # f0_hz: SECOND-ORDER joint dynamics (the tuned leg). The gait-clock
        # dissection (results-dwell.json) proved the step cycle is paced by
        # this filter -- tau_joint 15.8ms -> 4.2 Hz, 30ms -> 1.4 Hz, and
        # FASTER first-order joints fall instead of stepping faster. Real
        # insect legs are tuned second-order systems whose impedance peaks
        # AT stride frequency (Dudek & Full 2006, recovery 16-46 ms); fly
        # values unmeasured, so f0/zeta are searched with that knee as the
        # anchor. None = first-order, bit-identical -- and remember the
        # week-1 rule: a resonant limb can manufacture rhythm with no
        # connectome, so every tempo claim on this body needs its shuffled-
        # connectome control (built into joint2_pilot.py).
        self.lo, self.hi = ranges if ranges is not None else load_ranges()
        self.mid = 0.5 * (self.lo + self.hi)
        self.half = 0.5 * (self.hi - self.lo)
        tau_act = np.asarray(tau_act_ms, dtype=float)
        if tau_act.ndim == 0:
            if not np.isfinite(tau_act) or float(tau_act) <= 0.0:
                raise ValueError("tau_act_ms must be finite and positive")
            self.tau_act = float(tau_act)
        else:
            if (tau_act.shape != (len(self.lo),)
                    or not np.all(np.isfinite(tau_act))
                    or np.any(tau_act <= 0.0)):
                raise ValueError(
                    "tau_act_ms must be a scalar or finite positive 42-vector")
            self.tau_act = tau_act.copy()
        # tau_deact_ms: the relaxation constant of the two-channel activation
        # (design 686abf38's build, ported for design c593eb5c, study
        # a3f66049). Each antagonist channel rises at tau_act while its drive
        # is at or above its activation and falls at tau_deact otherwise, per
        # element; Azevedo 2020 measured a fast rise (half force in ~8.5 ms)
        # and ~100 ms for slow units to release. None (default) keeps the one
        # symmetric constant, bit-identical. A scalar or a finite positive
        # 42-vector, in tau_act_ms's idiom. step2's own low-pass only: step()
        # and step2's a_in path raise when it is set.
        if tau_deact_ms is None:
            self.tau_deact = None
        else:
            tau_deact = np.asarray(tau_deact_ms, dtype=float)
            if tau_deact.ndim == 0:
                if not np.isfinite(tau_deact) or float(tau_deact) <= 0.0:
                    raise ValueError("tau_deact_ms must be finite and positive")
                self.tau_deact = float(tau_deact)
            else:
                if (tau_deact.shape != (len(self.lo),)
                        or not np.all(np.isfinite(tau_deact))
                        or np.any(tau_deact <= 0.0)):
                    raise ValueError(
                        "tau_deact_ms must be a scalar or finite positive 42-vector")
                self.tau_deact = tau_deact.copy()
        self.tau_joint = float(tau_joint_ms)
        self.drive_scale = float(drive_scale)
        # f0_hz takes a scalar or a per-DOF 42-vector, in the idiom tau_act_ms
        # already uses. The resonance the 15-40 Hz anchor band was measured on
        # is a property of one joint's own mass and lever arm, so a scalar
        # cannot express "this joint tracks its command and that one does not",
        # and no setting of it ever could: C-N336 raised the scalar through the
        # band and delivery rose on all six graded DOFs while standing fell
        # from 11 of 12 to 2, because the same change reached the joints
        # holding the body up. A scalar keeps the old code path untouched and
        # therefore bit-identical; the vector branch is taken only for an
        # array, and its values face the same finiteness and positivity test
        # the scalar faced. Lane C, 2026-09-11.
        f0 = None if f0_hz is None else np.asarray(f0_hz, dtype=float)
        if f0 is not None and f0.ndim > 0:
            if (f0.shape != (len(self.lo),) or not np.all(np.isfinite(f0))
                    or np.any(f0 <= 0.0)):
                raise ValueError(
                    "f0_hz must be a scalar or a finite positive 42-vector")
            self.f0 = f0.copy()
        else:
            self.f0 = None if f0 is None else float(f0)
        self.zeta = float(zeta)
        self.exact = bool(exact_filters)
        # joint_integrator: "euler" (default, bit-identical) keeps the
        # semi-implicit Euler step of the second-order joint; "exact" steps
        # the same damped oscillator by its zero-order-hold matrix
        # exponential, theta_eq held over the tick. The Euler step's fast
        # eigenvalue is 1 - 2*zeta*w0*dt: at the champion dials (f0 20.55
        # Hz, zeta 3.0, tick 2.5 ms) that is -0.937, and the coupled pair is
        # (0.948, -0.989), so every joint carries a tick-alternating mode
        # that halves in 61 ticks while the physical fast mode (time
        # constant 1.33 ms) should decay to 0.15 per tick. Measured on qh1
        # captures 2026-09-04: 83% of consecutive joint-angle differences
        # change sign (0.5 = no alternation). The exact step has no such
        # mode at any (f0, zeta, dt). Lane A, 2026-09-04.
        if joint_integrator not in ("euler", "exact"):
            raise ValueError("joint_integrator must be 'euler' or 'exact'")
        self.joint_exact = joint_integrator == "exact"
        self._phi_cache = {}
        self.n = len(self.lo)
        # drive_scale_vec: PER-DOF loop gain for the joint equilibrium only
        # (theta_eq = mid + half*tanh(a/drive_scale_vec)). None = a uniform
        # vector at the scalar drive_scale, which divides bit-identically to
        # the scalar path. The lever exists because a leg whose motor drive
        # reaches the muscle weakly (measured: right hind flexor activation
        # 0.71x its left mirror, lane E R159) produces a small activation `a`;
        # since `a` saturates at `drive`, tau_act cannot recover the lost
        # amplitude, but a LOWER drive_scale on that DOF makes the same `a`
        # swing the joint further. The scalar self.drive_scale is unchanged
        # and still feeds the campaniform normalisation and twin_body.
        if drive_scale_vec is None:
            self.drive_scale_vec = np.full(self.n, self.drive_scale)
        else:
            dsv = np.asarray(drive_scale_vec, dtype=float)
            if (dsv.shape != (self.n,) or not np.all(np.isfinite(dsv))
                    or np.any(dsv <= 0.0)):
                raise ValueError(
                    "drive_scale_vec must be a finite positive 42-vector")
            self.drive_scale_vec = dsv.copy()
        # drive_offset_vec (design fcb48834 Build 2): a per-DOF offset o_d in
        # activation units added to a before the tanh only, theta_eq = mid +
        # half tanh((a + o_d) / s_d), so the stance operating point moves on
        # the tanh while a, cocon and every reader of them are unchanged.
        # None = the original expression, bit-identical.
        if drive_offset_vec is None:
            self.drive_offset = None
        else:
            off = np.asarray(drive_offset_vec, dtype=float)
            if off.shape != (self.n,) or not np.all(np.isfinite(off)):
                raise ValueError("drive_offset_vec must be a finite 42-vector")
            self.drive_offset = off.copy()
        if joint_bias_rad is None:
            self.joint_bias = None
        else:
            bias = np.asarray(joint_bias_rad, dtype=float)
            if (bias.shape != (self.n,) or not np.all(np.isfinite(bias))
                    or np.any(np.abs(bias) > np.radians(20.0))):
                raise ValueError(
                    "joint_bias_rad must be a finite 42-vector within 20 deg")
            self.joint_bias = bias.copy()
        self.reset()

    def _alpha(self, tau, dt_ms):
        return (1.0 - np.exp(-dt_ms / tau)) if self.exact else (dt_ms / tau)

    def _phi(self, dt):
        # expm(M dt) for M = [[0, 1], [-w0^2, -2 zeta w0]], the state
        # (theta - theta_eq, vel) of the damped oscillator over one tick.
        key = round(dt, 12)
        if key not in self._phi_cache:
            from scipy.linalg import expm
            if np.ndim(self.f0) == 0:
                w0 = 2.0 * np.pi * self.f0
                m = np.array([[0.0, 1.0], [-w0 * w0, -2.0 * self.zeta * w0]])
                self._phi_cache[key] = expm(m * dt)
            else:
                # Per-DOF f0: the same four coefficients, stacked per DOF, so
                # the caller's p[i, j] arithmetic stays elementwise. One expm
                # per distinct value, which is a handful.
                stack = np.empty((2, 2, self.n))
                for value in np.unique(self.f0):
                    w0 = 2.0 * np.pi * float(value)
                    m = np.array([[0.0, 1.0],
                                  [-w0 * w0, -2.0 * self.zeta * w0]])
                    stack[:, :, self.f0 == value] = expm(m * dt)[:, :, None]
                self._phi_cache[key] = stack
        return self._phi_cache[key]

    def reset(self):
        self.a = np.zeros(self.n)          # muscle activation, signed
        self.theta = self.mid.copy()       # joint angle, rad
        if self.joint_bias is not None:
            self.theta = np.clip(
                self.theta + self.joint_bias, self.lo, self.hi)
        self.vel = np.zeros(self.n)        # rad/s
        # two-channel state (only advanced by step2; zero otherwise)
        self.a_flex = np.zeros(self.n)     # flexor-side activation, >= 0
        self.a_ext = np.zeros(self.n)      # extensor-side activation, >= 0
        self.cocon = np.zeros(self.n)      # a_flex + a_ext, the stiffness drive
        self.vel_j = np.zeros(self.n)      # second-order joint velocity state

    def step(self, drive, dt_ms):
        """drive: signed 42-vector of motor-neuron activity (spikes/bin)."""
        if self.tau_deact is not None:
            raise ValueError("tau_deact_ms is a two-channel (step2) mechanism; "
                             "step() does not apply it")
        self.a += self._alpha(self.tau_act, dt_ms) * (
            np.asarray(drive, float) - self.a)
        # bounded by construction: tanh keeps the joint inside its working
        # range whatever the drive does, which is what a joint stop does.
        if self.drive_offset is None:
            theta_eq = self.mid + self.half * np.tanh(self.a / self.drive_scale_vec)
        else:
            theta_eq = self.mid + self.half * np.tanh(
                (self.a + self.drive_offset) / self.drive_scale_vec)
        if self.joint_bias is not None:
            theta_eq = np.clip(theta_eq + self.joint_bias, self.lo, self.hi)
        d_theta = self._alpha(self.tau_joint, dt_ms) * (theta_eq - self.theta)
        self.theta += d_theta
        self.vel = d_theta / (dt_ms / 1000.0)
        return self.theta, self.vel

    def step2(self, drive_flex, drive_ext, dt_ms, a_in=None):
        """Two-channel step (see MUSCLE-SPLIT.md): antagonist activations kept
        separately so their SUM (co-contraction) exists as a signal. The theta
        path is the same physics as step() -- the activation low-pass is
        linear, so lowpass(ext) - lowpass(flex) equals lowpass(ext - flex) up
        to float association -- the split's only NEW quantity is `cocon`.
        Nothing here maps cocon to stiffness; that lives in muscle_split.py.

        a_in (design fcb48834 Build 1): (a_flex, a_ext) already pooled from
        the per-unit forms (gpu/bob/forms.py), which carry the activation
        low-pass per unit, so they are written as the activations and the
        body's own low-pass is skipped; drive_flex and drive_ext are then
        unused. None = the low-pass below, bit-identical.
        """
        if a_in is None:
            drive_flex = np.asarray(drive_flex, float)
            drive_ext = np.asarray(drive_ext, float)
            if self.tau_deact is None:
                al = self._alpha(self.tau_act, dt_ms)
                self.a_flex += al * (drive_flex - self.a_flex)
                self.a_ext += al * (drive_ext - self.a_ext)
            else:
                # rise at tau_act, fall at tau_deact, per element and channel
                al_f = self._alpha(np.where(drive_flex >= self.a_flex,
                                            self.tau_act, self.tau_deact), dt_ms)
                al_e = self._alpha(np.where(drive_ext >= self.a_ext,
                                            self.tau_act, self.tau_deact), dt_ms)
                self.a_flex += al_f * (drive_flex - self.a_flex)
                self.a_ext += al_e * (drive_ext - self.a_ext)
        else:
            if self.tau_deact is not None:
                raise ValueError("tau_deact_ms acts on the body's own activation "
                                 "low-pass, which a_in skips")
            self.a_flex = np.array(a_in[0], dtype=float)
            self.a_ext = np.array(a_in[1], dtype=float)
            if self.a_flex.shape != (self.n,) or self.a_ext.shape != (self.n,):
                raise ValueError("a_in must be two 42-vectors (a_flex, a_ext)")
        self.a = self.a_ext - self.a_flex   # downstream consumers (prop.rates)
        self.cocon = self.a_ext + self.a_flex
        if self.drive_offset is None:
            theta_eq = self.mid + self.half * np.tanh(self.a / self.drive_scale_vec)
        else:
            theta_eq = self.mid + self.half * np.tanh(
                (self.a + self.drive_offset) / self.drive_scale_vec)
        if self.joint_bias is not None:
            theta_eq = np.clip(theta_eq + self.joint_bias, self.lo, self.hi)
        if self.f0 is not None:
            dt = dt_ms / 1000.0
            w0 = 2.0 * np.pi * self.f0
            if self.joint_exact:
                p = self._phi(dt)
                e = self.theta - theta_eq
                e_new = p[0, 0] * e + p[0, 1] * self.vel_j
                self.vel_j = p[1, 0] * e + p[1, 1] * self.vel_j
                self.theta = np.clip(theta_eq + e_new, self.lo, self.hi)
                self.vel = self.vel_j.copy()
                return self.theta, self.vel
            # tuned second-order joint, semi-implicit Euler (stable for the
            # w0*dt this control tick allows; cap f0 <= 20 Hz at dt = 5 ms.
            # Stable is not alternation-free: see joint_integrator above.)
            self.vel_j += dt * (w0 * w0 * (theta_eq - self.theta)
                                - 2.0 * self.zeta * w0 * self.vel_j)
            d_theta = dt * self.vel_j
            self.theta = np.clip(self.theta + d_theta, self.lo, self.hi)
            self.vel = self.vel_j.copy()
            return self.theta, self.vel
        d_theta = self._alpha(self.tau_joint, dt_ms) * (theta_eq - self.theta)
        self.theta += d_theta
        self.vel = d_theta / (dt_ms / 1000.0)
        return self.theta, self.vel

    def normalised(self):
        return np.clip((self.theta - self.lo) / (2 * self.half + 1e-12), 0, 1)


class Proprioceptors:
    """Joint state -> firing rate, one rate per BANC proprioceptor.

    ENCODING, from Mamiya, Gurung & Tuthill 2018 (Neuron 100:636) which
    characterised the three femoral chordotonal organ subtypes that BANC's
    `cell_sub_class` column names directly:

      claw  tonic, POSITION tuned. Two populations exist, flexion-tuned and
            extension-tuned, so half are given each sign.
      hook  phasic and DIRECTIONALLY SELECTIVE -- separate flexion and
            extension populations, each firing only for movement one way.
      club  phasic, bidirectional, non-directional: responds to movement and
            vibration regardless of sign. Encodes |velocity|.

    Plus two classes outside the FeCO:

      hair plate           fires near the joint's extremes
      campaniform sensilla cuticular strain, i.e. LOAD. Encoded from muscle
                           activation magnitude, which is the only force-like
                           quantity this reduced body has.

    ⚠️ WHAT IS ASSUMED RATHER THAN KNOWN. BANC names the subtype but not which
    member of a subtype is flexion- vs extension-tuned, so that split is made
    by index parity -- a hypothesised assignment, flagged here, and one a
    search could later replace. Hair plates sit at several joints and BANC does
    not say which, so they are attached to the same leg's ThC. Neither is a
    measurement and both are recorded in the returned `assumptions` dict.
    """

    R_MAX = 120.0        # Hz at full drive; fly leg afferents reach ~100-200
    R_REST = 5.0         # Hz baseline
    V_SCALE = 8.0        # rad/s giving a full phasic response
    # CD1-SM: population -> emit keys, for the "tonic:" high-pass exemption
    TONIC_POP_KEYS = {"claw": ("claw_flex", "claw_ext"),
                      "hook": ("hook_flex", "hook_ext"),
                      "club": ("club",), "hair": ("hair",),
                      "actcamp": ("campaniform",),
                      # lane B 2026-09-12: the claw's two tuning groups on
                      # their own, so the exemption can be split by polarity
                      # as well as by segment. New names, so every mode
                      # string written before this is unchanged.
                      "clawflex": ("claw_flex",), "clawext": ("claw_ext",)}
    # lane B 2026-09-12: optional segment scope on a tonic population,
    # "tonic:claw@front,club" -- the legs each segment name selects.
    SEGMENT_LEGS = {"front": ("lf", "rf"), "middle": ("lm", "rm"),
                    "hind": ("lh", "rh")}

    def __init__(self, meta, net, rng=None, split_seed=None,
                 claw_polarity=None, hook_polarity=None, phasic=0.0,
                 camp_src="act", hair_dof=None,
                 feco_polarity_mode=None,
                 feco_modality_mode=None,
                 proprioceptor_phasic_mode=None,
                 v_scale_hook=None, v_scale_club=None, claw_gain=None,
                 claw_code=None, claw_width=None, claw_flex_frac=None,
                 reflex_map=None, r_rest_hook=None, r_rest_hair=None,
                 claw_u_hz=None, claw_u_offset_deg=None, claw_u_coeffs=None,
                 claw_u_angle=None, claw_u_range=None):
        # phasic (TR1, 2026-08-17, default 0 = bit-identical): real FeCO
        # afferents are deeply phasic; ours are tonic-dominated because
        # position-coding cells fire at their absolute level (coord4:
        # only 4% of cells carry the cycle). phasic subtracts each
        # cell's own running baseline (EMA, ~250 ms at tick 2.5) and
        # re-amplifies the residual: r' = clip((r - phasic*ema) *
        # (1 + 2*phasic), 0, 400). The whisper's volume knob.
        self.phasic = float(phasic)
        # CD1-KL: absent/None -> the original single-joint pin.
        if hair_dof not in (None, "spread"):
            raise ValueError(f"hair_dof must be absent or 'spread', "
                             f"got {hair_dof!r}")
        self._hair_dof = hair_dof
        _hair_n = {}
        # camp_src (CD1, 2026-08-20): "act" (default, bit-identical) keeps
        # the historical stand-in that encodes campaniform from |muscle
        # activation|; "rest" silences that channel by emitting constant
        # R_REST for campaniform rows. Rates change, entry count and order
        # do NOT -- cns draws rng.random(len(drive_idx)) per step, so
        # removing rows would shift every cell's noise stream.
        if camp_src not in ("act", "rest"):
            raise ValueError(f"camp_src must be 'act' or 'rest', "
                             f"got {camp_src!r}")
        self.camp_src = camp_src
        # CD1-SK (lane A, 2026-09-04): per-class velocity scale for the two
        # velocity-encoded populations, None = V_SCALE = bit-identical. The
        # rad/s that gives a full phasic response; a larger value is a
        # weaker velocity signal. Tied at the class, not per cell.
        self.v_scale_hook = self.V_SCALE if v_scale_hook is None else float(v_scale_hook)
        self.v_scale_club = self.V_SCALE if v_scale_club is None else float(v_scale_club)
        # Design 302c1bfb, Build 12 (ledger 07caea12): the hook code's resting
        # rate in Hz, tied at the class; None = R_REST = bit-identical. It
        # covers every row v_scale_hook codes, the type-modality rows included.
        self.r_rest_hook = self.R_REST if r_rest_hook is None else float(r_rest_hook)
        # Design 4cbc7232, Build B4: the hair-plate code's resting rate in Hz,
        # its first term, tied at the class; None = R_REST = bit-identical. The
        # second term, the range-end rise past 0.7, is untouched.
        self.r_rest_hair = self.R_REST if r_rest_hair is None else float(r_rest_hair)
        # CD1-SN (2026-09-04): claw position gain, tied at the population;
        # absent = 1.0 = bit-identical. The clip rises with it so the dial
        # moves the level rather than the saturation.
        self.claw_gain = 1.0 if claw_gain is None else float(claw_gain)
        # CD1-TN (lane A, 2026-09-04, realism directive): claw_code absent
        # or "linear" = the one-line position code = bit-identical.
        # "fractionated": each claw cell gets its own threshold, tiled in
        # index order across its population's half of the range on each leg
        # (Mamiya 2018: flexion- and extension-tuned populations, each on
        # its own side of the mid-position; Mamiya 2023: single claw cells
        # reach half activation at angles staggered along the femur), with
        # one class-tied sigmoid width claw_width
        # (default 0.05 of the normalised range). Which cell takes which
        # threshold is a labelled guess, as hair_dof "spread" is.
        # Design 2284c922, Build B1 (study 310f4795): "dallmann_u", the claw's
        # U-shaped position code of Dallmann 2025 (measurement 662e8678: male,
        # left front leg, calcium, tethered on a ball, resting bouts,
        # UNREVIEWED). Per leg a = the Dallmann angle in degrees from the FTi
        # coordinate (claw_u_angle "180_minus_fti" or "fti", one value or a
        # per-leg dict); u(a) = dallmann_u(a, claw_u_coeffs, claw_u_offset_deg),
        # normalised at the polynomial's minimum over 0 to 180 degrees, which
        # sits at claw_u_amin (amendment 23cbec94: 78.09 at offset 80);
        # flexion-tuned rows fire R_REST + claw_u_hz x u(a) at or under
        # claw_u_amin and R_REST beyond it, extension-tuned rows the same at
        # or above it. The cap is R_REST + claw_u_hz, so u never clips
        # (critique cda37cbb, condition 1). The four keys are required with
        # this code and refused without it; absent = bit-identical.
        # Design bf165b63, Build 1 (study 3be5a2e2, ASSUMED): "dallmann_u_recruit",
        # the same U and keys read as a recruitment code. Per leg and type the
        # rows are ranked k = 1..n in index order, the order "fractionated"
        # ranks; on the U's on side f = min(1, claw_u_hz x u(a) / 115), on the
        # off side f = 0, and row k fires R_MAX when k <= floor(f n), R_REST
        # otherwise. The leg's mean is R_REST + 115 x floor(f n) / n.
        # Design bf165b63, Build 2 (ASSUMED): claw_u_range true evaluates the U
        # at a' = 55.9 + clip((a - lo) / (hi - lo), 0, 1) x 51.6, lo and hi the
        # row's FTi range from the body's own joint table in the U's angle
        # convention, in both the u and the on-side test. The clip acts only
        # outside the range, which the walking joint never leaves (the exact
        # integrator clips theta to it) and a held pose can. Absent = bit-identical.
        if claw_code not in (None, "linear", "fractionated", "dallmann_u",
                             "dallmann_u_recruit"):
            raise ValueError(f"claw_code must be absent, 'linear', "
                             f"'fractionated', 'dallmann_u' or "
                             f"'dallmann_u_recruit', got {claw_code!r}")
        self.claw_code = claw_code
        _u_args = {"claw_u_hz": claw_u_hz,
                   "claw_u_offset_deg": claw_u_offset_deg,
                   "claw_u_coeffs": claw_u_coeffs,
                   "claw_u_angle": claw_u_angle}
        if claw_u_range not in (None, True):
            raise ValueError(f"claw_u_range must be absent or true, got {claw_u_range!r}")
        if claw_u_range is not None and claw_code not in ("dallmann_u", "dallmann_u_recruit"):
            raise ValueError("claw_u_range needs claw_code 'dallmann_u' or "
                             f"'dallmann_u_recruit', got claw_code {claw_code!r}")
        self.claw_u_range = claw_u_range is True
        self._claw_u_span = None
        if claw_code in ("dallmann_u", "dallmann_u_recruit"):
            _missing = sorted(k for k, v in _u_args.items() if v is None)
            if _missing:
                raise ValueError(f"claw_code {claw_code!r} needs {_missing}")
            if claw_gain is not None:
                raise ValueError(f"claw_code {claw_code!r} takes its gain from "
                                 "claw_u_hz; claw_gain must be absent")
            # Amendment cda7e472, Build 6 (study 3be5a2e2): claw_u_hz may be a
            # dict {"claw_flex": Hz, "claw_ext": Hz}, one level per claw group
            # under either U code; claw_u_hz_of holds the group's level that
            # the emit sites read. A scalar fills both, bit-identical. For a
            # dict, claw_u_hz is the dict, so a reader of one scalar fails.
            if isinstance(claw_u_hz, dict):
                if set(claw_u_hz) != {"claw_flex", "claw_ext"}:
                    raise ValueError("claw_u_hz as a dict needs exactly the keys "
                                     f"'claw_flex' and 'claw_ext', got {sorted(claw_u_hz)}")
                self.claw_u_hz_of = {g: float(claw_u_hz[g]) for g in ("claw_flex", "claw_ext")}
                self.claw_u_hz = dict(self.claw_u_hz_of)
            else:
                self.claw_u_hz = float(claw_u_hz)
                self.claw_u_hz_of = {g: self.claw_u_hz for g in ("claw_flex", "claw_ext")}
            if not all(v > 0.0 for v in self.claw_u_hz_of.values()):
                raise ValueError(f"claw_u_hz must be > 0, got {claw_u_hz!r}")
            self.claw_u_offset_deg = float(claw_u_offset_deg)
            self.claw_u_coeffs = [float(c) for c in claw_u_coeffs]
            if len(self.claw_u_coeffs) < 2:
                raise ValueError("claw_u_coeffs must be a polynomial, highest "
                                 f"power first, got {claw_u_coeffs!r}")
            self.claw_u_amin, self.claw_u_pmin = dallmann_u_min(
                self.claw_u_coeffs, self.claw_u_offset_deg)
            _ang = (claw_u_angle if isinstance(claw_u_angle, dict)
                    else {lg: claw_u_angle for lg in MM.LEGS})
            if set(_ang) != set(MM.LEGS) or not set(_ang.values()) <= {
                    "180_minus_fti", "fti"}:
                raise ValueError("claw_u_angle must be '180_minus_fti' or 'fti', "
                                 "or a dict of one per leg "
                                 f"{MM.LEGS}, got {claw_u_angle!r}")
            self.claw_u_angle = dict(_ang)
        elif any(v is not None for v in _u_args.values()):
            raise ValueError("claw_u_* keys need claw_code 'dallmann_u', got "
                             f"claw_code {claw_code!r}")
        self.claw_width = 0.05 if claw_width is None else float(claw_width)
        if self.claw_width <= 0.0:
            raise ValueError(f"claw_width must be > 0, got {claw_width!r}")
        # CD1-TN amendment 4 (2026-09-04): a per-leg-pair flexion fraction
        # for the claw, [T1, T2, T3], mirror-symmetric by construction and
        # assigned in index order within each leg (cell k is flexion-tuned
        # when floor((k+1) f) > floor(k f)); it overrides the typing and the
        # polarity for claw rows.  Absent = bit-identical.
        if claw_flex_frac is not None:
            claw_flex_frac = [float(x) for x in claw_flex_frac]
            if len(claw_flex_frac) != 3 or not all(0.0 <= x <= 1.0 for x in claw_flex_frac):
                raise ValueError(f"claw_flex_frac must be three fractions in [0, 1], got {claw_flex_frac!r}")
        self.claw_flex_frac = claw_flex_frac
        _claw_k = {}
        # CD1-TO/CD1-YX (lane A, ported by the judge 2026-09-05): a per-cell
        # reflex-sign map, {"path": <json with "map": {id: {"polarity":
        # "flex"|"ext"}}>, "kinds": ["claw"|"hook", ...]}, consulted before
        # every other polarity rule for the listed kinds. Absent = BIT-IDENTICAL.
        # lanes/A/feco_front_claw_mirror.json corrects BANC's mirror-reversed
        # right-front claw typing (25 cells), labelled TRANSFERRED.
        self.reflex_map = None
        self.reflex_kinds = ()
        reflex_map_rows = 0
        if reflex_map is not None:
            import json as _json
            _m = _json.load(open(reflex_map["path"]))["map"]
            self.reflex_map = {k: v["polarity"] for k, v in _m.items()}
            self.reflex_kinds = tuple(reflex_map.get("kinds", ("claw", "hook")))
            if not set(self.reflex_kinds) <= {"claw", "hook"}:
                raise ValueError(f"reflex_map kinds must be claw/hook, got {self.reflex_kinds!r}")
        _ids = meta["banc_888_id"].astype(str).to_numpy()
        # CD1-TN amendment 3 (2026-09-04): "banc_type_hook" types the hook
        # rows as "banc_type" does and leaves the claw rows on the
        # claw_polarity / parity fallback, because the SNpp50/51 claw typing
        # is mirror-asymmetric (right front 5 SNpp50 against left front 24).
        if feco_polarity_mode not in (None, "banc_type", "banc_type_hook"):
            raise ValueError(
                "feco_polarity_mode must be absent, 'banc_type' or "
                f"'banc_type_hook', got {feco_polarity_mode!r}")
        self.feco_polarity_mode = feco_polarity_mode
        # R271 (lane E, 2026-09-04): BANC v888 has 18 exact SNpp39/41
        # rows whose coarser cell_sub_class says claw. Those named types are
        # hook neurons. The opt-in correction changes only their rate law;
        # rows remain in the incumbent output positions so the same Poisson
        # draws stay attached to the same sensory cells in paired runs.
        # R299 (lane E, 2026-09-04): "exact_type" generalises the R271
        # correction to every direction. The static audit found 71 of the 341
        # exact-typed leg sensory rows on the wrong rate law. Under this mode
        # exact type decides claw versus hook, and the subclass chain below is
        # the fallback for rows without one of those exact types.
        if feco_modality_mode not in (None, "banc_type", "exact_type"):
            raise ValueError(
                "feco_modality_mode must be absent, 'banc_type' or "
                f"'exact_type', got {feco_modality_mode!r}")
        self.feco_modality_mode = feco_modality_mode
        # CD1-SM (2026-09-04): "tonic:<pop>[,<pop>]" exempts the listed
        # populations (claw, hook, club, hair, actcamp) from the high-pass,
        # tied at the population; "tonic:claw" equals "claw_tonic". Absent
        # and "claw_tonic" are unchanged (bit-identical).
        # lane B (2026-09-12): a population may carry a segment scope,
        # "claw@front" or "claw@front+middle", which exempts only that
        # population's afferents on the named segments' legs. No "@" is the
        # whole population, so every mode string written before this is
        # bit-identical.
        self._tonic_pops = None
        self._tonic_scope = {}
        self._tonic_dose = {}
        self.phasic_dose = {}
        if isinstance(proprioceptor_phasic_mode, str) and \
                proprioceptor_phasic_mode.startswith("tonic:"):
            _pops, _scope, _dose = [], {}, {}
            for _tok in proprioceptor_phasic_mode[6:].split(","):
                # lane B (2026-09-12): "*<dose>" after the token is a PARTIAL
                # exemption, 1.0 the whole one and the default. Read before the
                # segment scope so "clawflex@hind*0.5" parses both.
                _head, _star, _d = _tok.partition("*")
                if _star:
                    try:
                        _dv = float(_d)
                    except ValueError:
                        raise ValueError(
                            "proprioceptor_phasic_mode tonic: dose after '*' "
                            f"must be a number, got {_d!r} on {_tok!r}") from None
                    if not 0.0 <= _dv <= 1.0:
                        raise ValueError(
                            "proprioceptor_phasic_mode tonic: dose must lie in "
                            f"[0, 1], got {_dv} on {_tok!r}")
                _q, _at, _sc = _head.partition("@")
                if _star:
                    if _q in _dose:
                        raise ValueError(
                            "proprioceptor_phasic_mode tonic: population "
                            f"{_q!r} appears twice and one carries a dose")
                    _dose[_q] = _dv
                _pops.append(_q)
                if not _at:
                    continue
                _segs = tuple(_sc.split("+"))
                _badseg = [g for g in _segs if g not in self.SEGMENT_LEGS]
                if _badseg or not _sc:
                    raise ValueError(
                        f"proprioceptor_phasic_mode tonic: unknown segments "
                        f"{_badseg or ['']!r} on {_q!r}; allowed "
                        f"{sorted(self.SEGMENT_LEGS)}")
                _scope[_q] = _segs
            _pops = tuple(_pops)
            _bad = [q for q in _pops if q not in self.TONIC_POP_KEYS]
            if _bad or not _pops:
                raise ValueError(
                    f"proprioceptor_phasic_mode tonic: unknown populations "
                    f"{_bad!r}; allowed {sorted(self.TONIC_POP_KEYS)}")
            self._tonic_pops = _pops
            self._tonic_scope = _scope
            self._tonic_dose = _dose
        elif proprioceptor_phasic_mode not in (None, "claw_tonic"):
            raise ValueError(
                "proprioceptor_phasic_mode must be absent, 'claw_tonic' or "
                f"'tonic:<pops>', got {proprioceptor_phasic_mode!r}")
        self.proprioceptor_phasic_mode = proprioceptor_phasic_mode
        self._ema = None
        self._ema_alpha = 0.01
        # POLARITY OVERRIDES. claw/hook_polarity = +1 assigns that whole
        # population flexion-tuned, -1 extension-tuned; None keeps the split.
        #
        # These exist because sign_test.py measured that the split decides the
        # reflex sign entirely -- 0 of 6 joints kept their sign across four
        # different random splits -- so the assignment is an unidentified
        # parameter and the parameter rule says search it rather than assume it.
        #
        # And control theory says what to look for. Claw encodes POSITION and
        # hook encodes VELOCITY, and in a delayed feedback loop those do
        # opposite things: negative position feedback is what makes a loop
        # oscillate, while negative velocity feedback DAMPS it and positive
        # velocity feedback anti-damps. So the interesting corner is negative
        # on claw and positive on hook, and that is a prediction rather than a
        # sweep.
        # split_seed selects WHICH members of claw/hook are flexion- vs
        # extension-tuned. None reproduces the original parity split exactly.
        # This is a parameter and not a constant because it is an ASSUMPTION:
        # BANC names the subtype but not the tuning, and if the reflex signs
        # depend on it then they are a property of this guess rather than of
        # the fly. sign_test.py measures exactly that.
        rng = rng or np.random.default_rng(0)
        self.split_seed = split_seed
        srng = np.random.default_rng(split_seed) if split_seed is not None \
            else None
        self.dof_index = {n: i for i, n in enumerate(MM.DOF_NAMES)}
        self._dof_name = {i: n for n, i in self.dof_index.items()}
        sub = meta["cell_sub_class"].fillna("").to_numpy()
        cls = meta["cell_class"].fillna("").to_numpy()
        ctype = (meta["cell_type"].fillna("").to_numpy()
                 if "cell_type" in meta else np.full(len(meta), ""))
        nerve = meta["nerve"].fillna("").to_numpy()
        side = meta["side"].fillna("").to_numpy()

        groups = {k: [] for k in ("claw_flex", "claw_ext", "hook_flex",
                                  "hook_ext", "club", "hair", "campaniform")}
        self.dof_of = {k: [] for k in groups}
        n_unassigned = 0
        exact_modality_moved = []
        type_polarity_rows = 0
        type_polarity_fallback_rows = 0
        type_modality_sign_by_row = {}
        dof_by_row = {}
        # LEG proprioceptors only. The first version selected on cell_class
        # alone and picked up chordotonal, hair-plate and campaniform afferents
        # from the wing, haltere, neck and abdomen -- 583 "club" neurons where
        # the legs have 342, and 571 campaniform where the legs have 61. Those
        # would have been driven by a leg joint's angle, which is simply wrong.
        # body_part_sensory is the selector the rest of the repo uses.
        part = meta["body_part_sensory"].fillna("").to_numpy()
        is_leg = np.isin(part, ["front_leg", "middle_leg", "hind_leg"])
        for row in np.flatnonzero(is_leg & np.isin(cls, [
                "chordotonal_organ_neuron", "hair_plate_neuron",
                "campaniform_sensillum_neuron"])):
            nv = nerve[row]
            leg = None
            for (sd, seg), code in NERVE_LEG.items():
                if nv.startswith(sd) and seg in nv:
                    leg = code
                    break
            if leg is None:
                n_unassigned += 1
                continue
            s = sub[row]
            if feco_modality_mode == "exact_type" and \
                    ctype[row] in EXACT_FECO_MODALITY:
                # The chain below tests substrings of s. Naming the exact
                # modality here routes the row through the incumbent claw or
                # hook path with its polarity and DOF unchanged.
                _exact = EXACT_FECO_MODALITY[ctype[row]]
                if _exact not in s:
                    exact_modality_moved.append(
                        (int(row), s, ctype[row], _exact))
                s = _exact
            if "claw" in s:
                if (feco_modality_mode == "banc_type"
                        and ctype[row] in ("SNpp39", "SNpp41")):
                    # Existing hook convention below: SNpp41 encodes
                    # flexion/positive FTi velocity; SNpp39 encodes
                    # extension/negative velocity.
                    type_modality_sign_by_row[int(row)] = (
                        1.0 if ctype[row] == "SNpp41" else -1.0)
                # Marin et al. 2024's MaleCNS circuit predicts opposing
                # tibia-motor effects for the two claw types: SNpp50 drives
                # extensor pathways (the resistance response to flexion),
                # whereas SNpp51 drives flexor pathways (extension).  This
                # deletes the selected fly's universal claw sign for the
                # evidence-covered rows.  Uncovered rows retain the explicit
                # source-theta assumption and are counted below.
                if "claw" in self.reflex_kinds and _ids[row] in self.reflex_map:
                    flex = self.reflex_map[_ids[row]] == "flex"
                    reflex_map_rows += 1
                elif self.claw_flex_frac is not None:
                    _f = self.claw_flex_frac[{"f": 0, "m": 1, "h": 2}[leg[1]]]
                    _k = _claw_k.get(leg, 0)
                    flex = int((_k + 1) * _f) > int(_k * _f)
                    _claw_k[leg] = _k + 1
                elif feco_polarity_mode == "banc_type" and ctype[row] in (
                        "SNpp50", "SNpp51"):
                    flex = ctype[row] == "SNpp50"
                    type_polarity_rows += 1
                elif claw_polarity is not None:
                    flex = claw_polarity > 0
                    if feco_polarity_mode == "banc_type":
                        type_polarity_fallback_rows += 1
                else:
                    flex = (row % 2 == 0) if srng is None \
                        else bool(srng.random() < .5)
                    if feco_polarity_mode == "banc_type":
                        type_polarity_fallback_rows += 1
                g = "claw_flex" if flex else "claw_ext"
                dof = f"{leg}_{FECO_DOF}"
            elif "hook" in s:
                # The same circuit reading assigns SNpp41 to the extensor
                # pathway (flexion signal) and SNpp39 to the flexor pathway
                # (extension signal).
                if "hook" in self.reflex_kinds and _ids[row] in self.reflex_map:
                    flex = self.reflex_map[_ids[row]] == "flex"
                    reflex_map_rows += 1
                elif feco_polarity_mode in ("banc_type", "banc_type_hook") \
                        and ctype[row] in ("SNpp41", "SNpp39"):
                    flex = ctype[row] == "SNpp41"
                    type_polarity_rows += 1
                elif hook_polarity is not None:
                    flex = hook_polarity > 0
                    if feco_polarity_mode == "banc_type":
                        type_polarity_fallback_rows += 1
                else:
                    flex = (row % 2 == 0) if srng is None \
                        else bool(srng.random() < .5)
                    if feco_polarity_mode == "banc_type":
                        type_polarity_fallback_rows += 1
                g = "hook_flex" if flex else "hook_ext"
                dof = f"{leg}_{FECO_DOF}"
            elif "club" in s or cls[row] == "chordotonal_organ_neuron":
                g = "club"
                dof = f"{leg}_{FECO_DOF}"
            elif cls[row] == "hair_plate_neuron":
                g = "hair"
                # CD1-KL (2026-08-24): hair_dof absent/None keeps every
                # hair plate on ThC_pitch = BIT-IDENTICAL. "spread"
                # distributes them index-ordered across all SEVEN of the
                # leg's DOF. A real fly's hair plates sit at several
                # joint articulations, and this class's own
                # `assumptions` dict already records the single-joint
                # pin as a guess BANC does not support. WHICH cell goes
                # to which joint is still a guess and is labelled one.
                if self._hair_dof == "spread":
                    # PER-LEG counter, as registered: "split index-
                    # ordered across all SEVEN of THAT LEG's DOF". A
                    # global counter gave 40/42 and 5/7 per leg, which
                    # is not what the registration says.
                    k = _hair_n.get(leg, 0)
                    dof = f"{leg}_{HAIR_SPREAD_DOF[k % 7]}"
                    _hair_n[leg] = k + 1
                else:
                    dof = f"{leg}_ThC_pitch"
            else:
                g = "campaniform"
                dof = f"{leg}_{FECO_DOF}"
            groups[g].append(row)
            self.dof_of[g].append(self.dof_index[dof])
            dof_by_row[int(row)] = self.dof_index[dof]

        self.groups = {k: np.array(v, dtype=np.int64) for k, v in groups.items()}
        self.dof_of = {k: np.array(v, dtype=np.int64)
                       for k, v in self.dof_of.items()}
        self.claw_thr = {}
        if self.claw_code == "fractionated":
            for g in ("claw_flex", "claw_ext"):
                dofs = self.dof_of[g]
                thr = np.zeros(len(dofs))
                for dof in np.unique(dofs):
                    k = np.flatnonzero(dofs == dof)   # index order within leg
                    n = len(k)
                    thr[k] = 0.5 + (np.arange(n) + 0.5) / n * 0.5
                self.claw_thr[g] = thr
        self.claw_u_a0, self.claw_u_sign = {}, {}
        self.claw_rank, self.claw_rank_n = {}, {}
        if self.claw_code in ("dallmann_u", "dallmann_u_recruit"):
            # a = a0 + sign x degrees(theta) per row, from its leg's convention
            for g in ("claw_flex", "claw_ext"):
                conv = [self.claw_u_angle[self._dof_name[int(i)].split("_", 1)[0]]
                        for i in self.dof_of[g]]
                minus = np.array([c == "180_minus_fti" for c in conv], dtype=bool)
                self.claw_u_a0[g] = np.where(minus, 180.0, 0.0)
                self.claw_u_sign[g] = np.where(minus, -1.0, 1.0)
        if self.claw_code == "dallmann_u_recruit":
            # bf165b63 Build 1: rank k = 1..n per leg in index order, as
            # "fractionated" ranks (np.flatnonzero(dofs == dof))
            for g in ("claw_flex", "claw_ext"):
                dofs = self.dof_of[g]
                rank = np.zeros(len(dofs), dtype=np.int64)
                n_of = np.zeros(len(dofs), dtype=np.int64)
                for dof in np.unique(dofs):
                    k = np.flatnonzero(dofs == dof)
                    rank[k] = np.arange(1, len(k) + 1)
                    n_of[k] = len(k)
                self.claw_rank[g], self.claw_rank_n[g] = rank, n_of
        self.n_unassigned = n_unassigned
        self.exact_modality_moved = exact_modality_moved
        self.type_polarity_rows = type_polarity_rows
        self.type_polarity_fallback_rows = type_polarity_fallback_rows
        self.all_rows = np.concatenate([v for v in self.groups.values()
                                        if len(v)])
        _order_position = {
            int(row): position for position, row in enumerate(self.all_rows)}
        _type_modality_rows = sorted(type_modality_sign_by_row)
        self._type_modality_positions = np.asarray([
            _order_position[row] for row in _type_modality_rows],
            dtype=np.int64)
        self._type_modality_dofs = np.asarray([
            dof_by_row[row] for row in _type_modality_rows], dtype=np.int64)
        self._type_modality_signs = np.asarray([
            type_modality_sign_by_row[row] for row in _type_modality_rows],
            dtype=np.float64)
        self.type_modality_rows = len(_type_modality_rows)
        self.type_modality_ids = [
            str(meta.iloc[row]["banc_888_id"]) for row in _type_modality_rows]
        self.type_modality_type_counts = {
            name: int(sum(ctype[row] == name for row in _type_modality_rows))
            for name in ("SNpp39", "SNpp41")}
        _claw_rows = np.concatenate((self.groups["claw_flex"],
                                     self.groups["claw_ext"]))
        if self._tonic_pops is not None:
            _claw_rows = np.concatenate([
                self._scoped_rows(k, self._tonic_scope.get(q))
                for q in self._tonic_pops
                for k in self.TONIC_POP_KEYS[q]])
        self.phasic_exempt_rows = int(len(_claw_rows))
        self._phasic_apply = ~np.isin(self.all_rows, _claw_rows)
        self._phasic_keep = None
        if self.type_modality_rows:
            # The corrected cells obey hook, not claw, high-pass exemptions.
            _hook_tonic = (self._tonic_pops is not None
                           and "hook" in self._tonic_pops)
            self._phasic_apply[self._type_modality_positions] = (
                not _hook_tonic)
            if (proprioceptor_phasic_mode == "claw_tonic"
                    or self._tonic_pops is not None):
                self.phasic_exempt_rows = int(
                    np.count_nonzero(~self._phasic_apply))
        if self._tonic_pops is not None:
            # The share of the RAW rate each row keeps: 1.0 for an exempt row,
            # 0.0 for a high-passed one, and the population's dose where one is
            # named. All ones and zeros is the exemption as it was.
            _keep = (~self._phasic_apply).astype(np.float64)
            for _q, _dv in self._tonic_dose.items():
                _rows = np.concatenate([
                    self._scoped_rows(k, self._tonic_scope.get(_q))
                    for k in self.TONIC_POP_KEYS[_q]])
                _keep[np.isin(self.all_rows, _rows)] = _dv
                if _q == "hook" and self.type_modality_rows:
                    _keep[self._type_modality_positions] = _dv
            if not np.array_equal(_keep, (~self._phasic_apply).astype(np.float64)):
                self._phasic_keep = _keep
            self.phasic_dose = dict(self._tonic_dose)
        self.assumptions = {
            "flex_ext_split": "by row parity — BANC names the subtype but not "
                              "which members are flexion- vs extension-tuned",
            "hair_plate_dof": "attached to the same leg's ThC_pitch — BANC "
                              "does not say which joint each hair plate sits "
                              "at",
            "campaniform_from_activation": "load encoded from |muscle "
                                           "activation|; this reduced body "
                                           "has no contact forces",
            "unassigned_by_nerve": n_unassigned,
            "feco_polarity_mode": feco_polarity_mode,
            "type_polarity_rows": type_polarity_rows,
            "type_polarity_fallback_rows": type_polarity_fallback_rows,
            "feco_modality_mode": feco_modality_mode,
            "type_modality_rows": self.type_modality_rows,
            "type_modality_type_counts": self.type_modality_type_counts,
            "proprioceptor_phasic_mode": proprioceptor_phasic_mode,
            "claw_code": claw_code,
            "claw_thresholds_by_index_order": claw_code == "fractionated",
            "claw_flex_frac": claw_flex_frac,
            "reflex_map": reflex_map,
            "reflex_map_rows": reflex_map_rows,
            "phasic_exempt_rows": (
                self.phasic_exempt_rows
                if proprioceptor_phasic_mode == "claw_tonic" else 0),
        }

    def counts(self):
        return {k: int(len(v)) for k, v in self.groups.items()}

    def _scoped_rows(self, key, segs):
        """A population's rows, or only those on the named segments' legs.

        Every afferent row was filed with the DOF it drives (`dof_of`), and
        every DOF name begins with its leg code, so the segment is read from
        the row's own DOF rather than from a second table.
        """
        rows = self.groups[key]
        if segs is None:
            return rows
        legs = {lg for s in segs for lg in self.SEGMENT_LEGS[s]}
        keep = np.array([self._dof_name[int(d)].split("_", 1)[0] in legs
                         for d in self.dof_of[key]], dtype=bool)
        if not keep.any():
            raise ValueError(f"tonic scope {segs!r} selects no {key} rows")
        return rows[keep]

    # bf165b63 Build 2: the front leg's lo and width in the U's convention as
    # hypothesis H03 read them (critique d9ce3948); stated, not searched
    CLAW_U_MAP_LO, CLAW_U_MAP_WIDTH = 55.9, 51.6

    def claw_u_span(self, g, body):
        """bf165b63 Build 2: (lo, hi) per row of claw group g, the row's FTi
        range from body.lo and body.hi in the U's angle convention."""
        if self._claw_u_span is None or self._claw_u_span[0] is not body:
            span = {}
            for gg in ("claw_flex", "claw_ext"):
                d = self.dof_of[gg]
                ends = (self.claw_u_a0[gg][:, None] + self.claw_u_sign[gg][:, None]
                        * np.degrees(np.stack([np.asarray(body.lo, dtype=np.float64)[d],
                                               np.asarray(body.hi, dtype=np.float64)[d]], axis=1)))
                span[gg] = (ends.min(axis=1), ends.max(axis=1))
            self._claw_u_span = (body, span)
        return self._claw_u_span[1][g]

    def claw_u_mapped(self, g, a, body):
        """bf165b63 Build 2: a' = 55.9 + clip((a - lo) / (hi - lo), 0, 1) x 51.6."""
        lo, hi = self.claw_u_span(g, body)
        return self.CLAW_U_MAP_LO + np.clip((a - lo) / (hi - lo), 0.0, 1.0) * self.CLAW_U_MAP_WIDTH

    def rates(self, body):
        """-> (row_indices, rates_hz) for every proprioceptor."""
        pos = body.normalised()
        vel = body.vel
        act = np.abs(body.a)
        out_idx, out_rate = [], []

        def emit(key, r, r_max=self.R_MAX):
            if len(self.groups[key]):
                out_idx.append(self.groups[key])
                out_rate.append(np.clip(r, 0.0, r_max))

        d = self.dof_of
        _cg = self.claw_gain
        if self.claw_code == "fractionated":
            # CD1-TN: per-cell sigmoid at that cell's tiled threshold.
            _w = self.claw_width
            for g, x in (("claw_flex", pos[d["claw_flex"]]),
                         ("claw_ext", 1 - pos[d["claw_ext"]])):
                z = np.clip((x - self.claw_thr[g]) / _w, -60.0, 60.0)
                emit(g, self.R_REST + _cg * self.R_MAX / (1.0 + np.exp(-z)),
                     self.R_MAX * _cg)
        elif self.claw_code == "dallmann_u":
            # 2284c922 B1: tonic U per row, capped at R_REST + G (never clips);
            # the hand-over at the U's minimum (amendment 23cbec94); G is the
            # group's own level (cda7e472 Build 6)
            _off, _am = self.claw_u_offset_deg, self.claw_u_amin
            for g in ("claw_flex", "claw_ext"):
                _g = self.claw_u_hz_of[g]
                a = self.claw_u_a0[g] + self.claw_u_sign[g] * np.degrees(
                    body.theta[d[g]])
                if self.claw_u_range:
                    a = self.claw_u_mapped(g, a, body)
                on = (a <= _am) if g == "claw_flex" else (a >= _am)
                u = dallmann_u(a, self.claw_u_coeffs, _off, self.claw_u_pmin)
                emit(g, self.R_REST + _g * np.where(on, u, 0.0),
                     self.R_REST + _g)
        elif self.claw_code == "dallmann_u_recruit":
            # bf165b63 Build 1: the U's level as the share of the leg's rows
            # at R_MAX, ranked in index order; f uses the group's own level
            # (cda7e472 Build 6)
            _off, _am = self.claw_u_offset_deg, self.claw_u_amin
            for g in ("claw_flex", "claw_ext"):
                _g = self.claw_u_hz_of[g]
                a = self.claw_u_a0[g] + self.claw_u_sign[g] * np.degrees(
                    body.theta[d[g]])
                if self.claw_u_range:
                    a = self.claw_u_mapped(g, a, body)
                on = (a <= _am) if g == "claw_flex" else (a >= _am)
                u = dallmann_u(a, self.claw_u_coeffs, _off, self.claw_u_pmin)
                f = np.clip(_g * np.where(on, u, 0.0) / (self.R_MAX - self.R_REST),
                            0.0, 1.0)
                lit = self.claw_rank[g] <= np.floor(f * self.claw_rank_n[g])
                emit(g, np.where(lit, self.R_MAX, self.R_REST))
        else:
            emit("claw_flex", self.R_REST + _cg * self.R_MAX * pos[d["claw_flex"]],
                 self.R_MAX * _cg)
            emit("claw_ext", self.R_REST + _cg * self.R_MAX * (1 - pos[d["claw_ext"]]),
                 self.R_MAX * _cg)
        emit("hook_flex", self.r_rest_hook + self.R_MAX
             * np.maximum(0.0, vel[d["hook_flex"]]) / self.v_scale_hook)
        emit("hook_ext", self.r_rest_hook + self.R_MAX
             * np.maximum(0.0, -vel[d["hook_ext"]]) / self.v_scale_hook)
        emit("club", self.R_REST + self.R_MAX
             * np.abs(vel[d["club"]]) / self.v_scale_club)
        # hair plates: silent in mid-range, firing as the joint nears a limit
        p = pos[d["hair"]]
        emit("hair", self.r_rest_hair + self.R_MAX
             * np.maximum(0.0, np.abs(2 * p - 1) - 0.7) / 0.3)
        if self.camp_src == "rest":
            # CD1: activation channel off. Constant rest rate keeps the
            # entry count and RNG draw structure identical; under the
            # phasic transform a constant input's residual is 0 from the
            # first tick (EMA initialises at the first rates vector).
            emit("campaniform",
                 np.full(len(d["campaniform"]), self.R_REST))
        else:
            emit("campaniform", self.R_REST + self.R_MAX
                 * np.minimum(1.0,
                              act[d["campaniform"]] / (2 * body.drive_scale)))
        idx_all = np.concatenate(out_idx)
        r_all = np.concatenate(out_rate)
        if self.type_modality_rows:
            _signed_velocity = (self._type_modality_signs
                                * vel[self._type_modality_dofs])
            r_all[self._type_modality_positions] = np.clip(
                self.r_rest_hook + self.R_MAX * np.maximum(0.0, _signed_velocity)
                / self.v_scale_hook,
                0.0, self.R_MAX)
        if self.phasic > 0.0:
            if self._ema is None:
                self._ema = r_all.copy()
            self._ema += self._ema_alpha * (r_all - self._ema)
            transformed = np.clip((r_all - self.phasic * self._ema)
                                  * (1.0 + 2.0 * self.phasic), 0.0, 400.0)
            if (self.proprioceptor_phasic_mode == "claw_tonic"
                    or self._tonic_pops is not None):
                # Mamiya 2018: claw is a sustained position channel, unlike
                # the phasic hook/club movement channels.  Keep the incumbent
                # high-pass everywhere else and delete it only from claw.
                if self._phasic_keep is None:
                    r_all = np.where(self._phasic_apply, transformed, r_all)
                else:
                    r_all = (self._phasic_keep * r_all
                             + (1.0 - self._phasic_keep) * transformed)
            else:
                r_all = transformed
        return idx_all, r_all
