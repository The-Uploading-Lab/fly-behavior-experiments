#!/usr/bin/env python3
"""FL2b: the flybody body's flight control derivatives, its hover trim, and a scripted hover.

FL2 (lanes/A/2026-09-09-fl2-flybody-lift.json) showed that the body lifts 93% of its
weight with the animal's hover kinematics, that the mean air force passes 0.41 mm behind
the centre of mass (+3.68 nN m nose-down), and that the released body tumbles within five
wingbeats. Robin's order of work is top-down: before the CNS is wired to the wings, the
body has to state what the motor layer must deliver. This runner measures exactly that,
with no nervous system. It is the labelled body-calibration control the goal names:
"prescribed wing motion or an external controller is a labeled body-calibration control.
It does not establish flight driven by the connectome."

Body, air, servo: FL2's rig unchanged (flybody flight configuration inside flygym's
composer, ellipsoid wing-fluid model, kp 1800 servos with the 1800 torque limit,
dt 5e-5 s, control 2e-4 s). Kinematics: flybody's Dickson-2008 hover cycle with three
dials on the stroke (wing yaw) angle of each wing: amplitude scale a about the cycle
mean, a constant fore-aft shift delta, and the wingbeat frequency f. Stroke deviation
and wing rotation stay the pattern's own.

Part 1, derivatives (tethered, 20 beats per point, last 10 read):
  a in {0.90..1.20}, delta in {-20..+20 deg}, f in {200..240 Hz}, and a left-right
  amplitude asymmetry D in {-0.10..+0.10} (a_l = 1 + D/2, a_r = 1 - D/2). Per point:
  mean air force (uN, world), torque about the CoM (nN m), |F|/weight, Fz/weight,
  fore-aft offset of the force line -tau_y/Fz (mm), aero power, achieved stroke amplitude
  and mean, servo-saturation and joint-limit fractions, per-cycle Fz sd.
Part 2, trim: a 7 x 9 grid over (a, delta) at 218 Hz, quadratic response surface, Newton
  solve of |F| = weight and tau_y = 0, then Newton on DIRECT residuals until |F|/W is
  within 0.5% and |tau_y| < 0.05 nN m. theta0 = -atan(Fx/Fz) is the body-pitch change at
  which the trimmed force is vertical. Time-step check at the trim: 1e-4, 5e-5, 2.5e-5 s.
  ADDED AFTER THE SMOKE RUN (15:5x, declared): the two-dial trim leaves 20% of the weight
  as forward thrust, so the trimmed force points 12 deg ahead of flybody's hover axis and a
  body that nulls the thrust flies at 59 deg pitch, outside the pre-set 47.5 +- 10 deg
  band for a frame reason. flybody's own `stroke_plane_angle` (fruitfly.py, default 0)
  is therefore the third dial: a three-dial Newton on direct residuals solves |F| = W,
  tau_y = 0 and Fx = 0 at 47.5 deg body pitch. The two-dial arm stays and is reported.
Part 3, release (EXTERNAL CONTROLLER, body calibration only):
  open_loop    three-dial trim, no feedback, 0.5 s: time to 5/10/45/90 deg of pitch and
               the growth rate of the pitch excursion (Ristroph et al. 2013 T_INST).
  closed_loop_flybody_frame  two-dial trim (stroke plane 0), plus the controller below.
  closed_loop  three-dial trim plus a beat-synchronous PD: once per wingbeat, from the
               previous beat's mean body state, stroke shift <- pitch (kp = I_yy w^2,
               kd = 2 zeta I_yy w, w = 2 pi 12 Hz, zeta 0.8), amplitude <- altitude
               (2 Hz, zeta 1), pitch set-point <- horizontal position (0.7 Hz, zeta 1)
               about theta0. 1.0 s of hover, then a 6 nN m, 5 ms nose-down torque pulse
               on the thorax and 0.5 s of recovery (Ristroph et al. 2013: one-beat
               magnetic pulses, 5-25 deg of pitch, recovery in about 60 ms / 15 beats,
               reaction time 13 +- 2 ms).
  delay scan   closed_loop with a pure delay D in {0..40 ms} between a beat's state and
               its command, each at bandwidths {4..16 Hz}: the largest D that hovers.
Pre-set readings, written before the run:
  TRIM     a solution inside the scanned box, verified by a direct run: |F|/W within
           0.5% and |tau_y| < 0.05 nN m (force line within 0.005 mm of the CoM).
  HOVERS   closed_loop at D = 0: CoM altitude within 2 mm of the release altitude AND
           body pitch within 10 deg of 47.5 deg for the whole 1.0 s (218 beats).
           One of the two: PARTIAL. Neither: FALLS.
  D_MAX    the largest scanned delay that still HOVERS at its best bandwidth: the latency
           budget the FL3 haltere -> steering-motor-neuron path has to meet.
Units mm-g-s: force g mm/s2 = uN, torque g mm2/s2 = nN m, power uW. No seed, no CNS, so
no Metal identity; the engine is MuJoCo, reported by version. Nothing outside lanes/A.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import mujoco as mj

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lanes.A import fl2_flybody_lift as FL2  # noqa: E402

DT = FL2.PHYSICS_DT
CONTROL_DT = FL2.CONTROL_DT
F0 = FL2.BASE_FREQ_HZ
BODY_PITCH_DEG = FL2.BODY_PITCH_DEG
G = FL2.GRAVITY
STROKE_RANGE_RAD = 1.5                 # flybody wing yaw joint range, +-1.5 rad

# scan design
AMP_SCALES = (0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20)
SHIFTS_DEG = (-20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0)
FREQS_HZ = (200.0, 210.0, 218.0, 230.0, 240.0)
ASYMMETRIES = (-0.10, -0.05, 0.0, 0.05, 0.10)
SCAN_CYCLES, SCAN_READ = 20, 10
TRIM_TOL_FORCE, TRIM_TOL_TAU = 0.005, 0.05     # |F|/W - 1, nN m
TRIM_MAX_ITER = 6
DT_CHECK = (1e-4, 5e-5, 2.5e-5)

# release protocol
PRE_RELEASE_CYCLES = 10
OPEN_LOOP_S = 0.5
HOVER_S = 1.0
KICK_NNM, KICK_AT_S, KICK_S, RECOVER_S = 6.0, 1.0, 0.005, 0.5
# three-dial trim (stroke plane added after the smoke run, see the docstring)
TRIM3_STEPS = (0.02, math.radians(2.0), 2.0)   # finite-difference steps: amp, shift (rad), stroke plane (deg)
TRIM3_TOL_FX = 0.005                            # |Fx|/W
TRIM3_MAX_ITER = 8
STROKE_PLANE_BOX_DEG = (-30.0, 30.0)
ALT_TOL_MM, PITCH_TOL_DEG = 2.0, 10.0
RECOVERED_DEG = 2.0

# controller design
PITCH_BW_HZ, PITCH_ZETA = 12.0, 0.8
ALT_BW_HZ, ALT_ZETA = 2.0, 1.0
POS_BW_HZ, POS_ZETA = 0.7, 1.0
SHIFT_MAX_RAD = math.radians(20.0)             # the scanned range
AMP_MIN, AMP_MAX = 0.90, 1.20                  # the scanned range
THETA_REF_MAX_RAD = math.radians(15.0)
DELAYS_MS = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0)
BANDWIDTHS_HZ = (4.0, 6.0, 8.0, 12.0, 16.0)

UW = FL2.UW_PER_UNIT

# free-flight record columns
C_T, C_F, C_TAU, C_COM, C_V, C_TH, C_THD, C_PITCH, C_ROLL, C_YAW = (
    0, slice(1, 4), slice(4, 7), slice(7, 10), slice(10, 13), 13, 14, 15, 16, 17)
C_AL, C_AR, C_SL, C_SR, C_TREF, C_KICK, C_PA, C_HZ = 18, 19, 20, 21, 22, 23, 24, 25
C_COLS = 26


# ---- kinematics ------------------------------------------------------------------------
@dataclass(frozen=True)
class Kin:
    """Three dials per wing on flybody's hover cycle."""
    freq_hz: float = F0
    amp: tuple = (1.0, 1.0)        # (left, right) stroke amplitude scale about the cycle mean
    shift: tuple = (0.0, 0.0)      # (left, right) constant added to the stroke angle, rad

    def sym(self, amp: float, shift: float) -> "Kin":
        return Kin(self.freq_hz, (amp, amp), (shift, shift))


class Pattern:
    def __init__(self, pattern: np.ndarray):
        self.p = pattern
        self.stroke_mean = float(pattern[:, 0].mean())

    def angles(self, phase: float, kin: Kin) -> np.ndarray:
        """Six wing angles [l yaw, l roll, l pitch, r yaw, r roll, r pitch] at a cycle phase."""
        base = FL2.pattern_at(self.p, phase)
        out = np.tile(base, 2)
        dev = base[0] - self.stroke_mean
        out[0] = self.stroke_mean + kin.amp[0] * dev + kin.shift[0]
        out[3] = self.stroke_mean + kin.amp[1] * dev + kin.shift[1]
        return out


_RIGS: dict = {}


def make_rig(stroke_plane_deg: float = 0.0, dt: float = DT) -> FL2.Rig:
    """FL2's rig with flybody's `stroke_plane_angle` (fruitfly.py, default 0) set.

    FL2.set_flight_frames reads the module constant STROKE_PLANE_DEG when the wing
    frames are built, so it is set for the build and restored. Rigs are cached per
    (angle, dt): the model is deterministic and `reset` restores the state.
    """
    key = (round(stroke_plane_deg, 6), dt, FL2.LEG_POSE)
    if key not in _RIGS:
        old = FL2.STROKE_PLANE_DEG
        FL2.STROKE_PLANE_DEG = stroke_plane_deg
        try:
            rig = FL2.Rig("ellipsoid", dt=dt)
        finally:
            FL2.STROKE_PLANE_DEG = old
        rig.stroke_plane_deg = stroke_plane_deg
        _RIGS[key] = rig
    return _RIGS[key]


def reset(rig: FL2.Rig, pat: Pattern, kin: Kin) -> None:
    d = rig.data
    mj.mj_resetData(rig.model, d)
    d.qpos[rig.root_qpos: rig.root_qpos + 7] = rig.spawn_qpos
    a0 = pat.angles(0.0, kin)
    a1 = pat.angles(rig.dt * kin.freq_hz, kin)
    d.qpos[rig.wing_qpos] = a0
    d.qvel[rig.wing_dof] = (a1 - a0) / rig.dt
    d.ctrl[rig.actuators] = a0
    mj.mj_forward(rig.model, d)


# ---- body state ------------------------------------------------------------------------
def inertia_about_com(rig: FL2.Rig) -> np.ndarray:
    """Whole-fly inertia tensor about its CoM, world frame at the current pose (g mm2)."""
    m, d = rig.model, rig.data
    com = d.subtree_com[rig.thorax]
    I = np.zeros((3, 3))
    for b in range(1, m.nbody):
        R = d.ximat[b].reshape(3, 3)
        r = d.xipos[b] - com
        I += R @ np.diag(m.body_inertia[b]) @ R.T
        I += m.body_mass[b] * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    return I


@dataclass
class BodyState:
    com: np.ndarray
    v: np.ndarray
    theta: float            # pitch tilt from the spawn attitude, rad, nose-down positive
    theta_dot: float        # rad/s, about world +y (left), nose-down positive
    body_pitch_deg: float   # flybody definition: thorax x axis above horizontal (47.5 in hover)
    roll_deg: float         # left side up positive
    yaw_deg: float          # heading of the thorax x axis, 0 = spawn (+x)
    hover_z: float


def body_state(rig: FL2.Rig) -> BodyState:
    d = rig.data
    R = d.xmat[rig.thorax].reshape(3, 3)
    fwd, left = R[:, 0], R[:, 1]
    body_pitch = math.degrees(math.atan2(fwd[2], math.hypot(fwd[0], fwd[1])))
    yaw = math.degrees(math.atan2(fwd[1], fwd[0]))
    roll = math.degrees(math.asin(max(-1.0, min(1.0, left[2]))))
    w_world = R @ d.qvel[rig.root_dof + 3: rig.root_dof + 6]
    return BodyState(com=d.subtree_com[rig.thorax].copy(),
                     v=d.qvel[rig.root_dof: rig.root_dof + 3].copy(),
                     theta=math.radians(BODY_PITCH_DEG - body_pitch), theta_dot=float(w_world[1]),
                     body_pitch_deg=body_pitch, roll_deg=roll, yaw_deg=yaw,
                     hover_z=float(rig.hover_axis_world()[2]))


# ---- external controller (labelled) -----------------------------------------------------------
class BeatController:
    """Beat-synchronous PD on the three dials the body exposes. EXTERNAL: no nervous system.

    Once per wingbeat it reads the mean body state over the beat just finished (a one-beat
    boxcar removes the 218 Hz torque ripple, 16 nN m peak in FL2) and issues one command
    that holds for the next beat, after a pure delay `delay_s`. Gains follow from the
    measured derivatives and the body's inertia, so the loop is stated in physical terms:
      stroke shift   tau_des = -(kp (theta - theta_ref) + kd theta_dot), shift = trim + tau_des / dtau_dshift
      amplitude      F_des = W + m (kz (z_ref - z) - kvz vz),            amp = trim + (F_des - W) / dF_damp
      pitch set-point theta_ref = theta0 - (kx (x - x_ref) + kvx vx) / g  (nose-up brakes forward drift)
    """

    def __init__(self, trim: Kin, dtau_dshift: float, dF_damp: float, I_yy: float, mass_g: float,
                 theta0: float, z_ref: float, x_ref: float, pitch_bw_hz: float = PITCH_BW_HZ,
                 delay_s: float = 0.0, position_loop: bool = True):
        self.trim, self.dtau_dshift, self.dF_damp = trim, dtau_dshift, dF_damp
        self.mass_g, self.weight = mass_g, mass_g * G
        w = 2 * math.pi * pitch_bw_hz
        self.kp, self.kd = I_yy * w * w, 2 * PITCH_ZETA * I_yy * w
        wz = 2 * math.pi * ALT_BW_HZ
        self.kz, self.kvz = wz * wz, 2 * ALT_ZETA * wz
        wx = 2 * math.pi * POS_BW_HZ
        self.kx, self.kvx = wx * wx, 2 * POS_ZETA * wx
        self.theta0, self.z_ref, self.x_ref = theta0, z_ref, x_ref
        self.delay_s, self.position_loop = delay_s, position_loop
        self.period = 1.0 / trim.freq_hz
        self.next_beat = self.period
        self.samples: list = []
        self.queue: deque = deque()
        self.theta_ref = theta0
        self.cmd = trim
        self.n_updates = 0
        self.clamped_shift = self.clamped_amp = 0

    def update(self, st: BodyState, t: float) -> Kin:
        self.samples.append([st.theta, st.theta_dot, st.com[2], st.v[2], st.com[0], st.v[0]])
        if t >= self.next_beat - 1e-9:
            self.next_beat += self.period
            theta, theta_dot, z, vz, x, vx = np.mean(self.samples, axis=0)
            self.samples = []
            if self.position_loop:
                ref = self.theta0 - (self.kx * (x - self.x_ref) + self.kvx * vx) / G
                self.theta_ref = float(np.clip(ref, -THETA_REF_MAX_RAD, THETA_REF_MAX_RAD))
            tau_des = -(self.kp * (theta - self.theta_ref) + self.kd * theta_dot)
            shift = self.trim.shift[0] + tau_des / self.dtau_dshift
            lo, hi = self.trim.shift[0] - SHIFT_MAX_RAD, self.trim.shift[0] + SHIFT_MAX_RAD
            if shift < lo or shift > hi:
                self.clamped_shift += 1
            shift = float(np.clip(shift, lo, hi))
            F_des = self.weight + self.mass_g * (self.kz * (self.z_ref - z) - self.kvz * vz)
            amp = self.trim.amp[0] + (F_des - self.weight) / self.dF_damp
            if amp < AMP_MIN or amp > AMP_MAX:
                self.clamped_amp += 1
            amp = float(np.clip(amp, AMP_MIN, AMP_MAX))
            self.queue.append((t + self.delay_s, self.trim.sym(amp, shift)))
            self.n_updates += 1
        while self.queue and self.queue[0][0] <= t + 1e-9:
            self.cmd = self.queue.popleft()[1]
        return self.cmd


class Kick:
    """A torque pulse on the thorax about world +y (nose-down positive), labelled disturbance."""

    def __init__(self, torque_nNm: float, at_s: float, dur_s: float):
        self.torque, self.at, self.dur = torque_nNm, at_s, dur_s

    def apply(self, rig: FL2.Rig, t: float) -> bool:
        on = self.at <= t < self.at + self.dur
        rig.data.xfrc_applied[rig.thorax, 3:6] = (0.0, self.torque if on else 0.0, 0.0)
        return on


# ---- the run -------------------------------------------------------------------------------
def run(rig: FL2.Rig, pat: Pattern, kin: Kin, tether_cycles: int, free_s: float,
        controller: BeatController | None = None, kick: Kick | None = None,
        frame_hook=None) -> dict:
    m, d = rig.model, rig.data
    steps_per_ctrl = max(1, int(round(CONTROL_DT / rig.dt)))
    reset(rig, pat, kin)
    period = 1.0 / kin.freq_hz
    t_release = tether_cycles * period
    t_end = t_release + free_s
    teth, free = [], []
    step, released, cmd, kick_on = 0, False, kin, False
    while d.time < t_end - 0.5 * rig.dt:
        if step % steps_per_ctrl == 0:
            if released and controller is not None:
                cmd = controller.update(body_state(rig), d.time - t_release)
            d.ctrl[rig.actuators] = pat.angles(d.time * cmd.freq_hz, cmd)
        if kick is not None and released:
            kick_on = kick.apply(rig, d.time - t_release)
        mj.mj_step(m, d)
        step += 1
        F, tau = rig.fluid_force_torque()
        p_servo, p_aero, p_damp = rig.powers()
        if not released:
            row = np.empty(FL2.T_COLS)
            row[FL2.T_T] = d.time
            row[FL2.T_F] = F
            row[FL2.T_TAU] = tau
            row[FL2.T_Q] = d.qpos[rig.wing_qpos]
            row[FL2.T_CTRL] = d.ctrl[rig.actuators]
            row[FL2.T_FORCE] = d.actuator_force[rig.actuators]
            row[FL2.T_PS], row[FL2.T_PA], row[FL2.T_PD] = p_servo, p_aero, p_damp
            row[FL2.T_TIP] = rig.tip("l")
            teth.append(row)
            d.qpos[rig.root_qpos: rig.root_qpos + 7] = rig.spawn_qpos
            d.qvel[rig.root_dof: rig.root_dof + 6] = 0.0
            if d.time >= t_release - 0.5 * rig.dt:
                released = True
                mj.mj_forward(m, d)
        else:
            st = body_state(rig)
            row = np.empty(C_COLS)
            row[C_T] = d.time - t_release
            row[C_F], row[C_TAU], row[C_COM], row[C_V] = F, tau, st.com, st.v
            row[C_TH], row[C_THD], row[C_PITCH], row[C_ROLL], row[C_YAW] = (
                st.theta, st.theta_dot, st.body_pitch_deg, st.roll_deg, st.yaw_deg)
            row[C_AL], row[C_AR] = cmd.amp
            row[C_SL], row[C_SR] = cmd.shift
            row[C_TREF] = controller.theta_ref if controller is not None else 0.0
            row[C_KICK], row[C_PA], row[C_HZ] = float(kick_on), p_aero, st.hover_z
            free.append(row)
        if frame_hook is not None:
            frame_hook(rig, released, cmd, controller)
    return {"tethered": np.array(teth), "free": np.array(free).reshape(-1, C_COLS),
            "period_s": period, "steps": step}


# ---- tethered readout -------------------------------------------------------------------------
def kin_dict(kin: Kin) -> dict:
    return {"freq_hz": kin.freq_hz, "amp_l": kin.amp[0], "amp_r": kin.amp[1],
            "shift_l_deg": math.degrees(kin.shift[0]), "shift_r_deg": math.degrees(kin.shift[1])}


def tethered_point(rig: FL2.Rig, pat: Pattern, kin: Kin, cycles: int = SCAN_CYCLES,
                   read: int = SCAN_READ) -> dict:
    out = run(rig, pat, kin, cycles, 0.0)
    lift = FL2.tethered_lift(rig, out, read)
    T = out["tethered"]
    t = T[:, FL2.T_T]
    w = t >= t[-1] - read * out["period_s"]
    q, ctrl, force = T[w, FL2.T_Q], T[w, FL2.T_CTRL], T[w, FL2.T_FORCE]
    F = np.array([lift["mean_fluid_force_uN"][k] for k in "xyz"])
    tau = np.array([lift["mean_torque_about_com_nNm_world"][k] for k in "xyz"])
    return {
        "kin": kin_dict(kin), "stroke_plane_deg": float(getattr(rig, "stroke_plane_deg", 0.0)),
        "F_uN": F.tolist(), "tau_nNm": tau.tolist(),
        "Fx_over_weight": float(F[0] / rig.weight_uN),
        "force_over_weight": float(np.linalg.norm(F) / rig.weight_uN),
        "lift_over_weight": float(F[2] / rig.weight_uN),
        "force_line_offset_mm": float(-tau[1] / F[2]) if abs(F[2]) > 1e-9 else float("nan"),
        "per_cycle_Fz_sd_uN": lift["per_cycle_Fz_sd_uN"],
        "aero_power_uW": lift["aero_power_uW"]["mean"],
        "l_stroke_cmd_amp_rad": float(ctrl[:, 0].max() - ctrl[:, 0].min()),
        "l_stroke_achieved_amp_rad": float(q[:, 0].max() - q[:, 0].min()),
        "l_stroke_cmd_mean_rad": float(ctrl[:, 0].mean()),
        "l_stroke_achieved_mean_rad": float(q[:, 0].mean()),
        "r_stroke_achieved_amp_rad": float(q[:, 3].max() - q[:, 3].min()),
        "l_stroke_saturated_fraction": float(np.mean(np.abs(force[:, 0]) >= 0.999 * FL2.WING_GAIN)),
        "l_stroke_at_joint_limit_fraction": float(np.mean(np.abs(q[:, 0]) >= 0.995 * STROKE_RANGE_RAD)),
        "steps": out["steps"],
    }


def slope(xs, ys) -> dict:
    """Least-squares line y = y0 + k x with residual rms; x and y as given."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    A = np.vstack([np.ones_like(xs), xs]).T
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    resid = ys - A @ coef
    return {"intercept": float(coef[0]), "slope": float(coef[1]),
            "residual_rms": float(np.sqrt(np.mean(resid ** 2)))}


# ---- trim ------------------------------------------------------------------------------------
def quad_features(a: float, s: float) -> np.ndarray:
    da = a - 1.0
    return np.array([1.0, da, s, da * da, da * s, s * s])


def quad_grad(coef: np.ndarray, a: float, s: float) -> np.ndarray:
    da = a - 1.0
    return np.array([coef[1] + 2 * coef[3] * da + coef[4] * s,
                     coef[2] + coef[4] * da + 2 * coef[5] * s])


def fit_surface(points: list, key) -> np.ndarray:
    X = np.array([quad_features(p["kin"]["amp_l"], math.radians(p["kin"]["shift_l_deg"])) for p in points])
    y = np.array([key(p) for p in points])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def solve_trim(rig: FL2.Rig, pat: Pattern, grid: list, log) -> dict:
    """Newton on the quadratic surfaces, then Newton on direct residuals."""
    W = rig.weight_uN
    c_force = fit_surface(grid, lambda p: p["force_over_weight"] - 1.0)
    c_tau = fit_surface(grid, lambda p: p["tau_nNm"][1])
    fit_rms = {"force_over_weight": float(np.sqrt(np.mean([(quad_features(p["kin"]["amp_l"], math.radians(p["kin"]["shift_l_deg"])) @ c_force - (p["force_over_weight"] - 1.0)) ** 2 for p in grid]))),
               "tau_y_nNm": float(np.sqrt(np.mean([(quad_features(p["kin"]["amp_l"], math.radians(p["kin"]["shift_l_deg"])) @ c_tau - p["tau_nNm"][1]) ** 2 for p in grid])))}
    a, s = 1.0, 0.0
    for _ in range(30):
        r = np.array([quad_features(a, s) @ c_force, quad_features(a, s) @ c_tau])
        J = np.vstack([quad_grad(c_force, a, s), quad_grad(c_tau, a, s)])
        da, ds = np.linalg.solve(J, -r)
        a, s = a + da, s + ds
        if abs(da) < 1e-9 and abs(ds) < 1e-9:
            break
    surface_solution = {"amp": a, "shift_deg": math.degrees(s)}
    log(f"[trim] surface solution a {a:.4f}, shift {math.degrees(s):+.2f} deg "
        f"(fit rms |F|/W {fit_rms['force_over_weight']:.4f}, tau_y {fit_rms['tau_y_nNm']:.3f} nN m)")
    history = []
    point = None
    for it in range(TRIM_MAX_ITER):
        point = tethered_point(rig, pat, Kin(F0).sym(a, s))
        r = np.array([point["force_over_weight"] - 1.0, point["tau_nNm"][1]])
        history.append({"iter": it, "amp": a, "shift_deg": math.degrees(s),
                        "force_over_weight": point["force_over_weight"], "tau_y_nNm": point["tau_nNm"][1]})
        log(f"[trim] direct {it}: a {a:.4f}, shift {math.degrees(s):+.2f} deg -> |F|/W {point['force_over_weight']:.4f}, "
            f"tau_y {point['tau_nNm'][1]:+.3f} nN m, Fx {point['F_uN'][0]:+.3f} uN")
        if abs(r[0]) <= TRIM_TOL_FORCE and abs(r[1]) <= TRIM_TOL_TAU:
            break
        J = np.vstack([quad_grad(c_force, a, s), quad_grad(c_tau, a, s)])
        da, ds = np.linalg.solve(J, -r)
        a, s = float(np.clip(a + da, AMP_SCALES[0], AMP_SCALES[-1])), float(np.clip(s + ds, math.radians(SHIFTS_DEG[0]), math.radians(SHIFTS_DEG[-1])))
    F = np.array(point["F_uN"])
    theta0 = -math.atan2(F[0], F[2])     # nose-down positive; forward thrust needs nose-up
    J = np.vstack([quad_grad(c_force, a, s), quad_grad(c_tau, a, s)])
    inside = AMP_SCALES[0] <= a <= AMP_SCALES[-1] and SHIFTS_DEG[0] <= math.degrees(s) <= SHIFTS_DEG[-1]
    ok = inside and abs(point["force_over_weight"] - 1.0) <= TRIM_TOL_FORCE and abs(point["tau_nNm"][1]) <= TRIM_TOL_TAU
    return {
        "kin": Kin(F0).sym(a, s), "amp": a, "shift_deg": math.degrees(s), "shift_rad": s,
        "point": point, "surface_solution": surface_solution, "surface_fit_rms": fit_rms,
        "history": history, "inside_scanned_box": bool(inside), "verified": bool(ok),
        "theta0_deg": math.degrees(theta0), "theta0_rad": theta0,
        "hover_body_pitch_deg": BODY_PITCH_DEG - math.degrees(theta0),
        "local_derivatives_at_trim": {
            "d_force_over_weight_d_amp": float(J[0, 0]), "d_force_over_weight_d_shift_per_rad": float(J[0, 1]),
            "d_tau_y_d_amp_nNm": float(J[1, 0]), "d_tau_y_d_shift_nNm_per_rad": float(J[1, 1]),
            "d_tau_y_d_shift_nNm_per_deg": float(J[1, 1] * math.pi / 180.0),
            "dF_d_amp_uN": float(J[0, 0] * W)},
        "verdict": "TRIM" if ok else "NO_TRIM",
    }


def solve_trim3(pat: Pattern, start: dict, log) -> dict:
    """Three dials (amp, shift, stroke plane) for three targets (|F| = W, tau_y = 0, Fx = 0)
    at 47.5 deg body pitch: Newton on direct tethered residuals with a finite-difference
    Jacobian, starting from the two-dial trim."""
    x = np.array([start["amp"], start["shift_rad"], 0.0])
    lo = np.array([AMP_SCALES[0], math.radians(SHIFTS_DEG[0]), STROKE_PLANE_BOX_DEG[0]])
    hi = np.array([AMP_SCALES[-1], math.radians(SHIFTS_DEG[-1]), STROKE_PLANE_BOX_DEG[1]])
    steps = np.array(TRIM3_STEPS)
    history, point, J = [], None, None

    def residual(v):
        p = tethered_point(make_rig(float(v[2])), pat, Kin(F0).sym(float(v[0]), float(v[1])))
        return np.array([p["force_over_weight"] - 1.0, p["tau_nNm"][1], p["Fx_over_weight"]]), p

    for it in range(TRIM3_MAX_ITER):
        r, point = residual(x)
        history.append({"iter": it, "amp": float(x[0]), "shift_deg": math.degrees(x[1]), "stroke_plane_deg": float(x[2]),
                        "force_over_weight": point["force_over_weight"], "tau_y_nNm": point["tau_nNm"][1],
                        "Fx_over_weight": point["Fx_over_weight"]})
        log(f"[trim3] {it}: a {x[0]:.4f}, shift {math.degrees(x[1]):+.2f} deg, stroke plane {x[2]:+.2f} deg -> "
            f"|F|/W {point['force_over_weight']:.4f}, tau_y {point['tau_nNm'][1]:+.3f} nN m, Fx/W {point['Fx_over_weight']:+.4f}")
        done = abs(r[0]) <= TRIM_TOL_FORCE and abs(r[1]) <= TRIM_TOL_TAU and abs(r[2]) <= TRIM3_TOL_FX
        J = np.zeros((3, 3))
        for k in range(3):
            xk = x.copy()
            xk[k] += steps[k]
            rk, _ = residual(xk)
            J[:, k] = (rk - r) / steps[k]
        if done:
            break
        dx = np.linalg.solve(J, -r)
        x = np.clip(x + dx, lo, hi)
    inside = bool(np.all(x > lo) and np.all(x < hi))
    ok = inside and abs(point["force_over_weight"] - 1.0) <= TRIM_TOL_FORCE and abs(point["tau_nNm"][1]) <= TRIM_TOL_TAU \
        and abs(point["Fx_over_weight"]) <= TRIM3_TOL_FX
    F = np.array(point["F_uN"])
    theta0 = -math.atan2(F[0], F[2])
    W = make_rig(float(x[2])).weight_uN
    return {
        "kin": Kin(F0).sym(float(x[0]), float(x[1])), "amp": float(x[0]), "shift_deg": math.degrees(x[1]), "shift_rad": float(x[1]),
        "stroke_plane_deg": float(x[2]), "point": point, "history": history, "inside_box": inside, "verified": bool(ok),
        "theta0_deg": math.degrees(theta0), "theta0_rad": theta0, "hover_body_pitch_deg": BODY_PITCH_DEG - math.degrees(theta0),
        "jacobian_rows_force_tau_Fx__cols_amp_shiftrad_planedeg": J.tolist(),
        "local_derivatives_at_trim": {
            "d_force_over_weight_d_amp": float(J[0, 0]), "d_tau_y_d_shift_nNm_per_rad": float(J[1, 1]),
            "d_tau_y_d_shift_nNm_per_deg": float(J[1, 1] * math.pi / 180.0), "d_tau_y_d_amp_nNm": float(J[1, 0]),
            "d_Fx_over_weight_d_plane_deg": float(J[2, 2]), "d_tau_y_d_plane_deg_nNm": float(J[1, 2]),
            "dF_d_amp_uN": float(J[0, 0] * W),
        },
        "verdict": "TRIM" if ok else "NO_TRIM",
    }


# ---- release readers ----------------------------------------------------------------------------
def hover_read(R: np.ndarray, window_s: float, setpoint_pitch_deg: float = BODY_PITCH_DEG) -> dict:
    """Altitude and attitude excursions over [0, window_s) of free flight. The verdict is
    against the pre-set 47.5 +- 10 deg band; the excursion about the controller's own
    set-point is reported alongside, not used for the verdict."""
    w = R[:, C_T] < window_s
    com = R[w, C_COM]
    pitch = R[w, C_PITCH]
    dz = com[:, 2] - com[0, 2]
    dpitch = pitch - BODY_PITCH_DEG
    t = R[w, C_T]
    alt_ok = bool(np.abs(dz).max() <= ALT_TOL_MM)
    pitch_ok = bool(np.abs(dpitch).max() <= PITCH_TOL_DEG)
    verdict = "HOVERS" if (alt_ok and pitch_ok) else ("PARTIAL" if (alt_ok or pitch_ok) else "FALLS")

    def first_cross(x, thr):
        idx = np.nonzero(np.abs(x) > thr)[0]
        return float(t[idx[0]]) if len(idx) else None

    return {
        "window_s": window_s, "n_beats": int(round(window_s * F0)),
        "max_abs_altitude_error_mm": float(np.abs(dz).max()), "final_altitude_error_mm": float(dz[-1]),
        "max_abs_pitch_error_deg": float(np.abs(dpitch).max()), "final_pitch_error_deg": float(dpitch[-1]),
        "setpoint_pitch_deg": setpoint_pitch_deg,
        "max_abs_pitch_excursion_about_setpoint_deg": float(np.abs(pitch - setpoint_pitch_deg).max()),
        "pitch_sd_about_setpoint_last_half_deg": float(pitch[len(pitch) // 2:].std()),
        "mean_body_pitch_deg": float(pitch.mean()), "body_pitch_sd_deg": float(pitch.std()),
        "max_abs_roll_deg": float(np.abs(R[w, C_ROLL]).max()), "max_abs_yaw_deg": float(np.abs(R[w, C_YAW]).max()),
        "horizontal_drift_mm": {"x_final": float(com[-1, 0] - com[0, 0]), "y_final": float(com[-1, 1] - com[0, 1]),
                                "max_abs_x": float(np.abs(com[:, 0] - com[0, 0]).max())},
        "speed_max_mm_s": float(np.linalg.norm(R[w, C_V], axis=1).max()),
        "pitch_error_first_cross_s": {"5deg": first_cross(dpitch, 5.0), "10deg": first_cross(dpitch, 10.0),
                                      "45deg": first_cross(dpitch, 45.0), "90deg": first_cross(dpitch, 90.0)},
        "altitude_first_cross_2mm_s": first_cross(dz, ALT_TOL_MM),
        "commands": {"amp_min": float(R[w, C_AL].min()), "amp_max": float(R[w, C_AL].max()),
                     "shift_min_deg": float(np.degrees(R[w, C_SL].min())), "shift_max_deg": float(np.degrees(R[w, C_SL].max())),
                     "theta_ref_min_deg": float(np.degrees(R[w, C_TREF].min())), "theta_ref_max_deg": float(np.degrees(R[w, C_TREF].max()))},
        "aero_power_uW_mean": float(R[w, C_PA].mean() * UW),
        "mean_force_uN": R[w, C_F].mean(axis=0).tolist(),
        "mean_torque_nNm": R[w, C_TAU].mean(axis=0).tolist(),
        "altitude_ok": alt_ok, "pitch_ok": pitch_ok, "verdict": verdict,
    }


def recovery_read(R: np.ndarray, kick: Kick) -> dict:
    t = R[:, C_T]
    pre = (t >= kick.at - 0.05) & (t < kick.at)
    post = t >= kick.at
    pitch_pre = float(R[pre, C_PITCH].mean())
    dp = R[post, C_PITCH] - pitch_pre
    tp = t[post] - kick.at
    peak_i = int(np.argmax(np.abs(dp)))
    outside = np.nonzero(np.abs(dp) > RECOVERED_DEG)[0]
    settle = float(tp[outside[-1]]) if len(outside) else 0.0
    thd = R[post, C_THD]
    return {
        "kick_nNm": kick.torque, "kick_at_s": kick.at, "kick_duration_s": kick.dur,
        "kick_impulse_nNm_s": kick.torque * kick.dur,
        "pre_kick_body_pitch_deg": pitch_pre,
        "peak_pitch_excursion_deg": float(dp[peak_i]), "time_to_peak_s": float(tp[peak_i]),
        "peak_pitch_rate_deg_s": float(np.degrees(np.abs(thd).max())),
        "settle_within_2deg_s": settle, "settle_within_2deg_beats": settle * F0,
        "pitch_error_at_end_deg": float(dp[-1]),
        "altitude_change_over_recovery_mm": float(R[post, C_COM][-1, 2] - R[post, C_COM][0, 2]),
        "recovered": bool(abs(dp[-1]) <= RECOVERED_DEG and settle < tp[-1] - 0.05),
    }


def open_loop_read(R: np.ndarray, window_ms: float = 50.0) -> dict:
    """Open-loop divergence: the envelope of the pitch excursion (max |dpitch| per 50 ms
    window, about 11 beats) and an exponential fit to that envelope while it is below
    45 deg. Ristroph et al. 2013 quote the fruit fly's instability growth time T_INST
    as about six times the 13 ms reaction time."""
    h = hover_read(R, float(R[-1, C_T]) + 1e-9)
    t = R[:, C_T]
    dpitch = R[:, C_PITCH] - BODY_PITCH_DEG
    dz = R[:, C_COM][:, 2] - R[0, C_COM][2]
    vx = R[:, C_V][:, 0]
    n_win = int(t[-1] // (window_ms * 1e-3))
    env = []
    for k in range(n_win):
        sel = (t >= k * window_ms * 1e-3) & (t < (k + 1) * window_ms * 1e-3)
        env.append({"t_mid_ms": (k + 0.5) * window_ms, "max_abs_pitch_deg": float(np.abs(dpitch[sel]).max()),
                    "max_abs_dz_mm": float(np.abs(dz[sel]).max()), "max_abs_vx_mm_s": float(np.abs(vx[sel]).max()),
                    "mean_pitch_deg": float(dpitch[sel].mean())})
    growth = None
    pts = [(e["t_mid_ms"] * 1e-3, e["max_abs_pitch_deg"]) for e in env if 0.5 < e["max_abs_pitch_deg"] < 45.0]
    if len(pts) >= 3:
        fit = slope([p[0] for p in pts], [math.log(p[1]) for p in pts])
        growth = {"rate_per_s": fit["slope"], "e_folding_ms": 1e3 / fit["slope"] if fit["slope"] > 0 else None,
                  "doubling_time_ms": math.log(2) / fit["slope"] * 1e3 if fit["slope"] > 0 else None,
                  "doubling_time_beats": math.log(2) / fit["slope"] * F0 if fit["slope"] > 0 else None,
                  "fit_residual_rms_log": fit["residual_rms"], "n_windows": len(pts)}
    # sign changes of the pitch excursion: an oscillatory mode shows several
    s = np.sign(dpitch[np.abs(dpitch) > 0.5])
    crossings = int(np.sum(s[1:] != s[:-1])) if len(s) > 1 else 0
    return {"duration_s": float(t[-1]), "hover_read": h, "envelope_50ms": env, "envelope_growth_fit": growth,
            "pitch_sign_changes": crossings, "max_abs_pitch_deg": float(np.abs(dpitch).max()),
            "final_hover_axis_z": float(R[-1, C_HZ]), "final_com_drop_mm": float(-dz[-1]),
            "final_vx_mm_s": float(vx[-1])}


# ---- footage ----------------------------------------------------------------------------------------
class HoverRecorder:
    """Two tethered beats slow, then the released, controlled body at 1/10 speed."""

    def __init__(self, rig: FL2.Rig, path: Path, tether_cycles: int, label: str):
        from PIL import Image, ImageDraw
        self.Image, self.ImageDraw = Image, ImageDraw
        self.rig, self.path, self.label = rig, path, label
        self.frames = []
        self.renderer = mj.Renderer(rig.model, height=720, width=960)
        self.cam = mj.MjvCamera()
        self.cam.type = mj.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = rig.data.xpos[rig.thorax]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 9.0, 150.0, -12.0
        self.opt = mj.MjvOption()
        self.opt.geomgroup[3] = 1
        self.tether_start = (tether_cycles - 2) / F0
        self.every_tethered, self.every_free = 2, 60
        self.step = 0
        self.z0 = None
        self.stamp = time.strftime("%Y-%m-%d %H:%M %Z")

    def tempo(self, every: int) -> str:
        return f"1/{1 / (every * self.rig.dt * 30):.0f} speed ({every * self.rig.dt * 1e3:.2f} ms per frame)"

    def __call__(self, rig: FL2.Rig, released: bool, cmd: Kin, controller) -> None:
        d = rig.data
        self.step += 1
        if not released:
            if d.time < self.tether_start or self.step % self.every_tethered:
                return
            tempo, phase = self.tempo(self.every_tethered), "TETHERED, trimmed pattern"
        else:
            if self.step % self.every_free:
                return
            tempo, phase = self.tempo(self.every_free), "RELEASED, external beat-synchronous PD"
            self.cam.lookat[:] = 0.7 * self.cam.lookat + 0.3 * d.xpos[rig.thorax]
        st = body_state(rig)
        if self.z0 is None:
            self.z0 = st.com[2]
            self.cam.lookat[:] = d.xpos[rig.thorax]      # the rig is cached: aim at this run's body
        self.renderer.update_scene(d, self.cam, self.opt)
        img = self.Image.fromarray(self.renderer.render())
        draw = self.ImageDraw.Draw(img)
        kick = bool(np.any(d.xfrc_applied[rig.thorax]))
        lines = [
            f"FL2b flybody hover control, {self.label}, recorded {self.stamp}",
            f"{phase}   t = {d.time * 1e3:7.1f} ms   {tempo}" + ("   TORQUE PULSE" if kick else ""),
            f"body pitch {st.body_pitch_deg:5.1f} deg (hover 47.5)   CoM altitude {st.com[2] - self.z0:+.2f} mm   "
            f"pitch set-point {BODY_PITCH_DEG - math.degrees(controller.theta_ref) if controller else BODY_PITCH_DEG:.1f} deg",
            f"commands: stroke amplitude x{cmd.amp[0]:.3f}, fore-aft shift {math.degrees(cmd.shift[0]):+.1f} deg",
            "EXTERNAL CONTROLLER, body calibration only: prescribed wing motion, no nervous system",
        ]
        y = 8
        for line in lines:
            draw.text((10, y), line, fill=(20, 20, 20))
            y += 18
        self.frames.append(np.asarray(img))

    def close(self) -> dict:
        import imageio.v2 as imageio
        self.renderer.close()
        imageio.mimwrite(self.path, self.frames, fps=30, codec="libx264", quality=8, macro_block_size=None)
        return {"path": str(self.path.relative_to(ROOT)), "frames": len(self.frames), "sha256": FL2.sha256(self.path)}


# ---- main ---------------------------------------------------------------------------------------------
def scan_1d(rig, pat, kins, log, name) -> list:
    points = []
    for kin in kins:
        t0 = time.time()
        p = tethered_point(rig, pat, kin)
        p["wall_s"] = round(time.time() - t0, 2)
        points.append(p)
        log(f"[{name}] f {kin.freq_hz:.0f} a {kin.amp[0]:.3f}/{kin.amp[1]:.3f} shift {math.degrees(kin.shift[0]):+5.1f} deg: "
            f"|F|/W {p['force_over_weight']:.4f} Fz/W {p['lift_over_weight']:.4f} Fx {p['F_uN'][0]:+.3f} uN "
            f"tau x/y/z {p['tau_nNm'][0]:+.3f}/{p['tau_nNm'][1]:+.3f}/{p['tau_nNm'][2]:+.3f} nN m "
            f"line {p['force_line_offset_mm']:+.3f} mm P {p['aero_power_uW']:.1f} uW "
            f"ach {math.degrees(p['l_stroke_achieved_amp_rad']):.0f} deg sat {p['l_stroke_saturated_fraction']:.2f} "
            f"lim {p['l_stroke_at_joint_limit_fraction']:.2f} ({p['wall_s']} s)")
    return points


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--render", action="store_true", help="MP4 of the closed-loop arm")
    ap.add_argument("--no-delay-scan", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="coarse grid, 0.2 s hovers, no receipt")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    started = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    t_start = time.time()
    log_lines = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    pattern = FL2.load_pattern()
    pat = Pattern(pattern)
    rig0 = make_rig(0.0)
    W, mass = rig0.weight_uN, rig0.mass_g
    I = inertia_about_com(rig0)
    I_yy = float(I[1, 1])
    st0 = body_state(rig0)
    FL2.require(abs(st0.body_pitch_deg - BODY_PITCH_DEG) < 0.2, f"spawn body pitch {st0.body_pitch_deg}")
    log(f"[body] mass {mass * 1e3:.4f} mg, weight {W:.3f} uN, I about CoM (g mm2) diag "
        f"{I[0, 0]:.3e} {I[1, 1]:.3e} {I[2, 2]:.3e}, spawn body pitch {st0.body_pitch_deg:.2f} deg, "
        f"stroke cycle mean {math.degrees(pat.stroke_mean):+.2f} deg")

    amp_scales = (0.9, 1.0, 1.1, 1.2) if args.smoke else AMP_SCALES
    shifts = (-20.0, -10.0, 0.0, 10.0, 20.0) if args.smoke else SHIFTS_DEG
    freqs = (200.0, 218.0, 240.0) if args.smoke else FREQS_HZ
    asym = (-0.1, 0.0, 0.1) if args.smoke else ASYMMETRIES

    # Part 1: derivatives in flybody's frame (stroke plane 0) ----------------------------------------
    grid = scan_1d(rig0, pat, [Kin(F0).sym(a, math.radians(s)) for a in amp_scales for s in shifts], log, "grid")
    freq_pts = scan_1d(rig0, pat, [Kin(f) for f in freqs], log, "freq")
    asym_pts = scan_1d(rig0, pat, [Kin(F0, (1 + d / 2, 1 - d / 2), (0.0, 0.0)) for d in asym], log, "asym")
    amp_line = [p for p in grid if abs(p["kin"]["shift_l_deg"]) < 1e-9]
    shift_line = [p for p in grid if abs(p["kin"]["amp_l"] - 1.0) < 1e-9]
    derivs = {
        "force_over_weight_vs_amp": slope([p["kin"]["amp_l"] for p in amp_line], [p["force_over_weight"] for p in amp_line]),
        "lift_over_weight_vs_amp": slope([p["kin"]["amp_l"] for p in amp_line], [p["lift_over_weight"] for p in amp_line]),
        "achieved_amp_deg_vs_amp": slope([p["kin"]["amp_l"] for p in amp_line], [math.degrees(p["l_stroke_achieved_amp_rad"]) for p in amp_line]),
        "aero_power_uW_vs_amp": slope([p["kin"]["amp_l"] for p in amp_line], [p["aero_power_uW"] for p in amp_line]),
        "tau_y_nNm_vs_shift_deg": slope([p["kin"]["shift_l_deg"] for p in shift_line], [p["tau_nNm"][1] for p in shift_line]),
        "force_line_mm_vs_shift_deg": slope([p["kin"]["shift_l_deg"] for p in shift_line], [p["force_line_offset_mm"] for p in shift_line]),
        "Fx_uN_vs_shift_deg": slope([p["kin"]["shift_l_deg"] for p in shift_line], [p["F_uN"][0] for p in shift_line]),
        "force_over_weight_vs_shift_deg": slope([p["kin"]["shift_l_deg"] for p in shift_line], [p["force_over_weight"] for p in shift_line]),
        "force_over_weight_vs_freq_hz": slope([p["kin"]["freq_hz"] for p in freq_pts], [p["force_over_weight"] for p in freq_pts]),
        "aero_power_uW_vs_freq_hz": slope([p["kin"]["freq_hz"] for p in freq_pts], [p["aero_power_uW"] for p in freq_pts]),
        "achieved_amp_deg_vs_freq_hz": slope([p["kin"]["freq_hz"] for p in freq_pts], [math.degrees(p["l_stroke_achieved_amp_rad"]) for p in freq_pts]),
        "tau_x_nNm_vs_asym": slope([p["kin"]["amp_l"] - p["kin"]["amp_r"] for p in asym_pts], [p["tau_nNm"][0] for p in asym_pts]),
        "tau_z_nNm_vs_asym": slope([p["kin"]["amp_l"] - p["kin"]["amp_r"] for p in asym_pts], [p["tau_nNm"][2] for p in asym_pts]),
        "Fy_uN_vs_asym": slope([p["kin"]["amp_l"] - p["kin"]["amp_r"] for p in asym_pts], [p["F_uN"][1] for p in asym_pts]),
    }
    for k, v in derivs.items():
        log(f"[deriv] {k}: slope {v['slope']:+.4f}, intercept {v['intercept']:+.4f}, resid rms {v['residual_rms']:.4f}")

    # Part 2: trims ------------------------------------------------------------------------------------
    trim2 = solve_trim(rig0, pat, grid, log)
    log(f"[trim2] {trim2['verdict']}: a {trim2['amp']:.4f}, shift {trim2['shift_deg']:+.2f} deg, |F|/W {trim2['point']['force_over_weight']:.4f}, "
        f"tau_y {trim2['point']['tau_nNm'][1]:+.3f} nN m, Fx/W {trim2['point']['Fx_over_weight']:+.4f}, theta0 {trim2['theta0_deg']:+.2f} deg "
        f"-> hover body pitch {trim2['hover_body_pitch_deg']:.1f} deg, aero {trim2['point']['aero_power_uW']:.1f} uW, "
        f"achieved stroke {math.degrees(trim2['point']['l_stroke_achieved_amp_rad']):.0f} deg")
    trim3 = solve_trim3(pat, trim2, log)
    log(f"[trim3] {trim3['verdict']}: a {trim3['amp']:.4f}, shift {trim3['shift_deg']:+.2f} deg, stroke plane {trim3['stroke_plane_deg']:+.2f} deg, "
        f"|F|/W {trim3['point']['force_over_weight']:.4f}, tau_y {trim3['point']['tau_nNm'][1]:+.3f}, Fx/W {trim3['point']['Fx_over_weight']:+.4f}, "
        f"theta0 {trim3['theta0_deg']:+.2f} deg, aero {trim3['point']['aero_power_uW']:.1f} uW, achieved stroke {math.degrees(trim3['point']['l_stroke_achieved_amp_rad']):.0f} deg")
    ld3 = trim3["local_derivatives_at_trim"]
    log(f"[trim3] local: d(|F|/W)/da {ld3['d_force_over_weight_d_amp']:+.3f}, dtau_y/dshift {ld3['d_tau_y_d_shift_nNm_per_deg']:+.4f} nN m/deg, "
        f"d(Fx/W)/dplane {ld3['d_Fx_over_weight_d_plane_deg']:+.4f} /deg, dtau_y/dplane {ld3['d_tau_y_d_plane_deg_nNm']:+.4f} nN m/deg")
    rig3 = make_rig(trim3["stroke_plane_deg"])
    dt_check = {}
    if not args.smoke:
        for dt in DT_CHECK:
            p = tethered_point(make_rig(trim3["stroke_plane_deg"], dt=dt), pat, trim3["kin"])
            dt_check[f"{dt:g}"] = {"force_over_weight": p["force_over_weight"], "tau_y_nNm": p["tau_nNm"][1],
                                   "Fx_over_weight": p["Fx_over_weight"], "aero_power_uW": p["aero_power_uW"]}
            log(f"[dt-check trim3] dt {dt:g}: |F|/W {p['force_over_weight']:.4f}, tau_y {p['tau_nNm'][1]:+.3f}, Fx/W {p['Fx_over_weight']:+.4f}, aero {p['aero_power_uW']:.1f} uW")

    # Part 3: release --------------------------------------------------------------------------------------
    hover_s = 0.2 if args.smoke else HOVER_S
    z_ref, x_ref = float(st0.com[2]), float(st0.com[0])

    def make_controller(trim: dict, bw_hz: float, delay_s: float, position_loop: bool = True) -> BeatController:
        ld = trim["local_derivatives_at_trim"]
        return BeatController(trim["kin"], ld["d_tau_y_d_shift_nNm_per_rad"], ld["dF_d_amp_uN"], I_yy, mass,
                              trim["theta0_rad"], z_ref, x_ref, pitch_bw_hz=bw_hz, delay_s=delay_s, position_loop=position_loop)

    def controller_record(c: BeatController, trim: dict, delay_ms: float, bw: float) -> dict:
        return {"pitch_bw_hz": bw, "pitch_zeta": PITCH_ZETA, "kp_nNm_per_rad": c.kp, "kd_nNm_s_per_rad": c.kd,
                "alt_bw_hz": ALT_BW_HZ, "pos_bw_hz": POS_BW_HZ, "delay_ms": delay_ms, "updates": c.n_updates,
                "clamped_shift_updates": c.clamped_shift, "clamped_amp_updates": c.clamped_amp,
                "dtau_dshift_nNm_per_rad": c.dtau_dshift, "dF_damp_uN": c.dF_damp, "I_yy_g_mm2": I_yy,
                "theta0_deg": trim["theta0_deg"], "setpoint_body_pitch_deg": trim["hover_body_pitch_deg"],
                "z_ref_mm": z_ref, "x_ref_mm": x_ref, "stroke_plane_deg": trim.get("stroke_plane_deg", 0.0)}

    arms = {}
    # open loop at the three-dial trim: the body's own pitch instability
    t0 = time.time()
    out = run(rig3, pat, trim3["kin"], PRE_RELEASE_CYCLES, 0.2 if args.smoke else OPEN_LOOP_S)
    arms["open_loop"] = {"trim": "three-dial", "read": open_loop_read(out["free"]), "wall_s": round(time.time() - t0, 2)}
    ol = arms["open_loop"]["read"]
    log(f"[open_loop] three-dial trim, no feedback: pitch crosses 5/10/45/90 deg at {ol['hover_read']['pitch_error_first_cross_s']} s, "
        f"max |pitch| {ol['max_abs_pitch_deg']:.1f} deg, sign changes {ol['pitch_sign_changes']}, envelope fit {ol['envelope_growth_fit']}, "
        f"final hover axis z {ol['final_hover_axis_z']:+.2f}, drop {ol['final_com_drop_mm']:.1f} mm, vx {ol['final_vx_mm_s']:+.0f} mm/s -> {ol['hover_read']['verdict']}")
    log("[open_loop] envelope max|pitch| per 50 ms: " + " ".join(f"{e['max_abs_pitch_deg']:.1f}" for e in ol["envelope_50ms"]) + " deg")

    # closed loop in flybody's frame (two-dial trim, stroke plane 0): the body nulls thrust by pitching up
    t0 = time.time()
    c2 = make_controller(trim2, PITCH_BW_HZ, 0.0)
    out = run(rig0, pat, trim2["kin"], PRE_RELEASE_CYCLES, hover_s, controller=c2)
    h2 = hover_read(out["free"], hover_s, trim2["hover_body_pitch_deg"])
    arms["closed_loop_flybody_frame"] = {"trim": "two-dial", "hover": h2, "controller": controller_record(c2, trim2, 0.0, PITCH_BW_HZ),
                                         "wall_s": round(time.time() - t0, 2)}
    log(f"[closed_loop_flybody_frame] {h2['verdict']}: altitude |err| max {h2['max_abs_altitude_error_mm']:.2f} mm, pitch |err| max {h2['max_abs_pitch_error_deg']:.2f} deg "
        f"about 47.5; mean {h2['mean_body_pitch_deg']:.1f} deg, sd about set-point (last half) {h2['pitch_sd_about_setpoint_last_half_deg']:.2f}, "
        f"drift x {h2['horizontal_drift_mm']['x_final']:+.1f} mm")

    # closed loop at the three-dial trim: the pre-set criterion test, rendered, then the kick
    t0 = time.time()
    c3 = make_controller(trim3, PITCH_BW_HZ, 0.0)
    kick = Kick(KICK_NNM, hover_s, KICK_S)
    recorder = HoverRecorder(rig3, HERE / "fl2b_hover_closed_loop.mp4", PRE_RELEASE_CYCLES, "closed loop, three-dial trim") if (args.render and not args.smoke) else None
    out = run(rig3, pat, trim3["kin"], PRE_RELEASE_CYCLES, hover_s + (0.1 if args.smoke else RECOVER_S), controller=c3, kick=kick, frame_hook=recorder)
    cl = {"trim": "three-dial", "hover": hover_read(out["free"], hover_s, trim3["hover_body_pitch_deg"]), "recovery": recovery_read(out["free"], kick),
          "controller": controller_record(c3, trim3, 0.0, PITCH_BW_HZ), "wall_s": round(time.time() - t0, 2), "steps": out["steps"]}
    if recorder is not None:
        cl["clip"] = recorder.close()
    arms["closed_loop"] = cl
    h, rc = cl["hover"], cl["recovery"]
    log(f"[closed_loop] {h['verdict']}: altitude |err| max {h['max_abs_altitude_error_mm']:.2f} mm, pitch |err| max {h['max_abs_pitch_error_deg']:.2f} deg "
        f"(mean {h['mean_body_pitch_deg']:.1f} sd {h['body_pitch_sd_deg']:.2f}, last-half sd {h['pitch_sd_about_setpoint_last_half_deg']:.2f}), roll max {h['max_abs_roll_deg']:.2f}, "
        f"drift x {h['horizontal_drift_mm']['x_final']:+.1f} mm, amp {h['commands']['amp_min']:.3f}-{h['commands']['amp_max']:.3f}, "
        f"shift {h['commands']['shift_min_deg']:+.1f}..{h['commands']['shift_max_deg']:+.1f} deg, aero {h['aero_power_uW_mean']:.1f} uW ({cl['wall_s']} s wall)")
    log(f"[closed_loop] kick {rc['kick_nNm']} nN m x {rc['kick_duration_s'] * 1e3:.0f} ms nose-down: peak {rc['peak_pitch_excursion_deg']:+.1f} deg at {rc['time_to_peak_s'] * 1e3:.0f} ms, "
        f"settled within 2 deg after {rc['settle_within_2deg_s'] * 1e3:.0f} ms ({rc['settle_within_2deg_beats']:.1f} beats), recovered {rc['recovered']}")

    # ablation: pitch loop only, no position loop
    t0 = time.time()
    c3n = make_controller(trim3, PITCH_BW_HZ, 0.0, position_loop=False)
    out = run(rig3, pat, trim3["kin"], PRE_RELEASE_CYCLES, hover_s, controller=c3n)
    hn = hover_read(out["free"], hover_s, trim3["hover_body_pitch_deg"])
    arms["closed_loop_no_position_loop"] = {"trim": "three-dial", "hover": hn, "wall_s": round(time.time() - t0, 2)}
    log(f"[no_position_loop] {hn['verdict']}: altitude |err| max {hn['max_abs_altitude_error_mm']:.2f} mm, pitch |err| max {hn['max_abs_pitch_error_deg']:.2f} deg, drift x {hn['horizontal_drift_mm']['x_final']:+.1f} mm")

    delay_scan, d_max = {}, None
    if not args.no_delay_scan and not args.smoke:
        for D in DELAYS_MS:
            rows = {}
            for bw in BANDWIDTHS_HZ:
                t0 = time.time()
                c = make_controller(trim3, bw, D * 1e-3)
                o = run(rig3, pat, trim3["kin"], PRE_RELEASE_CYCLES, hover_s, controller=c)
                hr = hover_read(o["free"], hover_s, trim3["hover_body_pitch_deg"])
                rows[f"{bw:g}"] = {"verdict": hr["verdict"], "max_abs_pitch_error_deg": hr["max_abs_pitch_error_deg"],
                                   "max_abs_altitude_error_mm": hr["max_abs_altitude_error_mm"],
                                   "pitch_sd_last_half_deg": hr["pitch_sd_about_setpoint_last_half_deg"], "wall_s": round(time.time() - t0, 2)}
            hovering = {bw: r for bw, r in rows.items() if r["verdict"] == "HOVERS"}
            best = min(hovering, key=lambda k: hovering[k]["max_abs_pitch_error_deg"]) if hovering else None
            delay_scan[f"{D:g}"] = {"by_bandwidth_hz": rows, "hovers": bool(hovering), "best_bandwidth_hz": float(best) if best else None,
                                    "best_max_abs_pitch_error_deg": hovering[best]["max_abs_pitch_error_deg"] if best else None}
            log(f"[delay {D:4.0f} ms] " + "  ".join(f"{bw} Hz: {r['verdict'][:6]} {r['max_abs_pitch_error_deg']:.1f} deg/{r['max_abs_altitude_error_mm']:.1f} mm" for bw, r in rows.items()))
        ok = [float(D) for D, r in delay_scan.items() if r["hovers"]]
        d_max = max(ok) if ok else None

    hover_dt = {}
    if not args.smoke:
        for dt in (2.5e-5,):
            r2 = make_rig(trim3["stroke_plane_deg"], dt=dt)
            c = make_controller(trim3, PITCH_BW_HZ, 0.0)
            o = run(r2, pat, trim3["kin"], PRE_RELEASE_CYCLES, hover_s, controller=c)
            hr = hover_read(o["free"], hover_s, trim3["hover_body_pitch_deg"])
            hover_dt[f"{dt:g}"] = {"verdict": hr["verdict"], "max_abs_pitch_error_deg": hr["max_abs_pitch_error_deg"],
                                   "max_abs_altitude_error_mm": hr["max_abs_altitude_error_mm"], "mean_body_pitch_deg": hr["mean_body_pitch_deg"]}
            log(f"[hover dt {dt:g}] {hr['verdict']}: pitch |err| max {hr['max_abs_pitch_error_deg']:.2f} deg, altitude |err| max {hr['max_abs_altitude_error_mm']:.2f} mm")

    wall = round(time.time() - t_start, 1)
    summary = {"trim2": trim2["verdict"], "trim3": trim3["verdict"], "hover_flybody_frame": h2["verdict"], "hover": h["verdict"],
               "open_loop": ol["hover_read"]["verdict"], "d_max_ms": d_max, "wall_s": wall}
    log(f"[summary] {json.dumps(summary)}")
    if args.smoke:
        return 0

    def trim_out(trim: dict) -> dict:
        o = {k: v for k, v in trim.items() if k != "kin"}
        o["kin"] = kin_dict(trim["kin"])
        return o

    receipt = {
        "experiment": "FL2b flybody control derivatives, trim and scripted hover (body calibration, external controller)",
        "started": started, "finished": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "wall_s": wall,
        "identity": {**FL2.identity(), "fl2b_script_sha256": FL2.sha256(Path(__file__)),
                     "fl2_script_sha256": FL2.sha256(Path(FL2.__file__))},
        "sources_added": {
            "stroke_plane_angle": "flybody fruitfly.py Fly.__init__(stroke_plane_angle=0.): 'Angle of wing stroke plane for initial flight pose, relative to ground, degrees. 0: horizontal stroke plane.' (TuragaLab/flybody d015e9b)",
            "hover_body_pitch_animal": "Fry, Sayaman & Dickinson 2005 J Exp Biol 208:2303, hovering example 'a body pitch of 45 deg'; flybody 47.5 from Muijres et al. 2014 (doi:10.1126/science.1248955)",
            "pitch_perturbation_animal": "Ristroph et al. 2013 J R Soc Interface 10:20130237: one-beat magnetic torque, 5-25 deg pitch, recovery about 60 ms / 15 beats, reaction time 13 +- 2 ms (n=12), corrective forward sweep of the stroke; T_INST/T_RXN about 6",
            "haltere_reflex_latency": "Fayyazuddin & Dickinson 1996 J Neurosci 16:5225 (haltere nerve -> mnb1 EPSP 0.87 ms, Calliphora); Mielke & Heide 1993 haltere nerve -> b1 spike 3-4 ms; Dickerson 2020 Proc R Soc B review: about 4 ms",
        },
        "parameters": {**FL2.parameters(), "amp_scales": list(AMP_SCALES), "shifts_deg": list(SHIFTS_DEG), "freqs_hz": list(FREQS_HZ),
                       "asymmetries": list(ASYMMETRIES), "scan_cycles": SCAN_CYCLES, "scan_read_cycles": SCAN_READ,
                       "trim_tolerance": {"force_over_weight": TRIM_TOL_FORCE, "tau_y_nNm": TRIM_TOL_TAU, "Fx_over_weight_trim3": TRIM3_TOL_FX},
                       "trim3_fd_steps": {"amp": TRIM3_STEPS[0], "shift_rad": TRIM3_STEPS[1], "stroke_plane_deg": TRIM3_STEPS[2]},
                       "pre_release_cycles": PRE_RELEASE_CYCLES, "open_loop_s": OPEN_LOOP_S, "hover_s": HOVER_S,
                       "kick": {"nNm": KICK_NNM, "at_s": KICK_AT_S, "duration_s": KICK_S, "recover_s": RECOVER_S},
                       "hover_tolerance": {"altitude_mm": ALT_TOL_MM, "pitch_deg": PITCH_TOL_DEG, "about_body_pitch_deg": BODY_PITCH_DEG},
                       "controller": {"pitch_bw_hz": PITCH_BW_HZ, "pitch_zeta": PITCH_ZETA, "alt_bw_hz": ALT_BW_HZ, "alt_zeta": ALT_ZETA,
                                      "pos_bw_hz": POS_BW_HZ, "pos_zeta": POS_ZETA, "shift_clamp_deg": 20.0, "amp_clamp": [AMP_MIN, AMP_MAX],
                                      "theta_ref_clamp_deg": 15.0, "update": "once per wingbeat from the previous beat's mean state"},
                       "delays_ms": list(DELAYS_MS), "bandwidths_hz": list(BANDWIDTHS_HZ), "dt_check_s": list(DT_CHECK),
                       "pre_set_reading": {"TRIM": "verified direct run inside the box: |F|/W within 0.5%, |tau_y| < 0.05 nN m (trim3 also |Fx|/W < 0.5%)",
                                           "HOVERS": "closed loop, D=0: altitude within 2 mm AND body pitch within 10 deg of 47.5 for 1.0 s",
                                           "D_MAX": "largest scanned delay that HOVERS at its best bandwidth"}},
        "body": {"mass_mg": mass * 1e3, "weight_uN": W, "inertia_about_com_g_mm2_world_at_spawn": I.tolist(), "I_yy_g_mm2": I_yy,
                 "spawn_body_pitch_deg": st0.body_pitch_deg, "stroke_cycle_mean_deg": math.degrees(pat.stroke_mean),
                 "frame": "world +x = nose at spawn, +y = left, +z = hover-up; pitch torque +y = nose-down; body pitch = thorax x axis above horizontal (flybody definition); stroke shift negative = wings swept forward"},
        "scans": {"grid_amp_shift": grid, "frequency": freq_pts, "asymmetry": asym_pts},
        "derivatives_at_baseline": derivs,
        "trim2": trim_out(trim2), "trim3": trim_out(trim3), "trim3_dt_check": dt_check,
        "arms": arms, "delay_scan": delay_scan, "d_max_ms": d_max, "hover_dt_check": hover_dt,
        "summary": summary, "log": log_lines,
    }
    out_path = Path(args.out) if args.out else HERE / f"{time.strftime('%Y-%m-%d')}-fl2b-control-derivatives.json"
    out_path.write_text(json.dumps(receipt, indent=1, default=float))
    print(f"receipt {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
