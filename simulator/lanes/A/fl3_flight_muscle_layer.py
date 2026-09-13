#!/usr/bin/env python3
"""FL3a: a source-bound flight-muscle layer on the FL2b body, closed by the animal's own
haltere reflex gains, then driven by the champion's recorded flight motor rates.

FL2b (lanes/A/2026-09-09-fl2b-control-derivatives.json) measured what the body needs: at the
three-dial trim the released body diverges in pitch unless a beat-synchronous loop shifts the
stroke fore-aft within about 25 ms of body motion. That loop was a DESIGN (kp = I w^2 at
12 Hz). This runner replaces the design with the ANIMAL'S MEASURED CONTROLLER and puts a
muscle layer between motor-neuron spikes and the wing dials, so that the next step (a live
CNS in the loop) has a physical, tested interface. It is still a body-calibration control:
the reflex is a model of the thoracic haltere -> steering-motor-neuron arc, not the
connectome, and the packet says so. No nervous system, no seed.

Layer (lanes/A only, default off in the shared model, nothing outside lanes/A changed):
  power    the 24 DLM/DVM motor neurons (12 per side). Their pooled rate r drives a Ca-like
           state c with Gordon & Dickinson 2006 kinetics (power rises with tau 0.442 s and
           falls with tau 1.79 s after a rate step; PNAS 103:4311) and steady-state power
           P = P_hover (c / 5 Hz)^0.63 (their "2-fold power for a 3-fold rate"; A-IFM
           motor neurons fire about 5 Hz in Drosophila flight). The body's own aero
           power-amplitude line from FL2b (P = -53.87 + 95.50 a uW) turns P into the stroke
           amplitude scale a, clamped to the scanned box 0.90..1.20 with a saturation flag.
           Per side, so a left-right power imbalance rolls the body.
  steering the 40 other wing motor neurons, one first-order activation each (tau 5 ms, about
           one wingbeat; ASSUMED), unit activation = one spike per wingbeat (218 Hz). Roles
           and phasic/tonic classes from Lindsay, Sustar & Dickinson 2017 (Curr Biol 27:345):
           b1, b2 extend the FRONT (ventral) stroke extreme; b3 the BACK (dorsal) extreme;
           i1, i2 reduce and iii1, iii3, iii4 raise stroke amplitude; iv1-4 (hg1-4) set the
           angle of attack (yaw; carried, gain 0, HYPOTHESISED); tp1, tp2, tpn, PS1, PS2,
           PSn_u set thoracic stiffness/frequency (carried, gain 0); MNxm01, PSI unassigned.
           Gain MAGNITUDES (deg of kinematic change at unit activation) are HYPOTHESISED:
           20 deg phasic, 10 deg tonic, 5 deg for the flat class.
  kinematics per wing: front extension f and back extension b (rad) act on the pattern's
           stroke extremes exactly: amplitude scale a = a_power + (f + b) / Phi0 and mean
           shift = trim + b - (f + b) dev_max / Phi0 (Phi0 = 128.8 deg, dev_max = 65.6 deg).
           A pure front extension therefore adds lift AND a nose-up shift, as in the animal.
  haltere  body-frame pitch and roll rates and angles, averaged over the beat just finished,
           one command per wingbeat after a pure delay:
           pitch  Whitehead, Beatus, Canale & Cohen 2015 J Exp Biol 218:3508: front stroke
                  change = K_i dtheta + K_p dtheta/dt, K_i 0.3 +- 0.15 deg/deg,
                  K_p 7 +- 2.1 ms, delay 6 +- 1.7 ms (latency 9.9 +- 2.1 ms, 90% correction
                  in 29 +- 8 ms for 5-40 deg). Bilateral front extension.
           roll   Beatus, Guckenheimer & Cohen 2015 J R Soc Interface 12:20150075: stroke
                  amplitude asymmetry = K_i rho + K_p drho/dt, K_i 0.6 +- 0.3, K_p 4.8 +- 2.4
                  ms, delay 4.6 +- 1 ms (response 5 ms, correction 6.8 +- 1.6 beats). The
                  wing on the low side flaps more.
           The animal's outputs are ACHIEVED kinematics; the servo delivers 85% of a
           commanded amplitude change, so commands are divided by the tethered gain
           measured in part 1.
Parts:
  1  tethered Jacobian of the new dials (front +10 deg, back +10 deg, asymmetry 10 deg)
     against FL2b's (a, shift, asymmetry) derivatives; servo gains achieved/commanded.
  2  released at the FL2b three-dial trim, power at the 5 Hz anchor, animal reflex on:
     1.0 s, then the FL2b pitch kick (6 nN m x 5 ms nose-down) at 1.0 s and a roll kick
     (8 nN m x 5 ms about the body long axis, left side down: about 4500 deg/s, mid-range of
     the Beatus perturbations) at 1.5 s.
     Arms: animal_reflex; + the FL2b altitude loop (labelled EXTERNAL); pitch loop only;
     no reflex (control); and a 3 x 3 x 3 grid of the pitch gains and delay at mean +- 1 sd.
     --render films the first two arms (the reflex model has no altitude term, so the first
     sinks about 6 mm/s and passes through the arena's ground plane, which has no contact
     geometry: contype/conaffinity 0, ncon 0 throughout; the second holds altitude under the
     labelled external loop).
  3  the layer driven by the champion's recorded FL1 wing motor rates (spent seed 203760,
     lanes/A/2026-09-09-fl1-flight-command-wing-motor.json; no CNS run here, no seed):
     arms wi1 (walking command), flight_dng02, flight_dnp31; reflex on. Per-side power and
     steering from the per-cell rates; the anchor's sensitivity (5, 10, 20 Hz) is tabulated.
Pre-set readings, written before the run:
  JACOBIAN  front extension gives nose-up torque (tau_y falls) and more lift; back extension
            nose-down; asymmetry rolls with the FL2b slope (0.167 nN m per 1%) within 25%.
  HOLDS     |body pitch - 47.5| <= 10 deg AND |roll| <= 10 deg for the 1.0 s hover.
  HOVERS    FL2b's reading (altitude within 2 mm AND pitch within 10 deg) on the arm with
            the altitude loop.
  RECOVERS  pitch: back within 2 deg and stays; the 90% correction time is compared with the
            animal's 29 +- 8 ms. roll: back within 10% of the peak, compared with 6.8 beats.
  POWER     for each FL1 arm, the layer's commanded a at the 5 Hz anchor; IN_BAND if
            0.98 <= a <= 1.08 (|F|/W within 10%), SATURATED if clamped at 1.20, else the
            value. The body run reports climb speed and attitude.
Units mm-g-s: force uN, torque nN m, power uW. Engine MuJoCo, reported by version.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import mujoco as mj

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lanes.A import fl2_flybody_lift as FL2  # noqa: E402
from lanes.A import fl2b_control_derivatives as FL2B  # noqa: E402

F0 = FL2.BASE_FREQ_HZ
CONTROL_DT = FL2.CONTROL_DT
BODY_PITCH_DEG = FL2.BODY_PITCH_DEG
G = FL2.GRAVITY
UW = FL2.UW_PER_UNIT

FL2B_RECEIPT = HERE / "2026-09-09-fl2b-control-derivatives.json"
FL2C_RECEIPT = HERE / "2026-09-09-fl2c-flight-leg-pose.json"      # flybody flight leg pose (Robin, 2026-09-09)
FL1_RECEIPT = HERE / "2026-09-09-fl1-flight-command-wing-motor.json"

SOURCES = {
    "power_kinetics_and_law": "Gordon & Dickinson 2006 PNAS 103:4311 (doi:10.1073/pnas.0510109103): A-IFM motor neurons fire about 5 Hz in Drosophila flight; a 3-fold change in steady-state spike rate accompanies a 2-fold change in mechanical power; after a rate step power rises with tau 0.442 s and falls with tau 1.79 s; Ca2+ rises in proportion to firing rate; power roughly proportional to (f Phi)^3",
    "power_ca_curve": "Wang, Zhao & Swank 2011 Biophys J 101:2207: skinned Drosophila IFM power 0 at pCa 5.8, maximal at pCa 5.25; a 2-fold [Ca] change gives 2-3 fold power and 1.2-fold kinetics",
    "power_rate_band": "Harcombe & Wyman 1977 J Comp Physiol 123:271: DLM motor units 5-20 Hz in flight, silent at rest",
    "steering_roles": "Lindsay, Sustar & Dickinson 2017 Curr Biol 27:345: b1/b2 extend the ventral (front) stroke extreme, b3 the dorsal (back) extreme (pitch); i1/i2 depress the stroke plane and reduce amplitude; iii1/iii3/iii4 antagonise them and raise amplitude; hg1-4 (iv1-4) regulate angle of attack for yaw; phasic b2, i1, iii1, hg1; tonic b1, b3, iii3, hg4; flat i2, iii4, hg2, hg3",
    "pitch_reflex": "Whitehead, Beatus, Canale & Cohen 2015 J Exp Biol 218:3508 (arXiv 1503.06507): bilateral front-stroke modulation, PI on pitch rate = K_i dtheta + K_p dtheta/dt with K_i 0.3 +- 0.15, K_p 7 +- 2.1 ms, delay 6 +- 1.7 ms (n = 9 fits); latency 9.9 +- 2.1 ms (n = 18); 90% correction 29 +- 8 ms (n = 32) for 5-40 deg",
    "roll_reflex": "Beatus, Guckenheimer & Cohen 2015 J R Soc Interface 12:20150075: stroke-amplitude asymmetry, PI on roll rate with K_i 0.6 +- 0.3, K_p 4.8 +- 2.4 ms, delay 4.6 +- 1 ms; response within one wingbeat (5 ms); correction in 6.8 +- 1.6 beats (30 +- 7 ms) for up to 100 deg; example 60 deg in 13.5 ms at 7000 deg/s",
    "haltere_latency": "Fayyazuddin & Dickinson 1996 J Neurosci 16:5225; Dickinson 1999 Phil Trans R Soc B 354:903 (Drosophila haltere-mediated equilibrium reflexes)",
    "steering_activation_tau": "ASSUMED 5 ms (about one wingbeat; synchronous twitch muscles, Tu & Dickinson 1994 J Exp Biol 192:207 b1 twitch of the order of the stroke period)",
    "steering_gain_magnitudes": "HYPOTHESISED: 20 deg (phasic), 10 deg (tonic), 5 deg (flat class) of kinematic change at one spike per wingbeat; signs and classes from Lindsay 2017",
    "body_jacobian": "lanes/A/2026-09-09-fl2b-control-derivatives.json (FL2b, this lane): aero power vs amplitude line, three-dial trim, stroke-plane 9.46 deg",
    "motor_rates": "lanes/A/2026-09-09-fl1-flight-command-wing-motor.json (FL1, this lane): per-cell wing motor neuron rates, wi1 / DNg02 / DNp31 arms, spent seed 203760, Metal",
}

# ---- power (asynchronous indirect flight muscle) ------------------------------------------------
POWER_ANCHOR_HZ = 5.0                       # A-IFM motor neuron rate in flight (Gordon & Dickinson 2006)
POWER_EXPONENT = math.log(2.0) / math.log(3.0)   # 2-fold power per 3-fold rate = 0.631
POWER_TAU_RISE_S, POWER_TAU_FALL_S = 0.442, 1.79
POWER_BAND_HZ = (5.0, 20.0)                 # Harcombe & Wyman 1977
POWER_STOP_HZ = 1.0                         # below this equivalent rate the muscle is not stretch-activatable: wings stop
AMP_MIN, AMP_MAX = FL2B.AMP_MIN, FL2B.AMP_MAX    # the FL2b scanned box
IN_BAND_AMP = (0.98, 1.08)                  # |F|/W within about 10% at 1.9-2.0 per unit

# ---- steering (synchronous) ----------------------------------------------------------------------
STEER_TAU_S = 0.005
STEER_UNIT_HZ = F0                          # one spike per wingbeat = unit activation
GAIN_PHASIC_DEG, GAIN_TONIC_DEG, GAIN_FLAT_DEG = 20.0, 10.0, 5.0
# muscle -> (dial, gain_deg_at_unit_activation, class); dial in {front, back, amp, aoa, freq, None}
STEERING_ROLES = {
    "b1": ("front", +GAIN_TONIC_DEG, "tonic"), "b2": ("front", +GAIN_PHASIC_DEG, "phasic"),
    "b3": ("back", +GAIN_TONIC_DEG, "tonic"),
    "i1": ("amp", -GAIN_PHASIC_DEG, "phasic"), "i2": ("amp", -GAIN_FLAT_DEG, "flat"),
    "iii1": ("amp", +GAIN_PHASIC_DEG, "phasic"), "iii3": ("amp", +GAIN_TONIC_DEG, "tonic"), "iii4": ("amp", +GAIN_FLAT_DEG, "flat"),
    "iv1": ("aoa", 0.0, "phasic"), "iv2": ("aoa", 0.0, "flat"), "iv3": ("aoa", 0.0, "flat"), "iv4": ("aoa", 0.0, "tonic"),
    "tp1": ("freq", 0.0, "unassigned"), "tp2": ("freq", 0.0, "unassigned"), "tpn": ("freq", 0.0, "unassigned"),
    "PS1": ("freq", 0.0, "unassigned"), "PS2": ("freq", 0.0, "unassigned"), "PSn_u": ("freq", 0.0, "unassigned"),
    "MNxm01": (None, 0.0, "unassigned"), "PSI": (None, 0.0, "interneuron"),
}
POWER_TYPES = {"DLM1-4": "DLM", "DLM5": "DLM", "DVM1a-c": "DVM", "DVM2a-b": "DVM", "DVM3a-b": "DVM"}

# ---- haltere reflex (animal gains) -------------------------------------------------------------------
PITCH_KI, PITCH_KI_SD = 0.3, 0.15           # deg front stroke per deg pitch
PITCH_KP_MS, PITCH_KP_MS_SD = 7.0, 2.1      # deg front stroke per deg/ms pitch rate
PITCH_DELAY_MS, PITCH_DELAY_MS_SD = 6.0, 1.7
ROLL_KI, ROLL_KI_SD = 0.6, 0.3              # deg amplitude asymmetry per deg roll
ROLL_KP_MS, ROLL_KP_MS_SD = 4.8, 2.4
ROLL_DELAY_MS, ROLL_DELAY_MS_SD = 4.6, 1.0
FRONT_CLAMP_DEG, ASYM_CLAMP_DEG = 40.0, 70.0

# ---- protocol --------------------------------------------------------------------------------------------
PRE_RELEASE_CYCLES = FL2B.PRE_RELEASE_CYCLES
HOVER_S = 1.0
PITCH_KICK = dict(nNm=6.0, at_s=1.0, dur_s=0.005)          # FL2b, Ristroph 2013 range
ROLL_KICK = dict(nNm=8.0, at_s=1.5, dur_s=0.005)           # about 4500 deg/s initial rate, mid-range of Beatus 2015
RECOVER_S = 0.5
JAC_STEP_DEG = 10.0
HOLD_TOL_DEG = 10.0
ANCHOR_SCAN_HZ = (5.0, 10.0, 20.0)
SENS_SCALES = (-1.0, 0.0, 1.0)              # mean + k sd


# ---- kinematics -------------------------------------------------------------------------------------------
class StrokeGeometry:
    """The pattern's stroke extremes at unit amplitude: front (min, forward) and back (max)."""

    def __init__(self, pat: FL2B.Pattern):
        s = pat.p[:, 0]
        self.mean = float(s.mean())
        self.dev_max = float(s.max() - self.mean)       # back (dorsal) extreme, positive
        self.dev_min = float(s.min() - self.mean)       # front (ventral) extreme, negative
        self.phi0 = self.dev_max - self.dev_min         # peak-to-peak at a = 1, rad

    def kin(self, a_power: tuple, front: tuple, back: tuple, trim_shift: float, freq_hz: float = F0) -> FL2B.Kin:
        """Per wing: extend the front extreme by `front` (forward) and the back extreme by
        `back` (backward), both in rad, about the power amplitude a_power; exact."""
        amps, shifts = [], []
        for ap, f, b in zip(a_power, front, back):
            amp = ap + (f + b) / self.phi0
            shift = trim_shift + b - (f + b) * self.dev_max / self.phi0
            amps.append(float(amp))
            shifts.append(float(shift))
        return FL2B.Kin(freq_hz, tuple(amps), tuple(shifts))

    def extremes(self, kin: FL2B.Kin, side: int) -> tuple:
        """(front, back) stroke angles in rad for the given wing of a Kin."""
        a, s = kin.amp[side], kin.shift[side]
        return self.mean + a * self.dev_min + s, self.mean + a * self.dev_max + s


# ---- power muscle -----------------------------------------------------------------------------------------
class PowerLine:
    """The body's aero power vs amplitude line (FL2b) and its hover anchor."""

    def __init__(self, receipt: dict):
        fit = receipt["derivatives_at_baseline"]["aero_power_uW_vs_amp"]
        self.intercept, self.slope = float(fit["intercept"]), float(fit["slope"])
        self.a_hover = float(receipt["trim3"]["amp"])
        self.p_hover = self.power(self.a_hover)

    def power(self, a: float) -> float:
        return self.intercept + self.slope * a

    def amplitude(self, p: float) -> float:
        return (p - self.intercept) / self.slope


class PowerMuscle:
    """Pooled DLM/DVM motor neuron rate -> Ca-like state -> mechanical power -> stroke amplitude.

    The Ca-like state c (an equivalent steady rate, Hz) follows the pooled rate r with
    Gordon & Dickinson's asymmetric first-order kinetics. Steady power P(c) = P_hover
    (c / anchor)^0.63; the body's line gives the amplitude, clamped to the scanned box.
    """

    def __init__(self, line: PowerLine, anchor_hz: float = POWER_ANCHOR_HZ, c0: float | None = None):
        self.line, self.anchor = line, anchor_hz
        self.c = anchor_hz if c0 is None else c0
        self.saturated_high = self.saturated_low = self.stopped = 0

    def step(self, rate_hz: float, dt: float) -> None:
        tau = POWER_TAU_RISE_S if rate_hz > self.c else POWER_TAU_FALL_S
        self.c += (rate_hz - self.c) * (1.0 - math.exp(-dt / tau))

    def steady_amplitude(self, c: float) -> tuple:
        """(amplitude, flag) for an equivalent rate c without clamping side effects."""
        if c < POWER_STOP_HZ:
            return 0.0, "STOPPED"
        p = self.line.p_hover * (c / self.anchor) ** POWER_EXPONENT
        a = self.line.amplitude(p)
        if a > AMP_MAX:
            return AMP_MAX, "SATURATED"
        if a < AMP_MIN:
            return AMP_MIN, "UNDERPOWERED"
        return a, "IN_BAND" if IN_BAND_AMP[0] <= a <= IN_BAND_AMP[1] else "OUT_OF_BAND"

    def amplitude(self) -> float:
        a, flag = self.steady_amplitude(self.c)
        if flag == "SATURATED":
            self.saturated_high += 1
        elif flag == "UNDERPOWERED":
            self.saturated_low += 1
        elif flag == "STOPPED":
            self.stopped += 1
        return a


# ---- steering muscles ------------------------------------------------------------------------------------
class SteeringMuscle:
    def __init__(self, name: str):
        self.name = name
        self.dial, self.gain_deg, self.cls = STEERING_ROLES[name]
        self.u = 0.0

    def step(self, rate_hz: float, dt: float) -> None:
        target = rate_hz / STEER_UNIT_HZ
        self.u += (target - self.u) * (1.0 - math.exp(-dt / STEER_TAU_S))

    def kinematic_deg(self) -> float:
        return self.gain_deg * self.u


@dataclass
class WingMotorIndex:
    """The 64 wing motor neurons: row, type, side; from the FL1 receipt's wing cell table."""
    rows: np.ndarray
    types: list
    sides: list

    @classmethod
    def from_fl1(cls, receipt: dict, arm: str = "wi1") -> "WingMotorIndex":
        cells = receipt["summary"][arm]["wing_cells"]
        return cls(np.array([c["row"] for c in cells]), [c["cell_type"] for c in cells], [c["side"] for c in cells])

    def rates_for(self, receipt: dict, arm: str) -> np.ndarray:
        cells = receipt["summary"][arm]["wing_cells"]
        FL2.require([c["row"] for c in cells] == self.rows.tolist(), "wing cell order differs between arms")
        return np.array([c["hz"] for c in cells], float)


class FlightMotorLayer:
    """Motor-neuron rates (per tick) -> power and steering activations -> wing kinematics,
    with the haltere reflex added in kinematic units. `update(st, t)` matches FL2B.run's
    controller interface; `feed(rates_hz)` sets the motor input for the next ticks."""

    def __init__(self, rig: FL2.Rig, geom: StrokeGeometry, line: PowerLine, index: WingMotorIndex,
                 trim_shift: float, theta_ref: float = 0.0, anchor_hz: float = POWER_ANCHOR_HZ,
                 reflex=None, altitude_loop=None, servo_gain_front: float = 1.0, servo_gain_amp: float = 1.0):
        self.rig, self.geom, self.line, self.index = rig, geom, line, index
        self.trim_shift, self.theta_ref = trim_shift, theta_ref
        self.power = {s: PowerMuscle(line, anchor_hz) for s in ("left", "right")}
        self.steer = {s: {m: SteeringMuscle(m) for m in STEERING_ROLES} for s in ("left", "right")}
        self.reflex, self.altitude_loop = reflex, altitude_loop
        self.servo_gain_front, self.servo_gain_amp = servo_gain_front, servo_gain_amp
        self.rates = np.zeros(len(index.rows))
        self.pooled = {"left": {"DLM": 0.0, "DVM": 0.0, "all": 0.0}, "right": {"DLM": 0.0, "DVM": 0.0, "all": 0.0}}
        self.cmd = geom.kin((line.a_hover, line.a_hover), (0.0, 0.0), (0.0, 0.0), trim_shift)
        self.last = {}

    def feed(self, rates_hz: np.ndarray) -> None:
        self.rates = np.asarray(rates_hz, float)
        for side in ("left", "right"):
            sel = [i for i, s in enumerate(self.index.sides) if s == side]
            dlm = [self.rates[i] for i in sel if POWER_TYPES.get(self.index.types[i]) == "DLM"]
            dvm = [self.rates[i] for i in sel if POWER_TYPES.get(self.index.types[i]) == "DVM"]
            self.pooled[side] = {"DLM": float(np.mean(dlm)) if dlm else 0.0, "DVM": float(np.mean(dvm)) if dvm else 0.0,
                                 "all": float(np.mean(dlm + dvm)) if (dlm or dvm) else 0.0}

    def set_power_steady(self, rate_hz: float) -> None:
        for p in self.power.values():
            p.c = rate_hz

    def steering_rates(self, side: str) -> dict:
        out = {}
        for i, (t, s) in enumerate(zip(self.index.types, self.index.sides)):
            if s == side and t in STEERING_ROLES:
                out[t] = out.get(t, 0.0) + float(self.rates[i])
        return out

    def step_muscles(self, dt: float) -> None:
        for side in ("left", "right"):
            self.power[side].step(self.pooled[side]["all"], dt)
            sr = self.steering_rates(side)
            for m, mus in self.steer[side].items():
                mus.step(sr.get(m, 0.0), dt)

    def steering_kinematics(self, side: str) -> dict:
        d = {"front": 0.0, "back": 0.0, "amp": 0.0, "aoa": 0.0, "freq": 0.0}
        for mus in self.steer[side].values():
            if mus.dial is not None:
                d[mus.dial] += mus.kinematic_deg()
        return d

    def update(self, st: FL2B.BodyState, t: float) -> FL2B.Kin:
        self.step_muscles(CONTROL_DT)
        front, back, asym = 0.0, 0.0, 0.0
        if self.reflex is not None:
            w = self.rig.data.qvel[self.rig.root_dof + 3: self.rig.root_dof + 6]     # thorax-frame angular velocity
            front, asym = self.reflex.update(st, float(w[1]), float(w[0]), t)     # deg achieved
        da_alt = self.altitude_loop.update(st, t) if self.altitude_loop is not None else 0.0
        a_pow = {s: self.power[s].amplitude() for s in ("left", "right")}
        fronts, backs, apows = [], [], []
        for k, side in enumerate(("left", "right")):
            sk = self.steering_kinematics(side)
            f_deg = sk["front"] + front / self.servo_gain_front
            b_deg = sk["back"]
            amp_deg = sk["amp"] + (asym if side == "left" else -asym) / (2.0 * self.servo_gain_amp)
            fronts.append(math.radians(f_deg + amp_deg / 2.0))
            backs.append(math.radians(b_deg + amp_deg / 2.0))
            apows.append(a_pow[side] + da_alt)
        self.cmd = self.geom.kin(tuple(apows), tuple(fronts), tuple(backs), self.trim_shift)
        self.last = {"front_deg": front, "asym_deg": asym, "a_power": a_pow, "da_alt": da_alt}
        return self.cmd


# ---- the reflex -------------------------------------------------------------------------------------------------
class AnimalReflex:
    """Whitehead 2015 pitch PI and Beatus 2015 roll PI, one command per wingbeat from the
    beat's mean body-frame rates and angles, after a pure delay. Outputs in ACHIEVED deg:
    bilateral front extension (positive = forward = nose-up torque) and stroke amplitude
    asymmetry left minus right (positive = left wing flaps more = left side up)."""

    def __init__(self, pitch_ki=PITCH_KI, pitch_kp_ms=PITCH_KP_MS, pitch_delay_ms=PITCH_DELAY_MS,
                 roll_ki=ROLL_KI, roll_kp_ms=ROLL_KP_MS, roll_delay_ms=ROLL_DELAY_MS,
                 pitch_on=True, roll_on=True, theta_ref: float = 0.0):
        self.pitch_ki, self.pitch_kp_ms, self.pitch_delay = pitch_ki, pitch_kp_ms, pitch_delay_ms * 1e-3
        self.roll_ki, self.roll_kp_ms, self.roll_delay = roll_ki, roll_kp_ms, roll_delay_ms * 1e-3
        self.pitch_on, self.roll_on, self.theta_ref = pitch_on, roll_on, theta_ref
        self.period = 1.0 / F0
        self.next_beat = self.period
        self.samples: list = []
        self.qp: deque = deque()
        self.qr: deque = deque()
        self.front, self.asym = 0.0, 0.0
        self.n_updates = self.clamped_front = self.clamped_asym = 0
        self.max_front = self.max_asym = 0.0

    def update(self, st: FL2B.BodyState, pitch_rate_body: float, roll_rate_body: float, t: float) -> tuple:
        self.samples.append([st.theta, pitch_rate_body, math.radians(st.roll_deg), roll_rate_body])
        if t >= self.next_beat - 1e-9:
            self.next_beat += self.period
            theta, thd, roll, rolld = np.mean(self.samples, axis=0)
            self.samples = []
            dth = math.degrees(theta - self.theta_ref)                     # nose-down positive, deg
            front = self.pitch_ki * dth + self.pitch_kp_ms * 1e-3 * math.degrees(thd) if self.pitch_on else 0.0
            if abs(front) > FRONT_CLAMP_DEG:
                self.clamped_front += 1
            front = float(np.clip(front, -FRONT_CLAMP_DEG, FRONT_CLAMP_DEG))
            # left side up = positive roll; the low side flaps more, so asym (left - right) = -PI(roll)
            asym = -(self.roll_ki * math.degrees(roll) + self.roll_kp_ms * 1e-3 * math.degrees(rolld)) if self.roll_on else 0.0
            if abs(asym) > ASYM_CLAMP_DEG:
                self.clamped_asym += 1
            asym = float(np.clip(asym, -ASYM_CLAMP_DEG, ASYM_CLAMP_DEG))
            self.qp.append((t + self.pitch_delay, front))
            self.qr.append((t + self.roll_delay, asym))
            self.n_updates += 1
        while self.qp and self.qp[0][0] <= t + 1e-9:
            self.front = self.qp.popleft()[1]
        while self.qr and self.qr[0][0] <= t + 1e-9:
            self.asym = self.qr.popleft()[1]
        self.max_front = max(self.max_front, abs(self.front))
        self.max_asym = max(self.max_asym, abs(self.asym))
        return self.front, self.asym

    def record(self) -> dict:
        return {"pitch": {"K_i_deg_per_deg": self.pitch_ki, "K_p_ms": self.pitch_kp_ms, "delay_ms": self.pitch_delay * 1e3, "on": self.pitch_on},
                "roll": {"K_i_deg_per_deg": self.roll_ki, "K_p_ms": self.roll_kp_ms, "delay_ms": self.roll_delay * 1e3, "on": self.roll_on},
                "updates": self.n_updates, "clamped_front": self.clamped_front, "clamped_asym": self.clamped_asym,
                "max_abs_front_deg": self.max_front, "max_abs_asym_deg": self.max_asym,
                "sampling": "one command per wingbeat from the beat's mean body-frame state, then the pure delay"}


class AltitudeLoop:
    """FL2b's altitude loop, EXTERNAL and labelled: amplitude <- altitude at 2 Hz, zeta 1,
    once per beat. The animal has no fast altitude reflex; this only holds the hover for the
    HOVERS reading and is reported as a separate arm."""

    def __init__(self, mass_g: float, z_ref: float, dF_damp: float):
        self.mass_g, self.weight, self.z_ref, self.dF_damp = mass_g, mass_g * G, z_ref, dF_damp
        wz = 2 * math.pi * FL2B.ALT_BW_HZ
        self.kz, self.kvz = wz * wz, 2 * FL2B.ALT_ZETA * wz
        self.period, self.next_beat = 1.0 / F0, 1.0 / F0
        self.samples: list = []
        self.da = 0.0

    def update(self, st: FL2B.BodyState, t: float) -> float:
        self.samples.append([st.com[2], st.v[2]])
        if t >= self.next_beat - 1e-9:
            self.next_beat += self.period
            z, vz = np.mean(self.samples, axis=0)
            self.samples = []
            F_des = self.weight + self.mass_g * (self.kz * (self.z_ref - z) - self.kvz * vz)
            self.da = float(np.clip((F_des - self.weight) / self.dF_damp, -0.1, 0.1))
        return self.da


class BodyKick:
    """A torque pulse about a thorax-frame axis (world torque = R axis), labelled disturbance."""

    def __init__(self, torque_nNm: float, at_s: float, dur_s: float, axis_body: tuple, name: str):
        self.torque, self.at, self.dur, self.axis, self.name = torque_nNm, at_s, dur_s, np.asarray(axis_body, float), name

    def apply(self, rig: FL2.Rig, t: float) -> bool:
        on = self.at <= t < self.at + self.dur
        if on:
            R = rig.data.xmat[rig.thorax].reshape(3, 3)
            rig.data.xfrc_applied[rig.thorax, 3:6] = self.torque * (R @ self.axis)
        return on


class KickSequence:
    def __init__(self, kicks: list):
        self.kicks = kicks

    def apply(self, rig: FL2.Rig, t: float) -> bool:
        rig.data.xfrc_applied[rig.thorax, 3:6] = 0.0
        return any(k.apply(rig, t) for k in self.kicks)


def pitch_kick(at_s: float = PITCH_KICK["at_s"]) -> BodyKick:
    # FL2b used world +y (nose-down positive); at yaw 0 the thorax y axis is world y
    return BodyKick(PITCH_KICK["nNm"], at_s, PITCH_KICK["dur_s"], (0.0, 1.0, 0.0), "pitch nose-down")


def roll_kick(at_s: float = ROLL_KICK["at_s"]) -> BodyKick:
    return BodyKick(ROLL_KICK["nNm"], at_s, ROLL_KICK["dur_s"], (-1.0, 0.0, 0.0), "roll left-side-down")


# ---- readers ----------------------------------------------------------------------------------------------------
def hold_read(R: np.ndarray, window_s: float) -> dict:
    h = FL2B.hover_read(R, window_s)
    pitch_ok = h["max_abs_pitch_error_deg"] <= HOLD_TOL_DEG
    roll_ok = h["max_abs_roll_deg"] <= HOLD_TOL_DEG
    h["holds_attitude"] = bool(pitch_ok and roll_ok)
    h["attitude_verdict"] = "HOLDS" if (pitch_ok and roll_ok) else ("PITCH_ONLY" if pitch_ok else ("ROLL_ONLY" if roll_ok else "TUMBLES"))
    w = R[:, FL2B.C_T] < window_s
    dz = R[w, FL2B.C_COM][:, 2] - R[0, FL2B.C_COM][2]
    t = R[w, FL2B.C_T]
    h["climb_mm_at_end"] = float(dz[-1])
    h["mean_vz_mm_s"] = float(dz[-1] / t[-1]) if t[-1] > 0 else 0.0
    h["amp_l_mean"] = float(R[w, FL2B.C_AL].mean())
    h["amp_r_mean"] = float(R[w, FL2B.C_AR].mean())
    h["shift_l_mean_deg"] = float(np.degrees(R[w, FL2B.C_SL].mean()))
    return h


def pitch_recovery_read(R: np.ndarray, kick: BodyKick, until_s: float) -> dict:
    t = R[:, FL2B.C_T]
    sel = t < until_s
    rc = FL2B.recovery_read(R[sel], FL2B.Kick(kick.torque, kick.at, kick.dur))
    post = (t >= kick.at) & sel
    dp = R[post, FL2B.C_PITCH] - rc["pre_kick_body_pitch_deg"]
    tp = t[post] - kick.at
    peak = rc["peak_pitch_excursion_deg"]
    # Whitehead 2015: time from the perturbation onset to 90% correction of the deflection
    after_peak = tp >= rc["time_to_peak_s"]
    within = np.abs(dp) <= 0.1 * abs(peak)
    idx = np.nonzero(after_peak & within)[0]
    rc["correction_90pct_s"] = float(tp[idx[0]]) if len(idx) else None
    rc["animal_correction_90pct_ms"] = "29 +- 8 (Whitehead 2015, n = 32)"
    return rc


def roll_recovery_read(R: np.ndarray, kick: BodyKick, until_s: float) -> dict:
    t = R[:, FL2B.C_T]
    pre = (t >= kick.at - 0.05) & (t < kick.at)
    post = (t >= kick.at) & (t < until_s)
    roll_pre = float(R[pre, FL2B.C_ROLL].mean())
    dr = R[post, FL2B.C_ROLL] - roll_pre
    tp = t[post] - kick.at
    peak_i = int(np.argmax(np.abs(dr)))
    peak = float(dr[peak_i])
    after_peak = tp >= tp[peak_i]
    idx = np.nonzero(after_peak & (np.abs(dr) <= 0.1 * abs(peak)))[0]
    outside = np.nonzero(np.abs(dr) > 2.0)[0]
    return {"kick_nNm": kick.torque, "kick_at_s": kick.at, "kick_duration_s": kick.dur, "axis": "thorax long axis, left side down",
            "pre_kick_roll_deg": roll_pre, "peak_roll_excursion_deg": peak, "time_to_peak_s": float(tp[peak_i]),
            "correction_to_10pct_s": float(tp[idx[0]]) if len(idx) else None,
            "correction_to_10pct_beats": float(tp[idx[0]] * F0) if len(idx) else None,
            "settle_within_2deg_s": float(tp[outside[-1]]) if len(outside) else 0.0,
            "roll_error_at_end_deg": float(dr[-1]), "recovered": bool(abs(dr[-1]) <= 2.0),
            "max_abs_pitch_error_during_deg": float(np.abs(R[post, FL2B.C_PITCH] - BODY_PITCH_DEG).max()),
            "max_abs_yaw_deg": float(np.abs(R[post, FL2B.C_YAW]).max()),
            "animal": "Beatus 2015: 60 deg in 13.5 ms, recovered in 35 ms / 8 beats; population 6.8 +- 1.6 beats to 10%"}


# ---- footage ------------------------------------------------------------------------------------------------------
class ReflexRecorder(FL2B.HoverRecorder):
    def __init__(self, rig, path, tether_cycles, label, layer):
        super().__init__(rig, path, tether_cycles, label)
        self.layer = layer
        self.every_free = 60

    def __call__(self, rig, released, cmd, controller):
        d = rig.data
        self.step += 1
        if not released:
            if d.time < self.tether_start or self.step % self.every_tethered:
                return
            tempo, phase = self.tempo(self.every_tethered), "TETHERED, trimmed pattern"
        else:
            if self.step % self.every_free:
                return
            tempo, phase = self.tempo(self.every_free), "RELEASED, thoracic reflex model"
            self.cam.lookat[:] = 0.7 * self.cam.lookat + 0.3 * d.xpos[rig.thorax]
        st = FL2B.body_state(rig)
        if self.z0 is None:
            self.z0 = st.com[2]
            self.cam.lookat[:] = d.xpos[rig.thorax]
        self.renderer.update_scene(d, self.cam, self.opt)
        img = self.Image.fromarray(self.renderer.render())
        draw = self.ImageDraw.Draw(img)
        kick = bool(np.any(d.xfrc_applied[rig.thorax]))
        last = self.layer.last or {}
        lines = [
            f"FL3a flybody + flight-muscle layer, {self.label}, recorded {self.stamp}",
            f"{phase}   t = {d.time * 1e3:7.1f} ms   {tempo}" + ("   TORQUE PULSE" if kick else ""),
            f"body pitch {st.body_pitch_deg:5.1f} deg (hover 47.5)   roll {st.roll_deg:+5.1f} deg   altitude {st.com[2] - self.z0:+.2f} mm (floor is visual only, no contact)",
            f"reflex (animal gains): front stroke {last.get('front_deg', 0.0):+5.1f} deg, L-R amplitude {last.get('asym_deg', 0.0):+5.1f} deg;   power a {cmd.amp[0]:.3f}/{cmd.amp[1]:.3f}",
            "BODY CALIBRATION: haltere -> steering reflex is a model with Whitehead/Beatus 2015 gains; no nervous system",
        ]
        if self.layer.altitude_loop is not None:
            lines.append(f"EXTERNAL altitude loop (labelled): amplitude <- altitude, 2 Hz, da {last.get('da_alt', 0.0):+.3f}; the reflex model has no altitude term")
        else:
            lines.append("no altitude loop: the reflex model holds attitude only, the body sinks about 6 mm/s")
        y = 8
        for line in lines:
            draw.text((10, y), line, fill=(20, 20, 20))
            y += 18
        self.frames.append(np.asarray(img))


# ---- main ------------------------------------------------------------------------------------------------------------
def load_receipts(rig_receipt: Path = FL2B_RECEIPT) -> tuple:
    """The body receipt (FL2b: spawn-pose legs; FL2c: flybody's retracted flight pose) and FL1."""
    FL2.require(rig_receipt.is_file() and FL1_RECEIPT.is_file(), "body and FL1 receipts are required")
    return json.loads(rig_receipt.read_text()), json.loads(FL1_RECEIPT.read_text())


def build(fl2b: dict, fl1: dict):
    """The rig is built in the leg pose the body receipt was measured in (FL2b has none: spawn)."""
    pat = FL2B.Pattern(FL2.load_pattern())
    geom = StrokeGeometry(pat)
    line = PowerLine(fl2b)
    index = WingMotorIndex.from_fl1(fl1)
    trim3 = fl2b["trim3"]
    FL2.LEG_POSE = fl2b.get("leg_pose", "spawn")
    rig = FL2B.make_rig(trim3["stroke_plane_deg"])
    FL2.require(rig.leg_pose == FL2.LEG_POSE, "rig leg pose differs from the receipt's")
    return pat, geom, line, index, trim3, rig


def jacobian_check(rig, pat, geom, line, trim3, tau_x_slope, log) -> dict:
    """Tethered: the new dials against FL2b's derivatives, and the servo's delivered gains.
    The asymmetry arm commands a left-minus-right peak-to-peak difference of JAC_STEP_DEG,
    split equally between the front and back extremes of each wing."""
    a0, s0 = trim3["amp"], trim3["shift_rad"]
    ld = trim3["local_derivatives_at_trim"]
    step = math.radians(JAC_STEP_DEG)
    base = FL2B.tethered_point(rig, pat, geom.kin((a0, a0), (0, 0), (0, 0), s0))
    front = FL2B.tethered_point(rig, pat, geom.kin((a0, a0), (step, step), (0, 0), s0))
    back = FL2B.tethered_point(rig, pat, geom.kin((a0, a0), (0, 0), (step, step), s0))
    asym = FL2B.tethered_point(rig, pat, geom.kin((a0, a0), (step / 4, -step / 4), (step / 4, -step / 4), s0))

    def d(p, key, i=None):
        v, b = (p[key][i], base[key][i]) if i is not None else (p[key], base[key])
        return float(v - b)

    # predictions from FL2b's local derivatives: a front extension of D rad is da = D/phi0 and
    # dshift = -D dev_max/phi0; a back extension da = D/phi0, dshift = +D (1 - dev_max/phi0)
    da = step / geom.phi0
    dshift_front = -step * geom.dev_max / geom.phi0
    dshift_back = step * (1.0 - geom.dev_max / geom.phi0)
    pred = {
        "front_dtau_y_nNm": ld["d_tau_y_d_amp_nNm"] * da + ld["d_tau_y_d_shift_nNm_per_rad"] * dshift_front,
        "front_dforce_over_W": ld["d_force_over_weight_d_amp"] * da,
        "back_dtau_y_nNm": ld["d_tau_y_d_amp_nNm"] * da + ld["d_tau_y_d_shift_nNm_per_rad"] * dshift_back,
        "back_dforce_over_W": ld["d_force_over_weight_d_amp"] * da,
        "asym_dtau_x_nNm": tau_x_slope * da,   # FL2b baseline slope, nN m per unit (a_l - a_r); here a_l - a_r = step / phi0
    }
    got = {
        "front_dtau_y_nNm": d(front, "tau_nNm", 1), "front_dforce_over_W": d(front, "force_over_weight"),
        "back_dtau_y_nNm": d(back, "tau_nNm", 1), "back_dforce_over_W": d(back, "force_over_weight"),
        "asym_dtau_x_nNm": d(asym, "tau_nNm", 0), "asym_dtau_z_nNm": d(asym, "tau_nNm", 2),
    }
    # servo gains: achieved change per commanded change (left wing). The front extreme is
    # estimated from the achieved mean and peak-to-peak with the pattern's front fraction.
    def front_extreme(p):
        return p["l_stroke_achieved_mean_rad"] + p["l_stroke_achieved_amp_rad"] * geom.dev_min / geom.phi0

    q_base = base["l_stroke_achieved_amp_rad"]
    servo_gain_front = (front_extreme(front) - front_extreme(base)) / (-step)   # forward is negative stroke angle
    servo_gain_pp = (front["l_stroke_achieved_amp_rad"] - q_base) / step        # peak-to-peak change per rad of front extension
    servo_gain_asym = ((asym["l_stroke_achieved_amp_rad"] - q_base) - (asym["r_stroke_achieved_amp_rad"] - q_base)) / step
    mean_shift_gain = (front["l_stroke_achieved_mean_rad"] - base["l_stroke_achieved_mean_rad"]) / dshift_front
    ok = (got["front_dtau_y_nNm"] < 0 and got["front_dforce_over_W"] > 0 and got["back_dtau_y_nNm"] > 0
          and abs(got["asym_dtau_x_nNm"] - pred["asym_dtau_x_nNm"]) <= 0.25 * abs(pred["asym_dtau_x_nNm"]))
    out = {"step_deg": JAC_STEP_DEG, "base": base, "front": front, "back": back, "asym": asym,
           "predicted_from_fl2b": pred, "measured": got,
           "servo": {"achieved_front_extreme_per_commanded": float(servo_gain_front),
                     "achieved_pp_change_per_commanded_front_rad": float(servo_gain_pp),
                     "achieved_asym_per_commanded_asym": float(servo_gain_asym),
                     "achieved_mean_shift_per_commanded": float(mean_shift_gain)},
           "verdict": "JACOBIAN" if ok else "JACOBIAN_MISMATCH"}
    log(f"[jacobian] front +{JAC_STEP_DEG:.0f} deg: dtau_y {got['front_dtau_y_nNm']:+.3f} (pred {pred['front_dtau_y_nNm']:+.3f}) nN m, "
        f"d|F|/W {got['front_dforce_over_W']:+.4f} (pred {pred['front_dforce_over_W']:+.4f}); back: dtau_y {got['back_dtau_y_nNm']:+.3f} "
        f"(pred {pred['back_dtau_y_nNm']:+.3f}); asym {JAC_STEP_DEG:.0f} deg: dtau_x {got['asym_dtau_x_nNm']:+.3f} (pred {pred['asym_dtau_x_nNm']:+.3f}), "
        f"dtau_z {got['asym_dtau_z_nNm']:+.3f}; servo: front extreme {servo_gain_front:.3f}, pp {servo_gain_pp:.3f}, asym {servo_gain_asym:.3f}, mean shift {mean_shift_gain:.3f} -> {out['verdict']}")
    return out


def released(rig, pat, layer, kin0, hover_s, kicks=None, recorder=None, extra_s=0.0) -> dict:
    out = FL2B.run(rig, pat, kin0, PRE_RELEASE_CYCLES, hover_s + extra_s, controller=layer, kick=kicks, frame_hook=recorder)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="short hovers, no grid, no receipt")
    ap.add_argument("--no-grid", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    started = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    t_start = time.time()
    log_lines: list = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    fl2b, fl1 = load_receipts()
    pat, geom, line, index, trim3, rig = build(fl2b, fl1)
    st0 = FL2B.body_state(rig)
    mass, W = rig.mass_g, rig.weight_uN
    a0, s0 = trim3["amp"], trim3["shift_rad"]
    kin0 = geom.kin((a0, a0), (0, 0), (0, 0), s0)
    FL2.require(abs(kin0.amp[0] - a0) < 1e-12 and abs(kin0.shift[0] - s0) < 1e-12, "trim reproduction")
    log(f"[body] mass {mass * 1e3:.4f} mg, weight {W:.3f} uN, trim3 a {a0:.4f} shift {math.degrees(s0):+.2f} deg stroke plane "
        f"{trim3['stroke_plane_deg']:+.2f} deg; Phi0 {math.degrees(geom.phi0):.1f} deg, dev_max {math.degrees(geom.dev_max):.1f}; "
        f"power line P = {line.intercept:+.2f} + {line.slope:.2f} a uW, P_hover {line.p_hover:.2f} uW at a {line.a_hover:.4f}")
    pm = PowerMuscle(line)
    a_max_rate = POWER_ANCHOR_HZ * ((line.power(AMP_MAX) / line.p_hover) ** (1.0 / POWER_EXPONENT))
    a_min_rate = POWER_ANCHOR_HZ * ((line.power(AMP_MIN) / line.p_hover) ** (1.0 / POWER_EXPONENT))
    power_law = {"anchor_hz": POWER_ANCHOR_HZ, "exponent": POWER_EXPONENT, "tau_rise_s": POWER_TAU_RISE_S, "tau_fall_s": POWER_TAU_FALL_S,
                 "amplitude_at_rate": {f"{r:g}": pm.steady_amplitude(r) for r in (1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 35.0, 67.0)},
                 "rate_at_amp_max_hz": a_max_rate, "rate_at_amp_min_hz": a_min_rate, "animal_band_hz": list(POWER_BAND_HZ)}
    log(f"[power] a(r): " + ", ".join(f"{r} Hz -> {v[0]:.3f} {v[1]}" for r, v in power_law["amplitude_at_rate"].items())
        + f"; the body's box 0.90..1.20 spans {a_min_rate:.2f}..{a_max_rate:.2f} Hz of the animal's 5-20 Hz band")

    # Part 1 ------------------------------------------------------------------------------------------------------------
    jac = jacobian_check(rig, pat, geom, line, trim3, fl2b["derivatives_at_baseline"]["tau_x_nNm_vs_asym"]["slope"], log)
    servo_front = min(1.0, max(0.3, jac["servo"]["achieved_front_extreme_per_commanded"]))
    servo_asym = min(1.0, max(0.3, jac["servo"]["achieved_asym_per_commanded_asym"]))

    # Part 2 ------------------------------------------------------------------------------------------------------------
    hover_s = 0.25 if args.smoke else HOVER_S
    recover_s = 0.1 if args.smoke else RECOVER_S
    kick_p = pitch_kick(hover_s)
    kick_r = roll_kick(hover_s + recover_s)
    kicks = KickSequence([kick_p, kick_r])
    total_extra = 2 * recover_s
    z_ref = float(st0.com[2])
    ld = trim3["local_derivatives_at_trim"]

    def make_layer(reflex, altitude=False, anchor=POWER_ANCHOR_HZ):
        alt = AltitudeLoop(mass, z_ref, ld["dF_d_amp_uN"]) if altitude else None
        lay = FlightMotorLayer(rig, geom, line, index, s0, theta_ref=trim3["theta0_rad"], anchor_hz=anchor, reflex=reflex,
                               altitude_loop=alt, servo_gain_front=servo_front, servo_gain_amp=servo_asym)
        return lay

    def steady_rates(power_hz: float) -> np.ndarray:
        r = np.zeros(len(index.rows))
        for i, t in enumerate(index.types):
            if t in POWER_TYPES:
                r[i] = power_hz
        return r

    arms = {}

    def run_arm(name, reflex, altitude=False, render=False, with_kicks=True, log_it=True, rates=None, anchor=POWER_ANCHOR_HZ):
        t0 = time.time()
        layer = make_layer(reflex, altitude, anchor)
        layer.feed(steady_rates(POWER_ANCHOR_HZ) if rates is None else rates)
        layer.set_power_steady(layer.pooled["left"]["all"] if rates is not None else POWER_ANCHOR_HZ)
        for side in ("left", "right"):
            layer.power[side].c = layer.pooled[side]["all"]
        # steering activations start at their steady state for constant rates
        for side in ("left", "right"):
            sr = layer.steering_rates(side)
            for m, mus in layer.steer[side].items():
                mus.u = sr.get(m, 0.0) / STEER_UNIT_HZ
        rec = ReflexRecorder(rig, HERE / f"fl3_{name}_hover.mp4", PRE_RELEASE_CYCLES, name, layer) if render else None
        out = released(rig, pat, layer, kin0, hover_s, kicks=kicks if with_kicks else None, recorder=rec, extra_s=total_extra if with_kicks else 0.0)
        R = out["free"]
        h = hold_read(R, hover_s)
        res = {"hover": h, "wall_s": round(time.time() - t0, 2), "steps": out["steps"],
               "reflex": reflex.record() if reflex is not None else None, "altitude_loop": altitude,
               "power_pooled_hz": layer.pooled, "power_saturated_ticks": {s: layer.power[s].saturated_high for s in layer.power},
               "steering_kinematics_deg": {s: layer.steering_kinematics(s) for s in ("left", "right")},
               "final_command": FL2B.kin_dict(layer.cmd)}
        if with_kicks:
            res["pitch_recovery"] = pitch_recovery_read(R, kick_p, kick_r.at)
            res["roll_recovery"] = roll_recovery_read(R, kick_r, hover_s + total_extra)
        if rec is not None:
            res["clip"] = rec.close()
        if log_it:
            log(f"[{name}] {h['attitude_verdict']} / {h['verdict']}: pitch |err| max {h['max_abs_pitch_error_deg']:.2f} deg (mean {h['mean_body_pitch_deg']:.1f}, sd last half "
                f"{h['pitch_sd_about_setpoint_last_half_deg']:.2f}), roll max {h['max_abs_roll_deg']:.2f}, yaw max {h['max_abs_yaw_deg']:.2f}, altitude |err| max "
                f"{h['max_abs_altitude_error_mm']:.2f} mm, climb {h['climb_mm_at_end']:+.2f} mm ({h['mean_vz_mm_s']:+.0f} mm/s), drift x {h['horizontal_drift_mm']['x_final']:+.1f} mm, "
                f"amp {h['amp_l_mean']:.3f}/{h['amp_r_mean']:.3f}, shift {h['shift_l_mean_deg']:+.2f} deg, aero {h['aero_power_uW_mean']:.1f} uW ({res['wall_s']} s)")
            if reflex is not None:
                rr = res["reflex"]
                log(f"[{name}]   reflex: {rr['updates']} beats, max |front| {rr['max_abs_front_deg']:.2f} deg, max |asym| {rr['max_abs_asym_deg']:.2f} deg, "
                    f"clamps {rr['clamped_front']}/{rr['clamped_asym']}")
            if with_kicks:
                pr, ro = res["pitch_recovery"], res["roll_recovery"]
                c90 = pr["correction_90pct_s"]
                c10 = ro["correction_to_10pct_s"]
                log(f"[{name}]   pitch kick: peak {pr['peak_pitch_excursion_deg']:+.1f} deg at {pr['time_to_peak_s'] * 1e3:.0f} ms, 90% corrected at "
                    f"{(c90 * 1e3) if c90 is not None else float('nan'):.0f} ms (animal 29 +- 8), within 2 deg after {pr['settle_within_2deg_s'] * 1e3:.0f} ms, recovered {pr['recovered']}")
                log(f"[{name}]   roll kick: peak {ro['peak_roll_excursion_deg']:+.1f} deg at {ro['time_to_peak_s'] * 1e3:.0f} ms, to 10% at "
                    f"{(c10 * 1e3) if c10 is not None else float('nan'):.0f} ms = {(ro['correction_to_10pct_beats'] or float('nan')):.1f} beats (animal 6.8 +- 1.6), "
                    f"pitch during {ro['max_abs_pitch_error_during_deg']:.1f} deg, yaw {ro['max_abs_yaw_deg']:.1f}, recovered {ro['recovered']}")
        return res

    arms["animal_reflex"] = run_arm("animal_reflex", AnimalReflex(theta_ref=trim3["theta0_rad"]), render=args.render and not args.smoke)
    arms["animal_reflex_altitude_loop"] = run_arm("animal_reflex_altitude_loop", AnimalReflex(theta_ref=trim3["theta0_rad"]), altitude=True,
                                                 render=args.render and not args.smoke)
    arms["pitch_only"] = run_arm("pitch_only", AnimalReflex(roll_on=False, theta_ref=trim3["theta0_rad"]))
    arms["no_reflex"] = run_arm("no_reflex", None, with_kicks=False)

    grid = {}
    if not args.smoke and not args.no_grid:
        n_hold = 0
        for ki_k in SENS_SCALES:
            for kp_k in SENS_SCALES:
                for d_k in SENS_SCALES:
                    ki, kp, dl = PITCH_KI + ki_k * PITCH_KI_SD, PITCH_KP_MS + kp_k * PITCH_KP_MS_SD, PITCH_DELAY_MS + d_k * PITCH_DELAY_MS_SD
                    r = run_arm(f"grid ki {ki:.2f} kp {kp:.1f} ms d {dl:.1f} ms", AnimalReflex(pitch_ki=ki, pitch_kp_ms=kp, pitch_delay_ms=dl, theta_ref=trim3["theta0_rad"]),
                                with_kicks=False, log_it=False)
                    h = r["hover"]
                    key = f"ki{ki:.2f}_kp{kp:.1f}_d{dl:.1f}"
                    grid[key] = {"K_i": ki, "K_p_ms": kp, "delay_ms": dl, "attitude_verdict": h["attitude_verdict"],
                                 "max_abs_pitch_error_deg": h["max_abs_pitch_error_deg"], "max_abs_roll_deg": h["max_abs_roll_deg"],
                                 "pitch_sd_last_half_deg": h["pitch_sd_about_setpoint_last_half_deg"], "climb_mm": h["climb_mm_at_end"]}
                    n_hold += int(h["holds_attitude"])
                    log(f"[grid] K_i {ki:.2f} K_p {kp:.1f} ms delay {dl:.1f} ms: {h['attitude_verdict']} pitch max {h['max_abs_pitch_error_deg']:.2f} deg sd {h['pitch_sd_about_setpoint_last_half_deg']:.2f}")
        grid["_summary"] = {"holds": n_hold, "of": len(SENS_SCALES) ** 3}
        log(f"[grid] {n_hold} of {len(SENS_SCALES) ** 3} points of the animal's +-1 sd box hold attitude for {hover_s} s")

    # Part 3 ------------------------------------------------------------------------------------------------------------
    fl1_arms = {}
    for arm in ("wi1", "flight_dng02", "flight_dnp31"):
        rates = index.rates_for(fl1, arm)
        lay = make_layer(None)
        lay.feed(rates)
        pooled = lay.pooled
        pred = {f"anchor_{a:g}Hz": {side: PowerMuscle(line, a).steady_amplitude(pooled[side]["all"]) for side in ("left", "right")} for a in ANCHOR_SCAN_HZ}
        steer = {}
        for side in ("left", "right"):
            sr = lay.steering_rates(side)
            for m, mus in lay.steer[side].items():
                mus.u = sr.get(m, 0.0) / STEER_UNIT_HZ
            steer[side] = {"rates_hz": sr, "kinematics_deg": lay.steering_kinematics(side)}
        log(f"[fl1 {arm}] pooled A-IFM Hz L/R {pooled['left']['all']:.1f}/{pooled['right']['all']:.1f} (DLM {pooled['left']['DLM']:.1f}/{pooled['right']['DLM']:.1f}, "
            f"DVM {pooled['left']['DVM']:.1f}/{pooled['right']['DVM']:.1f}); a at 5 Hz anchor L/R {pred['anchor_5Hz']['left'][0]:.3f} {pred['anchor_5Hz']['left'][1]} / "
            f"{pred['anchor_5Hz']['right'][0]:.3f}; at 10 Hz {pred['anchor_10Hz']['left'][0]:.3f} {pred['anchor_10Hz']['left'][1]}; at 20 Hz {pred['anchor_20Hz']['left'][0]:.3f} {pred['anchor_20Hz']['left'][1]}; "
            f"steering L front {steer['left']['kinematics_deg']['front']:+.1f} back {steer['left']['kinematics_deg']['back']:+.1f} amp {steer['left']['kinematics_deg']['amp']:+.1f} deg, "
            f"R front {steer['right']['kinematics_deg']['front']:+.1f} back {steer['right']['kinematics_deg']['back']:+.1f} amp {steer['right']['kinematics_deg']['amp']:+.1f} deg")
        body = run_arm(f"fl1 {arm} + reflex", AnimalReflex(theta_ref=trim3["theta0_rad"]), with_kicks=False, rates=rates)
        fl1_arms[arm] = {"pooled_hz": pooled, "predicted_amplitude": pred, "steering": steer, "body": body,
                         "power_verdict_5Hz_anchor": pred["anchor_5Hz"]["left"][1]}

    wall = round(time.time() - t_start, 1)
    h0 = arms["animal_reflex"]["hover"]
    summary = {"jacobian": jac["verdict"], "animal_reflex_attitude": h0["attitude_verdict"], "animal_reflex_hover": h0["verdict"],
               "animal_reflex_altitude_loop_hover": arms["animal_reflex_altitude_loop"]["hover"]["verdict"],
               "pitch_only_attitude": arms["pitch_only"]["hover"]["attitude_verdict"], "no_reflex": arms["no_reflex"]["hover"]["attitude_verdict"],
               "pitch_kick_90pct_ms": (arms["animal_reflex"]["pitch_recovery"]["correction_90pct_s"] or float("nan")) * 1e3,
               "roll_kick_10pct_beats": arms["animal_reflex"]["roll_recovery"]["correction_to_10pct_beats"],
               "grid_holds": grid.get("_summary"), "fl1_power": {a: v["power_verdict_5Hz_anchor"] for a, v in fl1_arms.items()}, "wall_s": wall}
    log(f"[summary] {json.dumps(summary, default=float)}")
    if args.smoke:
        return 0
    receipt = {
        "experiment": "FL3a flight-muscle layer with the animal's haltere reflex gains on the FL2b body, then the champion's recorded flight motor rates (body calibration; no CNS, no seed)",
        "started": started, "finished": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "wall_s": wall,
        "identity": {**FL2.identity(), "fl3_script_sha256": FL2.sha256(Path(__file__)), "fl2b_script_sha256": FL2.sha256(Path(FL2B.__file__)),
                     "fl2b_receipt_sha256": FL2.sha256(FL2B_RECEIPT), "fl1_receipt_sha256": FL2.sha256(FL1_RECEIPT)},
        "sources": SOURCES,
        "parameters": {"power": power_law, "steering": {"tau_s": STEER_TAU_S, "unit_hz": STEER_UNIT_HZ, "roles": {k: {"dial": v[0], "gain_deg": v[1], "class": v[2]} for k, v in STEERING_ROLES.items()}},
                       "reflex": {"pitch": {"K_i": PITCH_KI, "K_i_sd": PITCH_KI_SD, "K_p_ms": PITCH_KP_MS, "K_p_ms_sd": PITCH_KP_MS_SD, "delay_ms": PITCH_DELAY_MS, "delay_ms_sd": PITCH_DELAY_MS_SD},
                                  "roll": {"K_i": ROLL_KI, "K_i_sd": ROLL_KI_SD, "K_p_ms": ROLL_KP_MS, "K_p_ms_sd": ROLL_KP_MS_SD, "delay_ms": ROLL_DELAY_MS, "delay_ms_sd": ROLL_DELAY_MS_SD},
                                  "clamps_deg": {"front": FRONT_CLAMP_DEG, "asym": ASYM_CLAMP_DEG}, "servo_compensation": jac["servo"]},
                       "geometry": {"phi0_deg": math.degrees(geom.phi0), "dev_max_deg": math.degrees(geom.dev_max), "dev_min_deg": math.degrees(geom.dev_min), "stroke_mean_deg": math.degrees(geom.mean)},
                       "trim3": {k: trim3[k] for k in ("amp", "shift_deg", "stroke_plane_deg", "theta0_deg", "hover_body_pitch_deg")},
                       "protocol": {"pre_release_cycles": PRE_RELEASE_CYCLES, "hover_s": HOVER_S, "pitch_kick": PITCH_KICK, "roll_kick": ROLL_KICK, "recover_s": RECOVER_S,
                                    "hold_tol_deg": HOLD_TOL_DEG, "grid_scales_sd": list(SENS_SCALES), "anchor_scan_hz": list(ANCHOR_SCAN_HZ)},
                       "pre_set_reading": {"JACOBIAN": "front: nose-up and more lift; back: nose-down; asym roll slope within 25% of FL2b",
                                           "HOLDS": "|pitch - 47.5| <= 10 deg and |roll| <= 10 deg for 1.0 s", "HOVERS": "FL2b reading (altitude 2 mm, pitch 10 deg) with the altitude loop",
                                           "RECOVERS": "pitch back within 2 deg; 90% time vs 29 +- 8 ms; roll to 10% vs 6.8 +- 1.6 beats",
                                           "POWER": "a at the 5 Hz anchor: IN_BAND 0.98..1.08, SATURATED at 1.20, else value"},
                       **{k: v for k, v in FL2.parameters().items() if k in ("air_density_g_mm3", "air_viscosity_g_mm_s", "gravity_mm_s2", "physics_dt_s", "control_dt_s", "wingbeat_hz", "body_pitch_deg", "fluidcoef", "wing_servo_kp", "units")}},
        "body": {"mass_mg": mass * 1e3, "weight_uN": W, "spawn_body_pitch_deg": st0.body_pitch_deg},
        "jacobian": jac, "arms": arms, "grid": grid, "fl1_driven": fl1_arms, "summary": summary, "log": log_lines,
    }
    out_path = Path(args.out) if args.out else HERE / f"{time.strftime('%Y-%m-%d')}-fl3-flight-muscle-layer.json"
    out_path.write_text(json.dumps(receipt, indent=1, default=float))
    print(f"receipt {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
