"""Fixed-heading twin of the card's three left-right foot phase rows (judge, 2026-09-07).

The card's `phase_lf_rf_deg`, `phase_lm_rm_deg` and `phase_lh_rh_deg` come from
lanes/A/phasemeasures.from_capture, whose foot signal is each tip's fore-aft
coordinate in a frame that turns WITH the body (body_frame_fore_aft applies
R(t).T to tip minus root). A planted foot is fixed in the world, so while the
body yaws that foot's coordinate sweeps fore-aft in the body frame, and two
planted feet on opposite sides of the yaw axis sweep in opposite directions:
180 degrees of left-right phase with no step. Lane A measured it on the
champion (CD2-FD, packet exchange/2026-09-07-1300-laneA-judge-packet.md,
registration 6facf4288, reader lanes/A/cd2fd_is_the_front_row_the_body.py on
origin/lane-a): the front pair read 180, 174 and 168 degrees in the body frame
and 38, 5 and 51 with the frame held at the window's mean heading, on runs
whose heading swung 71 to 104 degrees in 5.5 s.

This module ports lane A's cure so the card carries it beside every body-frame
row, on the same capture, the same 500 ms transient, the same tick and the same
step-line bin (phasemeasures.step_line on the same net drive), so the twin is
comparable row for row:

  fixed_heading_fore_aft   the fore-aft tip coordinate projected on the window's
                           MEAN heading (yaw from the captured quaternion,
                           unwrapped; cos and sin of its mean), so the body's own
                           rotation is not folded into the coordinate
  stance fractions         per leg, 1 minus the mean of regression_walk's
                           hysteresis_swing over clearance = tip height minus its
                           5th percentile, the battery's own construction
  readable                 per pair: coherence at the line at or above the run's
                           own line_floor AND not both feet planted more than
                           PLANTED_FRAC of the window
  yaw context              yaw_range_deg and yaw_total_change_deg over the window

Reported only: no band, no grade, nothing in title_match_v38.CARD_ROWS. Legs
are in phasemeasures.LEGS order (lf lm lh rf rm rh); pairs are
phasemeasures.PAIRS indices 2 (lf-rf), 7 (lm-rm) and 11 (lh-rh). The phase
values are whole degrees, rounded as phasemeasures rounds the body-frame rows.
"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import regression_walk as RW  # noqa: E402

_spec = importlib.util.spec_from_file_location("phasemeasures", ROOT / "lanes/A/phasemeasures.py")
PM = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(PM)

# Lane A's registered numbers (CD2-FD, registration 6facf4288, 2026-09-07),
# kept as written there and not re-chosen here. Clause 3: a pair with both feet
# planted for more than this fraction of the window cannot be stepping in
# antiphase whatever its phase row reads. Clause 4: a pair whose coherence at
# the line is below the run's own line_floor (phasemeasures.step_line, the
# median pooled within-leg coherence over 1-30 Hz) is unreadable and decides
# nothing; the floor is a per-run quantity, so it is read from the run and the
# gate is "at or above it", with no margin.
PLANTED_FRAC = 0.85

PAIR_IDX = {"lf_rf": 2, "lm_rm": 7, "lh_rh": 11}                 # phasemeasures.PAIRS
PAIR_LEGS = {"lf_rf": (0, 3), "lm_rm": (1, 4), "lh_rh": (2, 5)}  # phasemeasures.LEGS order
assert all(tuple(PM.PAIRS[i]) == PAIR_LEGS[n] for n, i in PAIR_IDX.items())
assert PM.LEGS == ["lf", "lm", "lh", "rf", "rm", "rh"]


def yaw_of(quat):
    """Unwrapped heading (rad) from the captured MuJoCo quaternion (w, x, y, z)."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def fixed_heading_fore_aft(body, yaw, tips):
    """(T,6) fore-aft tip coordinate in a frame at the window's MEAN heading,
    so the body's own rotation is not folded into the coordinate."""
    c, s = np.cos(yaw.mean()), np.sin(yaw.mean())
    d = tips - body[:, None, :]
    return c * d[:, :, 0] + s * d[:, :, 1]


def stance_fraction(tz):
    """(6,) fraction of the window each foot is on the ground, on the battery's
    hysteresis read of clearance = tip height minus its own 5th percentile."""
    clear = tz - np.percentile(tz, 5, axis=0, keepdims=True)
    air = RW.hysteresis_swing(clear)
    return 1.0 - air.mean(0)


def fixed_heading_rows(cap, tick_ms, transient_ms=500.0):
    """Per-run fields for the card, from one capture. Same transient and tick
    as phasemeasures.from_capture; the step line is located on the same net
    drive with the same function, so `line_hz` equals the body-frame rows'
    `phase_line_hz` when rounded to two decimals (the caller checks it)."""
    trans = int(round(transient_ms / tick_ms))
    c2 = cap[trans:]
    body = np.array([c[0] for c in c2], dtype=float)
    quat = np.array([c[1] for c in c2], dtype=float)
    tips = np.array([c[10] for c in c2], dtype=float)[:, PM.CAPTURE_TO_LEGS, :]
    tz = np.array([np.asarray(c[5]) for c in c2], dtype=float)[:, PM.CAPTURE_TO_LEGS]
    net = np.array([np.asarray(c[7]) - np.asarray(c[6]) for c in c2], dtype=float)
    fs = 1000.0 / tick_ms
    k, hz, coh, floor = PM.step_line(net, fs)
    yaw = yaw_of(quat)
    ph, co = PM.phase_pattern(fixed_heading_fore_aft(body, yaw, tips), fs, k)
    st = stance_fraction(tz)
    out = {"line_hz": float(hz), "line_coh": float(coh), "line_floor": float(floor)}
    for name, i in PAIR_IDX.items():
        a, b = PAIR_LEGS[name]
        both_planted = bool(st[a] > PLANTED_FRAC and st[b] > PLANTED_FRAC)
        out[f"phase_{name}_fixed_deg"] = float(abs(round(float(np.degrees(ph[i])), 0)))
        out[f"phase_{name}_fixed_coh"] = round(float(co[i]), 4)
        out[f"phase_{name}_readable"] = bool(co[i] >= floor and not both_planted)
    for j, leg in enumerate(PM.LEGS):
        out[f"stance_frac_{leg}"] = round(float(st[j]), 4)
    out["yaw_range_deg"] = round(float(np.degrees(yaw.max() - yaw.min())), 2)
    out["yaw_total_change_deg"] = round(float(np.degrees(yaw[-1] - yaw[0])), 2)
    return out
