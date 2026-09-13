"""Inter-leg and within-leg phase pattern at the step line, from a capture.

Lane A, 2026-09-03. CD1-QK/QL found no within-leg joint timing on qg1 by
correlation; the spectral view of the same traces shows a shared line at
5.5-6.2 Hz in the motor drive of all 42 leg DOFs (within-leg coherence 0.41
against a 0.25 floor, between-leg 0.32 against 0.18) with an inter-leg
phase pattern that repeats across seeds (circular consistency 0.6-0.98)
and is not the tripod. These measures read that pattern so a dial screen
can grade "rotates the pattern toward tripod" instead of waiting for
tri_index events.

Inputs: arrays of shape (T, ...) after the transient, at `fs` Hz:
  net drive (T, 42) in motormap DOF order (leg-major: lf lm lh rf rm rh, then
  yaw pitch roll CTr_pitch CTr_roll FTi TiTa), and body-frame fore-aft tip
  position (T, 6) in leg order lf lm lh rf rm rh.

The line is located as the peak of the pooled within-leg drive coherence in
4-9 Hz; phases are cross-spectral phases at that bin. Tripod targets: pairs
inside one tripod (lf-rm, lf-lh, rm-lh, rf-lm, rf-rh, lm-rh) at 0, the nine
cross pairs at 180. `tripod_score` is the mean over the 15 pairs of
cos(phase - target): +1 tripod, -1 anti-tripod, 0 no relation.
"""
import itertools

import numpy as np
from scipy.signal import coherence, csd

LEGS = ["lf", "lm", "lh", "rf", "rm", "rh"]
# The closed-loop capture's per-leg arrays are built over sorted(LEG_KEYS.values())
# and this file labels its columns in motormap's leg-major order, which is the
# same six legs in a different order. from_capture reindexes with this, so a
# column here always means the leg LEGS names (lane A, 2026-09-06; the defect
# lane E found at 15:50 and both consumers compensated for until this commit).
CAPTURE_LEGS = ["lf", "lh", "lm", "rf", "rh", "rm"]     # sorted(LEG_KEYS.values())
CAPTURE_TO_LEGS = [CAPTURE_LEGS.index(l) for l in LEGS]  # [0, 2, 1, 3, 5, 4]
DOF = ["yaw", "pitch", "roll", "CTr_pitch", "CTr_roll", "FTi", "TiTa"]
TRIPOD_A = {"lf", "rm", "lh"}
PAIRS = list(itertools.combinations(range(6), 2))
TARGET = np.array([0.0 if ((LEGS[i] in TRIPOD_A) == (LEGS[j] in TRIPOD_A))
                   else np.pi for i, j in PAIRS])
NPERSEG = 512


def body_frame_fore_aft(body, quat, tips):
    """(T,6) fore-aft tip coordinate in the body frame (CD1-OP construction)."""
    out = np.empty((len(body), 6))
    for t in range(len(body)):
        w, x, y, z = quat[t]
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        out[t] = (R.T @ (tips[t] - body[t]).T)[0]
    return out


def _coh(x, y, fs):
    f, C = coherence(x, y, fs=fs, nperseg=min(NPERSEG, len(x)))
    return f, C


def _phase_at(x, y, fs, k):
    f, P = csd(x, y, fs=fs, nperseg=min(NPERSEG, len(x)))
    return float(np.angle(P[k]))


def step_line(net, fs, lo=4.0, hi=9.0):
    """Frequency of the shared drive line: peak of pooled within-leg
    coherence in [lo, hi]; also its height and the 1-30 Hz floor (median)."""
    sd = net.std(0)
    keep = sd > 0.2 * np.median(sd)
    acc = []
    f = None
    for j in range(6):
        idx = [7 * j + k for k in range(7) if keep[7 * j + k]]
        for a, b in itertools.combinations(idx, 2):
            f, C = _coh(net[:, a], net[:, b], fs)
            acc.append(C)
    W = np.mean(acc, 0)
    m = (f >= lo) & (f <= hi)
    k = int(np.flatnonzero(m)[np.argmax(W[m])])
    floor = float(np.median(W[(f >= 1.0) & (f <= 30.0)]))
    return k, float(f[k]), float(W[k]), floor


def phase_pattern(sig6, fs, k):
    """Phases (15,) and coherences (15,) between the six legs' signals at bin k."""
    ph = np.empty(len(PAIRS)); co = np.empty(len(PAIRS))
    for n, (i, j) in enumerate(PAIRS):
        ph[n] = _phase_at(sig6[:, i], sig6[:, j], fs, k)
        f, C = _coh(sig6[:, i], sig6[:, j], fs)
        co[n] = C[k]
    return ph, co


def tripod_score(ph):
    return float(np.mean(np.cos(ph - TARGET)))


def phase_measures(net, fore_b, fs):
    """Dict of readouts. `line_hz`, `line_coh`, `line_floor`; per joint
    (roll, CTr_pitch, FTi) the tripod score and the 15 phases in degrees;
    the tripod score on the feet's body-frame fore-aft position; the
    within-leg CTr_pitch-FTi drive phase and coherence per leg at the line."""
    k, hz, coh, floor = step_line(net, fs)
    out = {"line_hz": round(hz, 2), "line_coh": round(coh, 3),
           "line_floor": round(floor, 3)}
    for name in ("roll", "CTr_pitch", "FTi"):
        d = DOF.index(name)
        sig = net[:, [7 * j + d for j in range(6)]]
        ph, co = phase_pattern(sig, fs, k)
        out[f"tripod_{name}"] = round(tripod_score(ph), 3)
        out[f"phase_{name}_deg"] = [round(float(np.degrees(p)), 0) for p in ph]
        out[f"coh_{name}"] = round(float(np.median(co)), 3)
    ph, co = phase_pattern(fore_b, fs, k)
    out["tripod_feet"] = round(tripod_score(ph), 3)
    out["phase_feet_deg"] = [round(float(np.degrees(p)), 0) for p in ph]
    out["coh_feet"] = round(float(np.median(co)), 3)
    wl = {}
    for j, leg in enumerate(LEGS):
        c = 7 * j
        f, C = _coh(net[:, c + 3], net[:, c + 5], fs)
        wl[leg] = {"phase_deg": round(float(np.degrees(
            _phase_at(net[:, c + 3], net[:, c + 5], fs, k))), 0),
            "coh": round(float(C[k]), 3)}
    out["within_ctr_fti"] = wl
    return out


def from_capture(cap, tick_ms, transient_ms=500.0):
    trans = int(round(transient_ms / tick_ms))
    c2 = cap[trans:]
    body = np.array([c[0] for c in c2], dtype=float)
    quat = np.array([c[1] for c in c2], dtype=float)
    tips = np.array([c[10] for c in c2], dtype=float)[:, CAPTURE_TO_LEGS, :]
    net = np.array([np.asarray(c[7]) - np.asarray(c[6]) for c in c2], dtype=float)
    return phase_measures(net, body_frame_fore_aft(body, quat, tips), 1000.0 / tick_ms)


def from_npz(path, fs=400.0, trans=200):
    z = np.load(path)
    net = (z["a_ext"] - z["a_flex"])[trans:]
    fb = body_frame_fore_aft(z["body"][trans:], z["quat"][trans:], z["tips"][trans:])
    return phase_measures(net, fb, fs)
