"""Behavioural metrics computed from a body-part trajectory.

DESIGN RULE: every metric here must be computable from BOTH a real fly recording
and a MuJoCo simulation, using the same code path. Anything that can only be
computed on one side is not a metric, it is a bug waiting to happen.

INPUT CONTRACT
    pts : float array, shape (n_frames, 32, 2)
          Body-part positions in the fly's own frame. Low y anterior,
          high y posterior, x left-right, midline at x ~= 68.
    fps : float, sampling rate.

Each metric returns a scalar or NaN. NaN means "not computable for this
sequence", never a silently substituted zero.
"""
import numpy as np
from scipy.signal import hilbert, detrend, welch
from . import layout as L


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _leg_tip_axis(pts, leg):
    """Leg tip position projected on the body axis (y), detrended.

    Why y and not distance-from-body: during walking a leg tip sweeps
    anterior-posterior (protraction/retraction). That sweep is the step cycle.
    Detrending removes any residual slow drift left over from the alignment
    step so the oscillation is centred on zero.
    """
    tip = pts[:, L.LEG_TIPS[leg], 1]
    if not np.isfinite(tip).all():
        return None
    return detrend(tip)


def _instantaneous_phase(sig):
    """Analytic-signal phase of a roughly oscillatory 1D signal."""
    return np.angle(hilbert(sig))


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def step_frequency(pts, fps=L.FPS):
    """Dominant leg-tip oscillation frequency, Hz, median over the six legs.

    This is the single most robust locomotion number available from an
    egocentric recording: it needs no world frame, no calibration, no arena.
    Real Drosophila step at roughly 5-20 Hz depending on walking speed.
    """
    freqs = []
    for leg in L.LEGS:
        sig = _leg_tip_axis(pts, leg)
        if sig is None:
            continue
        # Welch rather than a bare FFT: 234 frames is short and noisy.
        f, p = welch(sig, fs=fps, nperseg=min(128, len(sig)))
        band = (f >= 1.0) & (f <= 30.0)      # exclude DC drift and noise floor
        if not band.any():
            continue
        freqs.append(f[band][np.argmax(p[band])])
    return float(np.median(freqs)) if freqs else np.nan


def leg_excursion(pts):
    """Median anterior-posterior sweep of the leg tips, pixels.

    How far a leg actually swings. Distinguishes a fly taking real steps from
    one twitching in place, which is exactly the distinction that separates
    score 1 from score 2 on the grading ladder.
    """
    exc = []
    for leg in L.LEGS:
        tip = pts[:, L.LEG_TIPS[leg], 1]
        if np.isfinite(tip).all():
            exc.append(np.percentile(tip, 95) - np.percentile(tip, 5))
    return float(np.median(exc)) if exc else np.nan


def tripod_contrast(pts):
    """Tripod coordination as a PLV contrast: within-tripod minus across-tripod.

    SUPERSEDES the earlier `tripod_index` (mean of cos(phase diff - ideal)),
    which returned ~0.5 (chance) on every real fly and was therefore measuring
    nothing. Diagnosis on 2026-08-05: leg-tip signals in this dataset have a
    spectral peak concentration of 0.185, well below the ~0.3 that Hilbert
    phase needs, so per-sequence phase is too noisy for an absolute index.

    The contrast form survives that noise because both terms are corrupted
    equally. Measured on the most-active third of real flies: within-tripod
    pairs PLV ~0.276, across-tripod ~0.230, contrast ~ +0.046. Small, but the
    four highest-PLV pairs of fifteen were all within-tripod, so the structure
    is real and merely weak.

    POPULATION-LEVEL ONLY. Do not read a single sequence's value as a gait
    measurement; pool over many sequences and compare group medians.

    Ideal tripod gait: the three legs of one tripod move in phase with each
    other and in antiphase with the other three, so within-tripod PLV should
    exceed across-tripod PLV. Positive contrast = tripod-like. Zero = none.
    """
    phases = {}
    for leg in L.LEGS:
        sig = _leg_tip_axis(pts, leg)
        if sig is None:
            return np.nan
        phases[leg] = _instantaneous_phase(sig)

    within, across = [], []
    legs = list(L.LEGS)
    for i, a in enumerate(legs):
        for b in legs[i + 1:]:
            plv = abs(np.mean(np.exp(1j * (phases[a] - phases[b]))))
            same = (a in L.TRIPOD_A) == (b in L.TRIPOD_A)
            (within if same else across).append(plv)
    return float(np.mean(within) - np.mean(across))


def gait_signal_quality(pts, fps=L.FPS):
    """Spectral peak concentration of the leg-tip signals, median over legs.

    A GUARD, not a behaviour. It says whether the gait metrics above are
    entitled to be believed for this recording. Real flies in the Zenodo
    arena data sit at ~0.185, which is why step_frequency and tripod_contrast
    are flagged unreliable there. A clean tethered-walking recording, or a
    simulation sampled at 1 kHz, should score much higher.
    """
    from scipy.signal import welch as _welch
    vals = []
    for leg in L.LEGS:
        sig = _leg_tip_axis(pts, leg)
        if sig is None:
            continue
        f, p = _welch(sig, fs=fps, nperseg=min(128, len(sig)))
        band = (f >= 1.0) & (f <= 30.0)
        if band.any() and p[band].sum() > 0:
            vals.append(p[band].max() / p[band].sum())
    return float(np.median(vals)) if vals else np.nan


def left_right_asymmetry(pts):
    """|left excursion - right excursion| / mean excursion.

    A fly walking straight is symmetric. Sustained asymmetry means turning, or
    a limp. Near zero for healthy straight walking; large for a model whose
    two sides are driven differently, which is a common failure mode when a
    connectome is wired up with a sign error on one side.
    """
    def side(names):
        v = []
        for leg in names:
            tip = pts[:, L.LEG_TIPS[leg], 1]
            if np.isfinite(tip).all():
                v.append(np.percentile(tip, 95) - np.percentile(tip, 5))
        return np.mean(v) if v else np.nan

    l, r = side(['L1', 'L2', 'L3']), side(['R1', 'R2', 'R3'])
    if not np.isfinite(l) or not np.isfinite(r) or (l + r) == 0:
        return np.nan
    return float(abs(l - r) / ((l + r) / 2))


def body_length(pts):
    """Thorax-to-abdomen distance, pixels. A scale and posture check.

    Mostly a sanity metric: it should be near-constant within a real fly and
    across real flies. If a simulated fly's body length wanders, the tracking
    or the coordinate conversion is broken, not the brain.
    """
    d = np.linalg.norm(pts[:, L.ABDOMEN] - pts[:, L.THORAX], axis=1)
    return float(np.median(d)) if np.isfinite(d).all() else np.nan


def posture_variability(pts):
    """SD of the thorax-abdomen angle over the sequence, radians.

    How much the body axis bends. Low for steady walking, high for grooming,
    turning, or a model flailing.
    """
    v = pts[:, L.ABDOMEN] - pts[:, L.THORAX]
    ang = np.arctan2(v[:, 0], v[:, 1])
    if not np.isfinite(ang).all():
        return np.nan
    return float(np.std(np.unwrap(ang)))


METRICS = {
    'step_frequency_hz': step_frequency,
    'leg_excursion_px': leg_excursion,
    'tripod_contrast': tripod_contrast,
    'gait_signal_quality': gait_signal_quality,
    'lr_asymmetry': left_right_asymmetry,
    'body_length_px': body_length,
    'posture_variability_rad': posture_variability,
}


def score_sequence(pts, fps=L.FPS):
    """Run every metric on one (n_frames, 32, 2) trajectory."""
    out = {}
    for name, fn in METRICS.items():
        try:
            out[name] = fn(pts, fps) if name in ('step_frequency_hz', 'gait_signal_quality') else fn(pts)
        except Exception:
            out[name] = np.nan
    return out
