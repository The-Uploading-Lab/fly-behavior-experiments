"""One rhythm detector, with a null attached, used by everything.

WHY THIS EXISTS. The rhythm measurement in this repo has now been wrong FIVE
separate ways, each time producing a confident number. That track record is
the single most useful thing to know about this file: on this problem, a
positive result is much more likely to be an artefact than a finding, and the
guards below exist because each one already fired in anger.

  1. Welch on the raw trace scored a STARTUP TRANSIENT as a 2.93 Hz rhythm
     (z = 166.5). Fixed by settling and detrending; the same point then
     scored z = -0.28.
  2. Mean inter-spike interval reported 46 Hz where the rhythm was 23 Hz,
     because the cell bursts and the mean ISI measures the within-burst rate.
  3. Autocorrelation with a wide search band reported 2.1-2.5 Hz on trains of
     ~40 spikes. Those lags sit at the far edge of the band where only a
     handful of samples overlap, so the estimate is noise and the maximum of
     noise is always somewhere.
  4. Taking the largest autocorrelation VALUE in the band reported whatever
     the band's edge was -- 52.63 Hz in a 4-60 band, 111.11 in 4-120, 25.64 in
     2-30 -- because a monotonically decaying autocorrelation has no interior
     maximum at all. It had already produced "22 of 84 joint channels rhythmic
     at 55.56 Hz", the same 55.56 Hz at three different synaptic delays. Fixed
     by scoring PROMINENCE and rejecting boundary maxima.
  5. A channel whose null draws mostly found no peak divided by a standard
     deviation of ~1e-8 and reported z = +77,161. Fixed by requiring at least
     half the null draws to be estimable AND the null sd to exceed 1e-4 --
     an absolute floor, because prominences are O(0.01-0.5) and a merely
     nonzero floor let the same value through a second time.

The common failure is a statistic with no null. A peak height means nothing on
its own -- what means something is the peak height RELATIVE TO what the same
spike train would produce with the rhythm removed and everything else kept.

THE NULL. Shuffle the inter-spike intervals. That preserves the number of
spikes, the mean rate, the ISI distribution, burstiness, and the refractory
period exactly, and destroys only the ORDER, which is where periodicity lives.
A cell that fires in bursts scores high on burstiness either way; only a cell
whose bursts are evenly SPACED beats its own shuffle.

AND A SECOND CHECK, which would have caught two of the five for free: a real
rhythm has a frequency, an artefact has whatever frequency the search window
implies. So every detection is re-measured in a wider and a narrower band and
rejected if the answer moves. Octave-tolerant, because a harmonic is the same
rhythm seen at 2x and a wider window legitimately lets one win on prominence.

Calibration on synthetic trains, after all five fixes:
    9 of 10 true bursters (7-30 Hz, 2 and 5 ms jitter) detected
    4 of 4  non-rhythms rejected -- Poisson, bursty-but-irregularly-spaced,
            dense monotonic decay, and a regular train below the band
"""
import numpy as np

BAND = (4.0, 60.0)      # Hz. Flies walk at 7-15; this brackets it generously.
MIN_SPIKES = 20         # below this an autocorrelation is not estimable
N_NULL = 200


def _bin(sp, nb, bin_ms):
    return np.bincount(np.clip((sp / bin_ms).astype(int), 0, nb - 1),
                       minlength=nb).astype(float)


def _acorr_rows(X):
    """Autocorrelation of every row of X, by FFT, normalised to lag 0.

    np.correlate does this by direct convolution, which is O(n^2): on a 4,500
    sample train that is ~2e7 operations per call, and the null needs 200
    calls per cell. Measured: the direct version put a 31-point sweep on
    course for roughly half an hour of pure autocorrelation. This is
    O(n log n) and batched over the whole null at once.
    """
    n = X.shape[1]
    nfft = 1 << int(2 * n - 1).bit_length()
    X = X - X.mean(axis=1, keepdims=True)
    F = np.fft.rfft(X, nfft, axis=1)
    ac = np.fft.irfft(F * np.conj(F), nfft, axis=1)[:, :n]
    return ac / (ac[:, :1] + 1e-12)


def _acorr(sp, dur_ms, bin_ms=1.0):
    nb = int(dur_ms / bin_ms)
    return _acorr_rows(_bin(sp, nb, bin_ms)[None, :])[0]


def _peak(ac, dur_ms, band, bin_ms=1.0):
    """Frequency and PROMINENCE of the best periodic peak, or (nan, nan).

    ⚠️ FOURTH TIME THIS STATISTIC HAS BEEN WRONG. It used to return the
    largest value of the autocorrelation inside the search band. If the
    autocorrelation is monotonically DECAYING -- which is what dense,
    short-timescale-correlated activity gives, and there is no rhythm in it at
    all -- then the largest value in any range is at that range's left edge,
    so the function returned a "frequency" set by the search band rather than
    by the data. Caught 2026-08-09 by moving the band and watching the answer
    follow it:

        band 4-60 Hz  -> 52.63 Hz        band 2-40 Hz -> 37.04 Hz
        band 4-120 Hz -> 111.11 Hz       band 2-30 Hz -> 25.64 Hz
        band 4-200 Hz -> 166.67 Hz

    The answer was always just inside the upper edge. It had already produced
    "22 of 84 joint channels rhythmic at 55.56 Hz" in the balanced regime,
    with the same 55.56 Hz at three different synaptic delays -- a rhythm that
    does not move when you change the thing that sets rhythm frequency.

    THE FIX. Score PROMINENCE: how far a local maximum rises above the highest
    trough separating it from anything taller. A decaying slope has zero
    prominence everywhere, so it can no longer score at all, whereas a genuine
    periodic bump keeps its prominence regardless of the baseline it sits on.
    Boundary maxima are rejected outright.
    """
    from scipy.signal import find_peaks
    lo = max(2, int(round(1000.0 / band[1] / bin_ms)))
    hi = min(len(ac) - 1, int(round(1000.0 / band[0] / bin_ms)))
    # Never search past a fifth of the record: beyond that the estimate is
    # built from too few overlapping samples to be an estimate.
    hi = min(hi, int(0.2 * dur_ms / bin_ms))
    if hi - lo < 5:
        return np.nan, np.nan
    seg = ac[lo:hi]
    idx, props = find_peaks(seg, prominence=0.0)
    if not len(idx):
        return np.nan, np.nan
    # interior only -- a "peak" in the first or last bin of the window is the
    # window's edge, not the signal's
    keep = (idx > 1) & (idx < len(seg) - 2)
    if not keep.any():
        return np.nan, np.nan
    idx, prom = idx[keep], props["prominences"][keep]
    k = int(idx[int(np.argmax(prom))]) + lo
    best_prom = float(prom.max())
    # OCTAVE CORRECTION. A jittered rhythm can have a taller peak at twice its
    # period than at its period -- jitter accumulates less, relatively, over
    # two cycles. Measured: a 14 Hz train with 8 ms of jitter was read as
    # 7.19 Hz. So if half (or a third of) the winning lag also carries a peak
    # worth most of the winner's prominence, prefer the shorter lag: the
    # fundamental, not its harmonic.
    for div in (2, 3):
        j = (k - lo) // div
        if j < 2:
            continue
        tol = max(1, j // 6)
        near = idx[(idx >= j - tol) & (idx <= j + tol)]
        if len(near):
            pn = prom[(idx >= j - tol) & (idx <= j + tol)]
            if pn.max() > 0.6 * best_prom:
                k = int(near[int(np.argmax(pn))]) + lo
                best_prom = float(pn.max())
                break
    return 1000.0 / (k * bin_ms), best_prom


def detect(sp, dur_ms, band=BAND, n_null=N_NULL, seed=0, bin_ms=1.0):
    """Is this spike train periodic, and at what frequency?

    sp : spike times in ms, already relative to the start of the analysed
         window (settle time removed by the caller).

    Returns a dict:
      n, rate_hz         how much data there is
      freq_hz            frequency of the strongest autocorrelation peak
      peak               its height, 0-1
      null_mean, null_sd what ISI-shuffled versions of the SAME train give
      z                  (peak - null_mean) / null_sd  <- the number that counts
      ring               peaks also present at 2x, 3x, 4x the lag (0-3)
      ok                 enough spikes to have measured anything
    """
    n = len(sp)
    rate = n / (dur_ms / 1000.0)
    out = {"n": n, "rate_hz": rate, "freq_hz": np.nan, "peak": np.nan,
           "null_mean": np.nan, "null_sd": np.nan, "z": np.nan, "ring": 0,
           "band_drift": np.nan, "null_valid_frac": np.nan, "ok": False}
    if n < MIN_SPIKES:
        return out
    ac = _acorr(sp, dur_ms, bin_ms)
    f, pk = _peak(ac, dur_ms, band, bin_ms)
    if not np.isfinite(pk):
        return out
    out.update(freq_hz=f, peak=pk, ok=True)

    # null: same ISIs, shuffled order -- all n_null draws in one batch
    rng = np.random.default_rng(seed)
    d = np.diff(sp)
    nb = int(dur_ms / bin_ms)
    shuf = rng.permuted(np.tile(d, (n_null, 1)), axis=1)
    times = sp[0] + np.concatenate(
        [np.zeros((n_null, 1)), np.cumsum(shuf, axis=1)], axis=1)
    X = np.stack([_bin(t, nb, bin_ms) for t in times])
    ACN = _acorr_rows(X)
    peaks = np.array([_peak(a, dur_ms, band, bin_ms)[1] for a in ACN])
    # DEGENERATE NULL GUARD. If most null draws find no peak at all, nanstd is
    # computed from a handful of values -- or one -- and the z-score explodes.
    # Measured 2026-08-09: a channel with 45% of its nulls returning NaN
    # produced z = +77,161. A null that cannot be estimated is not a null.
    valid = np.isfinite(peaks)
    out["null_valid_frac"] = float(valid.mean())
    if valid.sum() < max(10, 0.5 * n_null) or np.nanstd(peaks) <= 1e-4:
        out["ok"] = False
        return out
    out["null_mean"] = float(np.nanmean(peaks))
    out["null_sd"] = float(np.nanstd(peaks))
    out["z"] = float((pk - out["null_mean"]) / (out["null_sd"] + 1e-12))

    # BAND STABILITY. A real rhythm has a frequency; an artefact has whatever
    # frequency the search window implies. Two of the five ways this statistic
    # has been wrong would have been caught for free by re-measuring in a
    # different band and checking the answer does not move. So it is now part
    # of the measurement rather than something to remember to do afterwards.
    # Octave-tolerant: a harmonic is the SAME rhythm seen at 2x or 1/2x, not a
    # different answer, and a wider window legitimately lets a harmonic win on
    # prominence. Only a frequency that is unrelated to the original by any
    # small integer ratio counts as drift. Without this the check rejected
    # genuine 14 Hz and 20 Hz bursters, which is worse than the disease.
    alt = [(band[0], band[1] * 1.8), (max(1.0, band[0] * 0.5), band[1] * 0.7)]
    drifts = []
    for b in alt:
        fa, _ = _peak(ac, dur_ms, b, bin_ms)
        if not np.isfinite(fa):
            continue
        ratios = [fa / f * m for m in (1.0, 2.0, 3.0, 0.5, 1 / 3.0)]
        drifts.append(min(abs(r - 1.0) for r in ratios))
    if drifts:
        out["band_drift"] = float(max(drifts))
        if out["band_drift"] > 0.25:
            out["ok"] = False              # frequency follows the window
    else:
        out["band_drift"] = float("nan")
        out["ok"] = False

    # Ringing: an evenly spaced COMB of peaks, not one bump. Counted only
    # against the null -- the first version asked whether the harmonic peak
    # exceeded 30% of the main peak, and a Poisson train with a main peak of
    # 0.047 passes that trivially. Measured: it scored ring 3 on pure Poisson,
    # which makes the statistic worthless on its own. The bar is now the same
    # bar the main peak has to clear.
    floor = out["null_mean"] + 2.0 * out["null_sd"]
    k = int(round(1000.0 / f / bin_ms))
    tol = max(1, k // 5)
    for mth in (2, 3, 4):
        j = mth * k
        if j + tol >= len(ac):
            break
        if ac[j - tol:j + tol + 1].max() > floor:
            out["ring"] += 1
    return out


def summary(d):
    if not d["ok"]:
        return f"{d['n']:4d}sp (too few)"
    return (f"{d['freq_hz']:6.2f}Hz  peak {d['peak']:.3f}  "
            f"z {d['z']:+6.2f}  ring {d['ring']}  ({d['n']} spikes)")
