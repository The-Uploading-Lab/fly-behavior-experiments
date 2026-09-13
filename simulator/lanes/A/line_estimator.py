"""The canonical lane A line estimator, named so the judge can merge one function.

Judge, 06:38: six files under lanes/A compute something heave-peak shaped, and
guessing which is the wrong division of work. This is the one. Every lane A
reader that reports a line frequency imports `line_hz` from here from now on;
the earlier copies stay where they are so no committed receipt changes, and new
work uses this.

WHAT IT IS. The dominant frequency of a signal in a band, estimated as the argmax
of a Hanning-windowed periodogram zero-padded by four. Zero padding interpolates
the peak; it adds no information and is there so the estimate is not quantised to
the record's own bin, which is 0.18 Hz at a 2.5 ms tick over the scoring window
and 0.045 Hz after padding.

WHAT IT IS NOT. It is not a coherence estimator and it does not pick a line out
of a multi-line spectrum: on the summed motor drive, whose spectrum carries the
fundamental and both a subharmonic and a harmonic at comparable power, the argmax
jumps between them (CD2-BS). Use it on body height and on the summed foot load,
where the champion's spectrum is single-peaked; for the drive, use power share at
a named frequency instead.

VALIDATION. CD2-BR moved the transient cut to ticks 200, 300, 400 and 600, halved
the record, and zero-padded, and the champion's estimate stayed within 0.087 Hz
of 5.818; the halves, whose bins are twice as wide, agreed exactly. CD2-BM found
the estimate identical across six bodies whose tip cadences span 0.91 Hz.

CALLER'S CONTRACT. Pass STANDING runs only. A fallen body's height trace is
dominated by the fall and its peak is meaningless; the first version of CD2-BS
did not filter and that defect is what produced its one anomalous number.

    from line_estimator import line_hz
    hz = line_hz(capture["body"][200:, 2])          # tick 2.5 ms, default band
"""
import numpy as np

TICK_MS = 2.5
TRANSIENT_TICKS = 200          # the 500 ms the battery discards
BAND_HZ = (1.0, 30.0)
PAD = 4


def line_hz(x, tick_ms=TICK_MS, band=BAND_HZ, pad=PAD):
    """Dominant frequency of a 1-D signal, Hz. See the module docstring."""
    x = np.asarray(x, dtype=float)
    if x.size < 64:
        raise ValueError(f"line_hz needs at least 64 samples, got {x.size}")
    x = x - x.mean()
    fs = 1000.0 / float(tick_ms)
    n = x.size * int(pad)
    f = np.fft.rfftfreq(n, 1.0 / fs)
    p = np.abs(np.fft.rfft(x * np.hanning(x.size), n=n)) ** 2
    m = (f >= band[0]) & (f <= band[1])
    if not m.any():
        raise ValueError(f"no FFT bin inside {band}")
    return float(f[m][int(np.argmax(p[m]))])


def heave_line_hz(capture_body, transient_ticks=TRANSIENT_TICKS, **kw):
    """Convenience for the usual case: body height after the transient.

    capture_body: the capture's element 0 as an array, (T, 3). Standing runs only.
    """
    z = np.asarray(capture_body, dtype=float)[transient_ticks:, 2]
    return line_hz(z, **kw)
