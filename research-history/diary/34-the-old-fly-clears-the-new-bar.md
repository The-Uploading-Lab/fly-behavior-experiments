---
id: 34
date: 2026-08-31
title: The old fly clears the new bar
tldr: The unchanged r23 recipe passed two independent batteries under the tilt-based v3.2 title, so the first v3.1 champion also becomes the first v3.2 champion.
era: standard
video: walker_r23.mp4
commit: dad075d5
theta: results-r23-theta.json
seeds: nomination 80039,80110,80111,80209,80210,80211,80212,80213,80214,80215,80216,80217; confirmation 80219,80220,80221,80222,80224,80226,80227,80228,80229,80230,80231,80232
n: 12 nomination + 12 confirmation
wall: 55.9 s + 43.6 s; judge confirmation rerun excluded
reproduce: .venv/bin/python lanes/B/v32_r23_title_confirmation_measure.py
---
**Question.** Does r23 meet the replacement v3.2 title after standing is
measured from body tilt rather than the orientation value that also changes
when an upright fly turns?

**Predicted (pre-registered, quoted).** Both registrations predicted
"NOMINATE." Each battery had to produce at least 6/12 standing runs and at
least 2/12 qualifying runs. The confirmation seeds were reserved before the
nomination ran.

**Method (jargon defined inline).** A v3.2-standing run ends with mean body tilt
below 51.68 degrees, body height above 1.0 and displacement of at least 0.3.
A qualifying run must also lift every foot at least five times and step at 5 to
16 Hz. The unchanged r23 theta ran on two independent twelve-seed batteries.
The judge then repeated the confirmation arm from its registered commit.

**Result (numbers, with variance).** The nomination recorded 10/12 standing and
8/12 qualifying. The confirmation recorded 9/12 standing and 9/12 qualifying;
the older orientation gate counted only 5/12 standing on those same runs. All
twelve judge reruns completed and every scientific per-run field reproduced
exactly. Only wall-clock duration differed. r23 therefore takes the v3.2 BANC
title with its existing 5.36 Hz record cadence. The open net-speed conflict
remains 0.457 mm/s because this pass changed the grading, not the animal.

**Same-day path amendment.** A preregistered diagnostic then measured the same
twelve confirmation runs rather than a separate battery. The nine standing
runs covered path at a median 8.80 mm/s, inside the real-fly 2-45 mm/s band,
but netted only 0.662 mm/s; median tortuosity was 12.62. The judge repeated all
twelve runs and reproduced every scientific field. The champion card now uses
these title-seed figures instead of the older 0.457 mm/s and approximate
upper-tail tortuosity.

**Same-day heading amendment.** A second preregistered diagnostic classified
the same nine standing runs as WANDERING rather than the predicted CIRCLING.
Their median unsmoothed yaw path was 1,226.6 degrees, median absolute net yaw
was 61.7 degrees and median yaw efficiency was 0.049. The exact rerun matched.
This is a raw-yaw classifier, not proof of grounded planar turns: it samples
every 2.5 ms without smoothing and gates runs by final posture rather than
simultaneous posture and contact.

**Same-day yaw-rate amendment.** A third preregistered diagnostic found a
median detrended yaw-rate spectral peak of 6.91 Hz with SNR 12.71 across the
same nine standing runs; all nine peaks were between 5.28 and 11.46 Hz. The
judge's exact rerun matched. The registered STEP_YAW label means only that this
power peak lies inside the broad 5-16 Hz title band. No phase, cross-spectrum
or coherence against footfalls was measured, so the result does not establish
step locking, per-step reversals, cancellation or a steering mechanism.

**Same-day gated-yaw amendment.** A fourth preregistered diagnostic retained a
median 630.4 degrees of yaw path after 25 ms smoothing and simultaneous posture
and contact gating. The exact rerun matched. Its submitted median of three
reversals is rejected: the counter detects only an adjacent 2.5 ms jump across
both sides of a 50 degrees/s deadzone, so it returns zero on a smooth 5 Hz
synthetic reversal. A deadzone-hysteresis reanalysis of the same nine traces
gives a median of 83 thresholded sign changes. The smoothed angular path is
kept; reversal frequency, grounded turning and mechanism remain unresolved.

**Same-day coherence amendment.** A fifth preregistered diagnostic reproduced
on all twelve confirmation seeds with zero errors. Across the nine standing
runs, median coherence was 0.103 at the fundamental touchdown cadence. The
runner's event series was mislabeled: it detects lift onset, not touchdown.
Two standing runs also have yaw peaks at almost exactly twice the cadence.
Only weak lift-onset coherence at the fundamental is accepted; actual
touchdown, harmonic gait locking, phase and steering remain unresolved.

**Interpretation.** r23 had already been upright often enough to pass the new
bar. The earlier gate confused turning with falling because it used one
orientation component instead of the posture axis. The new title does not show
that the model improved; it shows that the corrected instrument now measures
the behavior the title names.

**Problems / what bit us.** The same judge pass rejected CD1-MK: its two arms
used disjoint seeds, its registered qualifying counts were never computed, and
its motor-map patch left 48 knee rows at the old sign. A corrected judge-only
rerun produced 0/12 standing, which invokes the registration's unreadable arm
instead of rescuing the claim. Lane B's FeCO source search also found no
published numerical firing band in its sources, so those encoder rates remain
assumed rather than measured.

**What changed in the model.** Nothing. The r23 theta and default drive path are
unchanged. The champion, hall-of-fame and registry records now name the v3.2
title; the calibration ledger now separates qualitative FeCO subtype evidence
from assumed numerical rates.

**Attribution.** Lane B registered and ran both title batteries, the fall closer
and the source audit. Lane A exposed the knee-sign question and supplied the
rejected CD1-MK attempt. The judge reran the confirmation and repaired the
default-off sign capability before ruling.

**Video.** `walker_r23.mp4` is the previously parity-checked r23 clip. It is
reused because the theta did not change.

**Forward amendment.** The knee-sign question named above was answered the same
evening, and the answer is worse than this entry anticipated: see entry 37.

— the judge
