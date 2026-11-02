---
id: 94
date: 2026-09-13
title: The animal's own instrument, and the leg that swings without lifting
tldr: Lane B built one step-onset instrument and ran it on both the tracked real flies and the champion. Measured that way the champion's six legs are almost as even as the animal's, 1.55 fastest-over-slowest against 1.44, which is inside the animal's own 5th to 95th percentile and unrecognisable as yesterday's spread of 4.08. The same instrument makes the other half far worse: real left and right legs of a segment alternate at 176 to 179 degrees, the champion's sit at 19.5, 3.0 and -26.5, and 1 of 24 leg-pair runs reaches even 120. The left hind does swing, 6.5 times a second, and only 15 per cent of those swings lift the foot clear of the floor. Nine gait rows nothing counted were read on the champion for the first time, and at 22:04 Robin ruled four of them into the count at v3.12.
era: standard
commit: the commit this entry lands in, on main; lane B's E1 registered at 3702ca2a8 with results at f2876284d, graded CONSTRAINTS.md 16:51 at fcdca10cd, which also carries the gauges; C-N417 16:33 at 2ed79baba; C-N416 16:09 at df81b8fa0; C-N418, C-N419 and C-N420 17:46 at 784e0b806; Robin's ruling filed at 053d6d64b
theta: no theta changed. Every arm is the champion results-wm1-theta.json, sha256_16 3a21ddb794be23cb, with one key altered per arm: lane B through presyn_gain_by_id, lane C through the hind knee force gain and through body_joint_tau_act_ms
seeds: lane B's three arms on eight fresh lane B seeds 81698-81705; lane C's C-N417 on two spent seeds 50370 and 50371 and C-N419 on the spent pair at 50370; the gauges and the animal side spend none, read from the champion's committed twelve-seed card on judge seeds 30556-30567 and from the Schulz 2024 tracking at Zenodo 11002776. Deterministic Metal engine, joint_int exact, spike_reg 44
n: 24 runs for lane B (8 per arm), 12 standing card runs for the nine gauges, 12,419 real-fly sequences with every leg above 4 Hz, 48 leg-runs for the drive correlation, 24 leg-pair runs for the phase read, two seeds per dose for the hind knee force
wall: 930 s for lane B's three arms, 130.68 s for C-N417, zero machine time for the gauges, the animal side and the three archive reads
reproduce: the gauges from lanes/judge/gauge_readers.py over lanes/judge/2026-09-12-card-c-adhhalf-wl1-metal.json, written to lanes/judge/2026-09-13-gauges-wm1.json and .md; the row writes at lanes/judge/2026-09-13-register-b1641-and-gauges.py; lane B's arms and both-sides instrument in its own packet at f2876284d; Robin's ruling verbatim at exchange/2026-09-13-2210-robin-ruling-v312-gait-gauges-count.md
media_note: A null arm, one instrument read on two sides, nine gauges over runs already on disk and three routes closed by their own results. Nothing here claims the fly got better, so under the 2026-09-04 footage rule the engine identity, the exact commands and the committed readers stand in for a clip; the runs the gauges were read from are the champion's own published card.
---

16:51 EEST. **Yesterday the six legs of this fly stepped at rates that differed by a
factor of four. Today, measured the way the real flies were measured, they differ by a
factor of 1.55, and the real flies differ by 1.44.** Entry 92 reported the spread from a
tip-clearance instrument, which counts a step when a foot rises above 0.15 mm. The
tracking data that the whole benchmark rests on has no foot height in it at all, so that
instrument has no animal side, and a number with no animal side cannot be a conflict with
the animal.

Lane B built the missing half. A step onset is read as the start of an anterior swing,
which both the tracked flies and the simulated one produce, and the same code runs on
both. On 12,419 real sequences in which every leg steps at least 4 Hz, the fastest leg
over the slowest is **1.44 (5th to 95th percentile 1.19 to 1.83)**, and 1.36 at a 5 Hz
cut. The champion reads **1.55** on that instrument, inside the animal's own interval.
The same runs read 4.05 on the clearance instrument, and the two instruments rank the
individual legs almost independently of each other, Spearman 0.08 to 0.18 over 48
leg-runs.

So the two instruments are not two readings of one quantity, and only one of them can be
compared with a fly. The registry row keeps its ordering axis open as a candidate
conflict and gains the animal side on the swing instrument.

## The half that got worse is the half that matters more

The same instrument reads left against right. In the animal the two legs of a segment
alternate: circular mean phases of **176.2, 179.1 and 178.0 degrees** for front, middle
and hind, which is as close to exact antiphase as a real animal gets. The champion reads
**19.5, 3.0 and -26.5 degrees**, which is the two legs of a pair swinging together, and
**1 of 24 leg-pair runs reaches even 120 degrees**.

This row has been a conflict on every champion for a week, but it had been read on one
pooled statistic. Now it is read pair by pair on both sides with one instrument, and
there is no reading of it that looks like a fly.

## The left hind does swing. It does not lift.

The leg that carried yesterday's spread turns out not to be a slow leg. It makes **6.5
anterior swings a second, and 15 per cent of them clear 0.15 mm**; its tip peaks at
**0.052 mm**, about a third of the bar the clearance instrument uses. The clearance
instrument was right that the foot is not getting off the floor and wrong about why:
the timing is there and the height is not.

Lane B read the drive during those swings and found it reversed against the leg's own
job. In Hertz summed over each pool during the left hind's swing phase, against the right
hind's: trochanter extensors **186 against 76**, trochanter flexors **84 against 192**,
knee extensor **29 against 88**, tarsus depressor with the long tendon **281 against
177**. The fraction of swings that lift tracks the knee extensor's swing-phase rate,
Spearman **+0.75 over 48 leg-runs**.

Three of lane C's reads point the same way and rule out the obvious suspect, the motor
map. The knee extensor pool's weighted rate is **5.07 Hz on the left hind against 10.07
on the right with equal coefficient mass** (0.1354 against 0.1292), so the left leg's
deficit is which cells get recruited, not how strongly the map wires them. The
trochanter drive during the left hind's own anterior swing is **downward over upward by
2.38 and 1.99** on two saved runs where the right hind's is 0.40 and 0.50, with map
weights identical on both sides (0.475 down, 0.525 up). And the left hind's fast
extensor is not late: it **rises from 95 to 110 Hz before the lift and falls to 9 to 18
Hz after it**.

## Three levers aimed at it, all closed by their own results

**Force.** Raising the left hind fast extensor's force gain from 0.1 to 1.0 raises that
leg's lift rate from 2.73 to 4.91 Hz and from 2.18 to 3.64 Hz on the two seeds, raises
speed from 1.127 to 1.269 and 1.312 to 1.725 mm/s, and halves the six-leg spread (2.93 to
1.48, 3.42 to 2.25). It also pushes footfall separation out of band (1.108 to 1.185 and
1.104 to 1.175 body lengths against 1.02 to 1.11) and worsens the left hind knee's stance
direction. Route closed by lane C on its own numbers, with all 16 archived channels
reproduced exactly on the off arms.

**Timing.** Slowing the left hind knee's activation to 6 ms lengthens travel from 6.20 to
10.78 mm and raises straightness from 0.121 to 0.231, and drops that leg from 15 lifts to
4, under the bar's floor of 5. Footfall separation goes out, left-right asymmetry goes
out at 0.0007 against a band of 0.008 to 0.323, slip rises. One spent pair, no nominee.

**Anatomy.** Lane B's own registered hypothesis was that the spread comes from the
connectome's left-right annotation asymmetry, which is real: by outgoing synaptic mass
the claw afferents run 1.23 and 1.58 left over right, the campaniform sensilla 1.51, the
hair plates 0.57 to 0.60. Equalising it is **a null**. Of the three registered
predictions, the left hind's lift rate rose on 4 of 6 comparisons against a registered 5
of 6 (median +0.2 Hz), the spread narrowed on 4 of 7 against a registered 6 of 7 (median
-0.47), and only the ordering prediction held, Spearman 0.90 to 1.00 on 7 of 7. Standing
was 8 of 8, 7 of 8 and 6 of 8 across the three arms, speed 1.60, 1.55 and 1.68 mm/s, the
body's bounce line 8.05 to 8.09 Hz, and every per-leg motor pool sum moved by under 3 per
cent. The asymmetry is in the annotations and the six rates are not its shape.

## Nine rows read on the champion for the first time

Under the ruling in entry 93, nine gait rows that no instrument had ever computed were
read off the champion's committed twelve-seed card, reported and counted by nothing. What
they say about the reigning fly:

- it crabs, **19.0 degrees of slip angle**, every one of the twelve runs to the same side,
  9 of 12 outside the animal's -13.2 to 13.8 band;
- its stride is short, **0.381 body lengths against 0.55 to 0.75**;
- its tripod contrast is **0.013**, on the nearest field the card carries;
- its per-leg lift-rate ratio is **3.74** with a 5.45 Hz spread, on the instrument with no
  animal side;
- its cord rhythm is **8.05 Hz**, inside 5 to 15, and the only one of the nine that looks
  like a fly;
- unsigned yaw rate **39.6 degrees per second** sits inside 3.7 to 229.7, on a proxy that
  samples every 0.24 s rather than every 33 ms and therefore reads low;
- forward speed 1.92 mm/s, lateral 0.69, straightness 0.249, yaw range 86 degrees, all
  descriptive because those rows have no band written yet;
- speed against turning correlates at Spearman 0.14 over 12 runs, also unbanded;
- and leg tip side fidelity is **NOT READ**, because the per-frame sign needs the tip
  traces and the card does not carry them.

## Robin's ruling, at 22:04

The judge's report asked whether these four of the nine should count. Robin answered
verbatim:

> yes:
> 1. v3.12: may the four gait gauges enter the count? Default: gauges only,
>    reported not counted. My recommendation: yes.

So **heading slip, stride length, tripod contrast and left-right phase become counted
rows at v3.12**. As of this entry's commit that ruling is filed in `exchange/` and is not
yet written into CONSTRAINTS.md or into the bar, which is the judge's to do.

Two consequences are on the record before it lands, filed with the ruling. The first is
arithmetic: those are four rows the reigning champion fails, moving from uncounted to
counted in one version bump, and a rule already in force refuses a challenger on any
held-out row the champion itself fails. That rule already froze the fleet on one such row
on 2026-09-12. The second is a reporting duty: seven open conflicts has been the headline
figure in every report for two days, it was computed by a bar that could not see
crabbing, and the first report after v3.12 has to say what the number becomes and that
the change is in the bar rather than in the fly.

## Problems / what bit us

The clearance instrument's 4.08 was never wrong about what it measured, and it was
reported in entry 92 as a property of the fly's stepping. It is a property of its foot
height. That is the same failure the project has now hit at three sizes in ten days, an
instrument read against a band the instrument cannot produce, and the cheap defence is
the one lane B used here: build the reader on the animal's data first and only then point
it at the model.

Six of the nine gauges are proxies and say so in their own output. The yaw rate smooths
fast turns by construction, the slip angle uses window means rather than the row's 500 ms
displacement-weighted windows, and the tripod row reads flyscore's contrast rather than
the index the band was written for. They are good enough to report and are not yet good
enough to grade, which is exactly the distinction the v3.12 question turns on.

## What changed in the model

Nothing. No parameter moved and no title changed hands. Lane B's equalisation is a null,
lane C closed three routes by their own results, and the champion remains wm1 at
sha256_16 3a21ddb794be23cb. What moved is the ledger: the per-leg spread row gains an
animal side on the swing instrument, the contralateral phase row gains its reading on one
instrument across both sides, and nine rows carry a first measured value on the champion.

## Attribution

**Robin** ruled the four gait rows into the count at 22:04, in the words quoted above,
against the stated default of reporting them uncounted. **Mistral** filed the ruling and,
unprompted, recorded the deadlock it couples to and the two questions it leaves the judge,
which is why they are in this entry rather than discovered at a refusal. **Lane B**
registered its predictions before running, reported its own null against them, and built
the one instrument that runs on both the animal and the model. **Lane C** supplied the
three archive reads that rule out the motor map and closed two of its own routes on the
evidence. **The judge** graded the packets, wrote the gauge readers and the row writes.
