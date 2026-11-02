---
id: 49
date: 2026-09-03
title: The floor is one leg
tldr: Four challengers took the crowned fly to the title instrument in five hours and all four lost on the same clause, a standing body stepping a fraction of a hertz under the 5 Hz floor. Lane D split that loss per leg and found it is not a slow fly, it is one slow leg. The right hind leg steps under 5 Hz on every one of the champion's standing title runs, 22 of 22, and on 23 of 23 of tonight's challenger; the right front leg never does, at 7 to 16 Hz. The battery's cadence is the median of six legs, so one leg pinned at 2 Hz pulls the whole fly under the bar. Lane E's -50 mV resting potential on eighteen premotor cells travels further and falls less than the champion and lost 17 to 19 anyway, because that leg did not move. Lane A closed the tonic motor-drive route on a dose curve with no dose response. The scorer's own 3,000 real flies hold the per-leg measurement that decides whether a five-fold spread between legs is a conflict; nobody kept it.
era: standard
commit: f30e6086c (state the batteries ran on); the match record, the registry row and this entry land in the judge commit that follows
seeds: judge 30001-30024 paired with ng1's; D 60479-60486 (Metal); A 200873-200875; E 100138-100149 (lane block)
n: two title batteries, one paired comparison of 24; D six seeds by one arm plus three prior; A five arms by three seeds; E three arms by twelve seeds
wall: 963.8 s + 952.0 s battery wall (NumPy, in parallel, about 80 s per seed); A 396.9 s; D about 2 min compute; 02:03-02:20
reproduce: PYTHONPATH=. .venv/bin/python lanes/judge/title_match_v33.py --theta lanes/E/theta-ng1-13b-vrest-minus50-r154.json --seeds 30001-30012 --battery 1 --out receipt.json
video: walker_ng1_body.mp4
video_url: https://media.theuploadinglab.ai/walker_ng1_body.mp4
---

**Question.** Three questions, one answer. Lane E: does resting the
eighteen 13B premotor cells at -50 mV, which won its own twelve seeds
10 to 8 against the champion and 10 to 8 against a label-permuted twin,
beat the champion on the champion's seeds? Lane D: when a challenger
loses on cadence, which legs are under the floor? Lane A: does tonic
drive into the slow motor neurons, the one dial that had raised the
uncommanded fly's cadence on three seeds, do it as a function of dose?

**Predicted (pre-registered).** Lane E's rule, written 01:52 with the
theta's hash: promote only if the untouched holder-seed match wins, a
tie keeps ng1. Lane D registered DIFFUSE against a two-leg hypothesis
(left front with right hind as the two lowest on most seeds). Lane A's
clause A required the 256 Hz median cadence at least 1.0 Hz above the
0 Hz median with 0, 64 and 256 non-decreasing.

**Method.** The match followed the ordering clause: two twelve-seed
batteries on ng1's 24 judge seeds, the whole female CNS (188,508
neurons) on NumPy, afferent spike regularity k = 44, the engine hashed
before and after (9b870eb98ad85e44, unchanged), higher paired
qualifying count takes the title. A run qualifies when its final-second
tilt is under 51.68 degrees, it lifts every foot at least five times,
its timing is its own, and the tip cadence (the median over six feet of
tip-swing onsets in 5.5 s) sits in 5 to 16 Hz. Lane D ran the commanded
champion on six fresh seeds on the laptop's GPU and read the battery's
per-leg step frequency; the judge then read the same question off the
per-foot onset counts the title batteries already keep, for ng1's 24
judge runs and the challenger's 24, with no new simulation. Lane A ran
five tonic-drive settings (0, 32, 64, 128, 256 Hz) on the uncommanded
body on three fresh seeds. One seed of each lane's result was replayed
by the judge on main before it went on a row: every field exact.

**Result.** The match: ng1 19, the -50 mV body 17. The challenger
stood on 23 of 24 (ng1 22), including both seeds where ng1 falls, and
its median travel was 4.50 mm against 3.90. It took three seeds ng1
loses (30018, 30020, 30023) and lost five ng1 wins: one fall (30010)
and four standing bodies at 4.36, 4.45, 4.55 and 4.73 Hz. Lane D: the
right hind leg under 5 Hz on 6 of 6 fresh seeds at 0.91 to 3.27 Hz,
the right front and right middle on 0 of 6, the three left legs on 2
to 4 of 6. Replayed seed 60480: lf 3.27, lh 7.09, lm 4.00, rf 12.36,
rh 0.91, rm 5.09 Hz, median 4.55. The judge's cross-check on the title
rows: RH under 5 lifts per second on 22 of 22 of ng1's standing runs
(0.4 to 4.7, the lowest foot on 18), LF on 17, LH 7, LM 6, RM 6, RF on
none (6.9 to 16.2, median about 10); on the challenger's 23 standing
runs, RH on 23. Lane A: median cadence across the five doses 2.91,
4.91, 4.36, 5.09, 3.68 Hz; median travel 3.24, 2.74, 2.59, 1.48,
2.82 mm; at 64 Hz the change against each seed's own control was
+1.27, -0.54 and +1.45 Hz; no setting had three seeds standing,
travelling and in band.

**Interpretation.** The median of six numbers is the mean of the third-
and fourth-lowest, and the right hind leg sits below that pair on every
standing run of both arms of tonight's match. So a dial that lifts that
one leg past the fourth-lowest foot promotes the fourth and fifth feet
into the median, which on D's replayed seed moves 4.55 to 5.05 and
across the floor; a dial that raises every leg spends most of itself on
the four already in band and on the right front leg, which is already
twice too fast. That is why four different challengers, built from
membership, delivery form and resting potential, all lost by the same
fraction of a hertz: none of them touched the leg that sets the median.
Whether it is a conflict with the real animal depends on a number
nobody has computed. In real tripod walking every leg steps once per
stride, so a five-fold spread between the right front and right hind
legs of a standing, travelling body is the shape of a conflict; the
scorer's reference set of about 3,000 real flies holds the per-leg
frequencies (its step-frequency metric computes six per-leg peaks and
keeps only their median), and that band is what promotes or closes the
row. It is also what the fixers would aim at. Lane A's result closes a
route rather than opening one: eight-fold more tonic drive into the slow
motor pool buys no cadence, and the earlier three-seed lift at 64 Hz
does not reproduce on three more seeds, so the read that it was a lever
is withdrawn where it was written. Two instrument questions were
settled by the same numbers. The tip-cadence estimator's grain (one
onset on a deciding foot moves the median 0.091 Hz) decides a minority
of the losses: of 28 losing runs across the three earlier defences, 23
were standing bodies under the floor, 3 of them one onset under, 3 two
under, 17 from 3 to 12 under. The bar does not move; the onset distance
now prints for every losing seed.

**Problems.** Both per-leg readouts are tip-based step detectors, D's
at a 0.15 mm clearance and the title's on swing onsets, so a hind-right
joint range or a contact geometry would produce the same numbers as a
motor-pool deficit; the measurement that separates them (the RH motor
pool's spike rate against the RH tip trace on the same seeds) was named
by lane D and has not been run. The leg finding is one animal's; the
male has not been asked. Lane E's lane-block advantage (10 to 8 to 8 at
n = 12) was a real count that did not carry to the paired 24, where the
dial won 3 seeds and lost 5; twelve seeds resolve a two-seed difference
about as often as they invent one. Lane A's dose curve is three seeds
per setting. ng1's own title receipts were taken on an earlier engine
hash (d13ab3f0beb00852); every engine change since has been default-off
and probe bit-identical, and the challenger ran on today's file.

**What changed in the model.** Nothing in the champion. Registry:
fourth defence entry; new `gait.per_leg_cadence_spread`,
CONFLICT-CANDIDATE, FEMALE-ONLY, with the DLC per-leg band as its
promotion condition; new `gap.per_leg_cadence_band` (expires 09-10);
`gap.tip_cadence_floor_resolution` resolved with no bar change; the
third defence's reading carries a dated superseding sentence on the
tonic dial. Ledger: `cns.graded_premotor_v_rest_mv` carries the R154
block and the match, still searched; `cns.mn_tonic_slow_hz` carries the
dose curve and the withdrawal. The challenger theta carries its
`match_result`. Video: none new; ng1's clip stands,
https://media.theuploadinglab.ai/walker_ng1_body.mp4

**Attribution.** Robin's 21:45 instruction (enable the lanes, do not
direct them) shaped the three feedback files. Lane D designed the
per-leg decomposition, registered its reading before the draw, reported
the tally against its own hypothesis and named the separating
measurement. Lane E built the resting-potential prior from Agrawal
2020, ran the permuted twin at equal parameter count, nominated on a
frozen hash and wrote the tie rule against itself. Lane A registered
the dose curve with four clauses, ran it, and withdrew its own earlier
reading in the same document. The judge accepted the nomination, ran
the batteries, replayed one seed of each result, read the leg question
off the title instrument's existing rows, and recorded the defence, the
row, the gap and the resolution.

cites:: lanes/judge/2026-09-03-title-v33-13bvrest-battery1.json, lanes/judge/2026-09-03-title-v33-13bvrest-battery2.json, lanes/E/theta-ng1-13b-vrest-minus50-r154.json, lanes/E/2026-09-03-ng1-13b-vrest-confirmation-r154-result.json, exchange/2026-09-03-0152-laneE-r154-judge-packet.md, lanes/D/2026-09-03-cadence-bottleneck-legs-result.json, exchange/2026-09-03-0156-laneD-judge-packet.md, lanes/A/2026-09-03-cd1oj-ng1-whole.json, lanes/A/2026-09-03-0148-cd1oj-result.md, lanes/judge/2026-09-02-title-v33-ng1-battery1.json, lanes/judge/2026-09-02-title-v33-ng1-battery2.json
uses-data:: CHAMPION.json, CONSTRAINT-REGISTRY.json, CALIBRATION-LEDGER.json
