---
id: 55
date: 2026-09-05
title: The eye worked and the legs did not hear it
tldr: The first behavioural test of sight on a walking champion. Lane C repaired the coincidence seam that had made the model's motion detector blind to its own inputs, showed the T4 detector fires direction-selectively at the champion's own balance, and showed its left-right signal reaches the HS steering cells through the connectome alone and vanishes with the detector off. The judge pre-registered a grade before any seed ran, amended it before any seed ran when lane E pointed out the optic-lobe exemption was not needed, and ran five arms on twelve fresh seeds. The fly turned the same way whichever direction the grating moved: 4 of 8 counted seeds agreed in sign, the direction-dependent yaw difference (0.78 rad) no larger than the seeds' own divergence without the detector (0.64 rad), the product applied on every sighted run. Sight reaches the steering cells and stops short of the legs. Recorded as a conflict with the optomotor response of real flies, one point bounded, three forks named. On the same morning lane A located the oldest conflict on the card: only the front legs propel wa1, because the map gives the middle and hind coxae no protractor motor neuron, and every relabelling of the cells BANC names swung those legs backwards.
era: standard
commit: the commit this entry lands in
seeds: judge 30208-30219 (five arms, 60 runs) and 30207 (one-seed smoke test); lane C 52564-52572; lane A 202420-202455 and earlier blocks; lane E 101163-101194; lane B 80759-80760; lane D 61141-61142
---

2026-09-05 12:02 EEST. **Champion: wa1**, title held under v3.6. Nothing crowned today.

## The question Robin set

Robin, 10:45: "the grade is a single question, does sight change what a
standing fly does, on twelve seeds, engine-identical, carrying the caveat C
stated before running. Hold it at neural until it clears. Crown nothing on a
body that cannot stand."

## The chain that reached the steering cells (lane C)

The day's neural work rests on a defect lane C found at 08:23: `cns.py` wrote
the coincidence seam's presynaptic indicator before driven spikes were merged
into the fired set, so the multiplication that makes a Reichardt detector had
never once applied a non-zero term. Every earlier "the seam is inert" reading
(C-N29, C-N32) was withdrawn on that. With the write moved (ported to main
under the bit-identity probe), the T4 detector separates the two grating
directions at `bal_target` 0 (C-N43), survives the lag control (C-N44), and,
the part that mattered for the grade, is live at the champion's own balance
of 0.43519 at one gain, 0.3 (C-N45: T4a -0.054 against T4b +0.084, 3 of 3
seeds; gain 1 saturates, gain 0 sits under the rate floor). Lane C's caveat,
stated before running and carried into every record today: T4's detector only
(fast Mi1, delayed Mi9/Mi4/C3, lag 60 ms), T5's is not implemented.

Then C-N48 and C-N49 (11:50): on the stitched build the left-right voltage
asymmetry of the HS cells, the lobula-plate tangential cells that feed the
turn command, flips sign with grating direction on 3 of 3 paired seeds
(+0.099, +0.035, +0.022) and does not flip with the detector off (largest
0.007). Nothing hand-wired past the detector. The DNa02 spike readout was
withdrawn by lane C's own control as unpowered (one spiking cell per side);
the voltage read runs next. Two instrument facts from the same runs: the
column frame carries 6,938 right-lobe against 5,736 left-lobe carriers under
a constant HS asymmetry near -0.7, so the sign of a flip is not yet a turn
direction; and switching the detector on moves the HS baseline from +0.14 to
-0.70, unexplained.

## The grade

Pre-registered at 11:15 (`lanes/judge/2026-09-05-vision-grade-prereg.md`,
instrument `lanes/judge/vision_grade.py`), amended at 11:2x before any seed
ran: lane E's 11:05 packet pointed out that C-N45 had already cleared at the
champion's own balance, so the optic-lobe exemption the first draft carried
would change a second thing against wa1 for no reason the question needed.
The body became wa1's theta of record plus the carrier in lane C's
deliver-only mode (the grating reaches T4/T5 and nothing is written to the
turn command, so the connectome is the only route from eye to cord) plus the
T4 detector at gain 0.3.

Five arms per seed on NumPy (the detector refuses Metal and its 60 ms lag is
above Metal's reproducibility bound): grating +1 and -1 with the detector,
+1 and -1 with the coincidence keys removed, and blind. Twelve judge seeds,
30208 to 30219, unseen by any seat. A seed counted only if it stood, by the
title's tilt clause, in all five arms. The primary: the sign of yaw(+1) minus
yaw(-1) over the five-second stimulus, two-sided exact binomial under 0.05.

Result at 11:59. Standing: +1 12 of 12, -1 9 of 12, product-off +1 12 of 12,
product-off -1 11 of 12, blind 12 of 12; eight seeds counted. The sign agreed
on 4 of 8 (p = 1.0). The control without the detector agreed on 6 of 8
(p = 0.29). Median |D| 0.78 rad with the detector against 0.64 rad without
(one-sided p = 0.47). Lateral speed contrasts null both ways. Blind against
sighted: forward speed 1.90 against 1.99 and 1.88 mm/s, cadence 4.86 against
5.14 and 5.00 Hz. The coincidence receipt shows the product applied on all
twelve sighted seeds (1.65e8 each). Every run, blind or sighted, turned about
-2 rad in five seconds, the champion's straightness conflict, which the paired
contrast removes.

Reading, by the outcome table written before the runs: the detector fires and
its direction signal does not reach the legs within five seconds at this gain.
Any direction-following turn is smaller than the walker's own sensitivity to a
changed input. Recorded as `vision.optomotor_turning`, CONFLICT on the sighted
build at one point (gain 0.3, contrast 0.25, 4 Hz, 5 s, T4 only), FEMALE-ONLY.
Three forks, in cost order: DNa02 in voltage (lane C, running); a lobe-balanced
frame, since the -1 direction fell 3 of 12 where +1 fell none (Fisher 0.22,
recorded not claimed); then the gain ladder in behaviour and a longer trial.
A paired extension on the exemption build runs beside it and grades nothing.

Run metadata: theta `lanes/judge/theta-wa1-sight-nominee.json` (sha256_16
37d387d30f8358e5, 69 keys), 60 runs at 65 to 120 s each, about 20 minutes per
arm with three then two workers; arms 1 to 3 ran walk_search
995a924b7c17828e and arms 4 to 5 the engine after a default-off port, probe
BIT-IDENTICAL; verdict `lanes/judge/2026-09-05-vision-grade-verdict.json`.

## The oldest conflict, located (lane A)

Speed on wa1 is 1.6 mm/s against real flies' 10 to 20. Lane A read the strokes
on 45 standing runs across three seed blocks: the front legs run forward in the
air and backward on the ground on 45 of 45; the middle legs barely stroke; the
left hind leg swings backwards through the air on 45 of 45 and is dragged
forward on the ground; the right hind is inert. Two of six legs propel with a
0.1 mm step. The cause is static and in the map: the joint that swings a
middle or hind leg fore-aft is ThC_pitch, and `motormap.build_map` gives it
live motor neurons lf 6, rf 6, lm 1, lh 1, rh 1, rm 0, because BANC and the
male annotate the promotor on the front leg only and the eight protractor-side
rows per middle and hind leg sit at gain 0. The map's own comment named the
hazard when it was written; no search ever turned the rows on. Four
registrations then failed to give those legs a forward stroke: grip (ZC), both
rotators live (ZD), their sign flipped (ZE), the anterior rotator alone (ZH),
every one swinging the legs backwards in the air. Lane A withdrew its own
speed lead at 11:38 as survivorship and stopped the line. The T2/T3 coxal
protractor is MISSING in both datasets as labelled; the male's 12 to 14
unlabelled motor neurons per middle and hind leg are the only source of a
label.

## Also today

Lane E measured the campaniform stand-in replaceable by the load signal at a
factor of 3.2365, neutral over 24 paired seeds; nominated `balance_by_type`
as a default-off search dial (ported, nine tests green); read the motor pool
recruitment-ordered during walking on both champions. Lane D returned from
silence with a two-seed scout that dropped its own route on the posture gate.
Lane B: DNa01 at champion rank adds travel on one wa1 seed of two; DNg100 on
four of four. Three sensory corrections measured neutral on their own seeds
(load-fed campaniform, exact-type FeCO modality, the 25-cell front-claw mirror
that corrects BANC's one mirror-reversed typing) were bundled into one
challenger, wb1, nominated with a frozen hash and matched on the holder's 24
judge seeds this afternoon.

## Attribution

Robin: the goal and its three clauses (10:45); the rulings this grade ran
under (footage only for improvement claims; the engine rule; hold at neural).
Lane C: the seam defect, the neural chain from C-N43 to C-N49, the caveats.
Lane E: the 11:05 catch that the exemption was not needed, which changed the
body under grade before a seed ran. Lane A: the stroke read and the map
finding, and its own withdrawal. The judge: the pre-registration and its
amendment, the instrument, the five-arm design, the verdict, and the ports
under the probe (deliver-only, the column frame, yaw fields, balance_by_type,
FeCO exact-type, reflex_map).
