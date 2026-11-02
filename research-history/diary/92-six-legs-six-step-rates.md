---
id: 92
date: 2026-09-12
title: Six legs, six step rates, and the lever that will not pull them together
tldr: Lane C read the six legs' step rates one leg at a time instead of from one spectral line, and they are not the same rate. On the reigning champion's own card the left hind lifts 11 times in five seconds while the right front lifts 46, a spread of 4.08 inside a single run, and the three left-right phase rows are all read at one frequency that four of the six legs are not stepping at. The lever aimed at it, four times the knee drive denominator, stood 24 of 24 against the champion's 22 and never reached the count: the held-out row it would be judged on is open on the champion itself, at 64.2865 degrees against a 64.3 floor. wm1 keeps the title on its first defence, and a sentence in its coronation record is corrected by appending.
era: standard
budget_exception: Fourth entry of the day because entries 89 to 91 were already down when these verdicts landed, and neither arc can be merged backward into a coronation entry it postdates: 08:58 establishes a reporting rule that binds every lane from now on, and 09:50 both refuses a challenger and corrects a claim in the public champion record made at 08:24.
commit: the commit this entry lands in, on main; verdicts of record CONSTRAINTS.md 08:40 at 2380abd04, 08:58 at 96960d274, 09:50 at d092ba8b3 and 10:21 at 36faf1f36, the title verdict itself the dry run at 7cb84131f, identical to the live run
seeds: the per-leg reading spends none. Lane C read 28 committed standing captures on the wk1 and wl1 bases; the judge's confirmation is over the three champions' committed cards, judge seeds 30556-30567 for wm1 and wl1 and the wk1 card of 2026-09-11. The title match ran both arms on 30532-30555 with the card on 30556-30567; lane C's dose table used seeds 50369-50392 with zero fresh seeds. Deterministic Metal engine, dylib c40c5e1a, joint_int exact, spike_reg 44, three workers
theta: no theta changed. The champion is results-wm1-theta.json; the refused challenger is lanes/judge/2026-09-12-c-fti4-wm1-theta.json (file sha256 5644007f, canonical sha16 7258a08815eb8f1b), wm1 with body_joint_drive_scale 0.010 on the six *_FTi_pitch DOFs against the champion's global drive_scale 0.0025, the theta diff exactly ["body_joint_drive_scale"]
n: 28 lane captures plus 12, 10 and 10 standing judge card runs for the per-leg rates; 24 paired battery runs per arm plus 12 card runs per arm for the match; 24 seeds per dose over four doses; 46 standing runs for the outlier-leg reads
wall: zero machine time on the per-leg reading, the dose table and the outlier-leg reads, all from receipts already on disk; the title match ran 09:42 to 09:50, nomination to verdict fifteen minutes
reproduce: for the per-leg rates, take the runs with standing_v32 true in each of lanes/judge/2026-09-12-card-c-adhhalf-wl1-metal.json, lanes/judge/2026-09-12-card-c-t4exempt-wk1-metal.json and lanes/judge/2026-09-11-card-c-fti125-metal.json and read runs[].lifts, six counts per run over the 500-5500 ms window, divided by five for hertz; for the verdict, lanes/judge/2026-09-12-title-v311-c-fti4-wm1-verdict-REFUSAL.json against lanes/judge/2026-09-12-title-v36-c-fti4-wm1-battery{1,2}.json and the two cards
media_note: Two readings of committed receipts, one refused challenger and one closed dose family. Nothing here claims the fly got better, and under the 2026-09-04 footage rule a negative carries its engine identity, its command and a committed reader instead of a clip; all three are above. The runs the six rates were read from are the champions' own published clips.
---

08:58 EEST. **The six legs of this fly do not step at the same rate, and they have not
since wk1.** Lane C's C-N407 (packet 0833, zero seed, registered before the read) counted
each leg's swing onsets separately on 28 committed standing captures and got 5.03, 2.78,
6.43, 8.16, 7.14 and 6.20 Hz for the left front, left hind, left middle, right front, right
hind and right middle **inside one run** — a factor of 2.88 between the fastest and the
slowest leg. The card's three left-right phase rows, meanwhile, are read at one line
frequency, 7.81 Hz on those runs, which four of the six legs are not stepping at.

**Confirmed on the judge's own instrument, with no seed spent.** Every card receipt since
wk1 carries per-leg lift counts over the 500-5500 ms window, so the check costs a file
read. On the reigning champion wm1, 12 of 12 standing, the median lifts are 29, 11, 42.5,
46, 43 and 34, which is 5.8 / 2.2 / 8.5 / 9.2 / 8.6 / 6.8 Hz, and the per-run
fastest-over-slowest ratio has a median of 4.08 with a range of 2.88 to 7.14. It is not a
property of that body:

    champion   standing   lf    lh    lm    rf    rh    rm     max/min   under hysteresis
    wm1          12      5.8   2.2   8.5   9.2   8.6   6.8      4.08           3.74
    wl1          10      5.7   2.9   7.9   9.8   8.6   6.7      3.40           2.93
    wk1          10      6.0   2.5   8.4   9.7   8.9   6.4      3.96           3.67

All rates in hertz, medians over the standing runs of each champion's own card. On the same
runs the card's pooled foot-tip step line sits at 6.5 to 6.8 Hz and the vertical bob at
8.05 to 8.09 Hz.

## Why this matters more than a spread

An animal's six legs step at one frequency. Whatever the gait — tripod, tetrapod, the slow
wave a fly uses when it is barely moving — each leg takes one step per stride, so the
periods match and only the phases differ. This body's left hind lifts eleven times in five
seconds while its right front lifts forty-six, and its left front runs at two thirds of its
right front's rate.

**A phase between two feet that do not share a period is not a quantity.** That is the part
that reaches the title. Three of the rows the conflict count reads,
`gait.phase_lf_rf_deg` and its two siblings, are computed from one spectral bin, and they
spread 13 to 36 degrees between the holder's own judge blocks, which is why they are
already unreadable for the count under v3.10. Lane C's onset-based reader and the card's
line reader disagree by 34 to 50 degrees on the same runs: 77 against 18 degrees for the
front pair, 61 against 29 for the middle, 64 against 27 for the hind. Two readers, one set
of runs, no agreement.

So Robin's rhythm ask has a target in front of the one it names. **Six legs, one period,
and then alternation.** Alternation is a statement about phase, and phase is only defined
once the periods match.

**What is not ruled:** which reader is right. The card's reader keeps v3.11's identity and
the count is frozen on this finding — no row moved, no verdict moved, no champion was
re-graded. Both reads are reported from here. The v3.12 queue takes two items: a
shared-period registry row, whose band needs a source in the animal before it can be
written, and an onset-based reader for the pair rows.

From the same packet, C-N406 is accepted as the null it is: the feet do not leave the
ground together either. Two to four feet are down for 72 per cent of the window, mean 3.99,
and the spread in how many are grounded does not track the path tax (rho -0.10, permutation
p 0.60).

## The lever aimed at it, refused before the count

09:50 EEST. Lane C's C-N409 is wm1 with one key added: `body_joint_drive_scale` at 0.010 on
the six knee (`*_FTi_pitch`) degrees of freedom, four times the champion's global
`drive_scale` of 0.0025. Both batteries passed, 12 of 12 and 12 of 12 standing, qualifying
12 and 10. The card stood 12 of 12. On the bar's own quantities it would have stood 24 of
24 against wm1's 22 and qualified 22 against 21 — the best-standing challenger of the day.

The verdict is **SUCCESSOR BATTERY REFUSAL**, and the reason is not about the challenger.
Adding a key the champion does not carry opens a new parameter block, and a new block is
gated on a held-out set of rows nobody selected on. One of those rows, `rom.hind_fti_deg`,
the hind femur-tibia working range, **is open on the champion itself**: wm1's own card of
record reads 64.2865 degrees against a 64.3 floor, in band 5 of 12. A row the holder fails
cannot constrain a challenger's new block. The challenger is worse on it anyway, 57.4167
degrees and 0 of 12, so either reading refuses. Its front and middle knee ranges shrink
too, 97.44 to 83.72 and 18.80 to 14.54, both still in band, which is what a larger drive
denominator does to a knee's excursion.

Had the count been reached it would not have crowned either: the judge's reader puts the
challenger at 9 open rows against the holder's 8, newly opening
`posture.footfall_separation` at 1.1458 against a 1.02 to 1.11 band, with
`gait.contra_phase` open only on the holder. Recorded as wm1's first defence.

And it does not buy what it was registered to buy. C-N410, registered at 08:44 before any
trace of C-N409 existed, asked whether this lever pulls the six rates together. **No.** The
fastest-over-slowest ratio moves +0.228 paired, sign p 0.83, and the pairs do not move
toward alternation, p 1.0. What the dose does buy is a lower hop on 22 of 22 paired seeds
(-0.061) and a less irregular left hind (interval CV -0.14, p 0.0009). The lever that buys
path length and the bob does not buy rhythm.

## The sentence that was wrong, and how it is fixed

wm1's coronation record, written at 08:24 this morning, says "the four joint working ranges
and stride length are in band on both arms". That was wrong by **0.0135 degrees on one
row**. Halving the adhesion moved the hind femur-tibia range from wl1's 66.28 (9 of 10 in
band) to 64.29 (5 of 12), across the floor, and nobody read it because the held-out gate
does not apply to a change of a legacy scalar.

It is corrected in `CHAMPION.json` by **appending a dated correction, not by rewriting the
sentence**. The original claim stays where it was made, at the standard and the hour it was
made.

The consequence is structural and it is the expensive part: every block contract carries the
standard held-out set, so **under v3.11 no new or changed parameter block can be admitted on
the wm1 base until the champion closes that row or the held-out set is revised at v3.12.**
This was the second refusal on that row in one day; b-dnnotcmd-wl1 opened it at 06:54, from
66.28 to 62.07.

## The dose family, closed by the card at every dose

10:21 EEST. Lane C had already put a dose ladder under the same lever, 24 seeds per dose,
no fresh seeds spent:

    dose (drive units)      0.0025 (wm1)   0.005    0.006    0.010
    path over 5 s, mm            41.09     36.16    35.65    32.05
    standing of 24                  22        19       24       24
    qualifying                      21        18       21       22
    footfall separation, BL     1.0890    1.1038   1.1126   1.1218
      (ceiling 1.11; in band)    22/22     15/17    11/24     1/24
    hind FTi range, deg          65.57     63.59    62.28    56.11
      (floor 64.3)

The card cost is monotone in the dose — footfall separation, the hind and front knee ranges,
path, hop and straightness all order with it — while the count is not (22, 19, 24, 24) and
sits inside the block-to-block spread. The separation ceiling is crossed between 2x and
2.4x.

**The mechanism is one number seen at two joints** (C-N412, zero seed): a larger denominator
shrinks the commanded knee sweep, the leg straightens, the foot lands further out (front
footprint +0.0166 body lengths on 22 of 22 paired seeds, lateral +0.0188 on 21 of 22), and
the path shortens. The gain and the cost are the same effect read at two places, and no dose
in the family separates them. Lane C reads the knee stance sweep as staying under the 4.05
degree floor at the left front and left hind at every dose, with the left front at 5.19 /
2.31 / 3.10 / 2.31 degrees across base, 2x, 2.4x and 4x — not monotone — and the two middle
legs under 1 degree throughout.

**Accepted as a closed family.** No dose of this key on the knee DOFs is a nominee on wm1:
every dose moves `rom.hind_fti_deg` the wrong way, 0 of 22 in band at 4x, and the gate
refuses the block while that row is open on the holder.

## The holder straddles its own floor

Lane C stated this against its own dose table, which is the part worth recording. The
judge's 12-seed card of record reads wm1 at 64.2865 on that row, 5 of 12 in band and a
conflict by the card's grade. Lane C's 22-seed block reads 65.5696, 15 of 22 in band. **The
two blocks straddle the 64.3 floor by 1.28 degrees**: exactly the edge-row case v3.10 was
written for.

The ruling: the 09:50 refusal stands on any block, because the challenger's arm reads
57.4167 and 0 of 12 and the gate requires both arms. But the structural consequence is
qualified — before the next block nominee is gated, the holder's read on that row is taken
on **two** judge blocks, the card of record plus a second twelve-seed judge card, pooled. If
the pooled read is in band, the row constrains the challenger's arm only, as the gate
intends. Lane C's block is evidence for the read, not the read itself: the gate runs on
judge receipts. A second block needs a fresh-seed licence, which does not exist today, so
the read waits for Robin.

## The left hind leg is a different leg

From the same traces, zero seed, 46 standing runs (C-N413, C-N414). The left hind lifts 25
fewer times than the right hind (0 of 22 seeds go the other way), is planted 0.93 of the
time (+0.57 against its mirror), carries +0.185 of the vertical force share, and is
commanded 4.80 degrees less knee and 4.55 degrees less coxa-trochanter peak-to-peak — every
one at sign p below 1.2e-04, and unchanged at four times the dose.

On wm1 the command saturates (|tanh| above 0.99) on 20 to 71 per cent of ticks at every leg
and every joint, and **the left hind's saturation is one-sided**: the knee sits at its
positive stop on 44 per cent of ticks and its negative stop on 7, where the right hind reads
27 and 28; the coxa-trochanter reads 41 and 19 against the mirror's 20 and 37. Accepted as
measured facts. That the lever is the cord's flexor-extensor balance at that leg is lane C's
reading and not a ruling, and lane C stated its own limit: across the six legs the bias does
not order the step rate.

## The sibling ruling: two rows stay two rows

08:40 EEST, eighteen minutes before the per-leg reading. Lane C asked for
`gait.straightness_5p5s` and `body.height_oscillation` to be counted as one fact, so that
closing them would be worth one conflict instead of two. The case: a partial correlation of
-0.642 between the hop and straightness at matched travel (C-N396), and the whole
straightness deficit sitting in body motion above 3 Hz (C-N403) in the band where the body's
height power sits at 5 to 9 Hz (C-N402).

**Refused.** The only grouped rows in the counter are the four left-right phase rows, and
they are grouped because they are the same measurement of one event. That is identity, not
association, and lane C met exactly that bar once before: C-N396 proved straightness is
*identically* net travel over path length on 353 rows, which is why that reading amended the
judge's. A partial correlation of -0.642 is 41 per cent shared variance on one body, and a
shared frequency band is a shared cause at most.

And lane C's own intervention table dissociates them, on C-N308's six seeds (straightness
band 0.50 to 0.98, higher better; hop at most 0.10, lower better):

    arm         straightness    hop      reading
    baseline       0.2650      0.613
    ctr12          0.2193      0.539     hop better, straightness worse
    fti12          0.3240      0.516     both better
    fti24          0.3126      0.452     both better
    both12         0.3211      0.378     both better

Three arms co-moving is what a shared cause looks like. One arm moving them in opposite
directions is what two rows look like. The judge's own decisive block adds nothing either
way: halving the adhesion moved straightness 0.2365 to 0.2493 and the hop 0.6487 to 0.6508,
both inside the noise. What would carry the grouping later is written down so it can be
bought: an instrument identity of C-N396's kind, a derivation showing one row is a function
of the other, or twelve paired seeds where one arm moves both rows toward band with 9 of 12
agreeing on each and no arm in the same family moves them oppositely.

## Problems / what bit us

The per-leg reading was available for free the whole time. Lift counts have been in every
card receipt since the hysteresis fields landed on 2026-09-03, and three champions were
crowned while the pooled line frequency and the three phase rows were reported and nobody
divided the six counts by five. The instrument was not missing; the question was.

C-N414's first read printed zero saturation everywhere, because
`_resolve_body_joint_drive_scale` returns None for a theta without the per-DOF key and the
baseline arm has no such key. It was caught by the two arms disagreeing on joints neither
arm touched, fixed, and no published number moved. A reader that returns a clean zero for
the arm that lacks a key is the shape of bug that gets believed.

The one to sit with is the gate. A challenger that stood 24 of 24 was stopped by a row the
champion fails, and the champion failed it because a change three coronations old crossed a
floor by 0.0135 degrees in a read nobody was required to take. As written that closes every
block route on this base until the row is settled. What the gate should do when the holder
itself fails a held-out row is with Robin, with the default that v3.11 stands as written.

## What changed in the model

Nothing in the fly. No parameter moved, no band moved, no verdict moved, and wm1 keeps the
title with this recorded as its first defence. What changed is the reporting contract:
**from now on every walking arm reports the six per-leg step rates from lifts, and their
fastest-over-slowest ratio, beside the three pair rows and the hop.** A single pooled cadence
number hid a factor of four for three champions, and a row nobody reads is a row that
drifts. Beside it: one corrected sentence in the public champion file, one closed dose
family, one qualified structural consequence, and four measured facts about a leg that
barely steps.

## Attribution

Lane C proposed, registered and ran every reading here (C-N406, C-N407, C-N409 to C-N415,
and the grouping request in packet 0810 with its own supporting and contradicting evidence),
including the packet at 09:54 that argued against its own nomination four minutes after
making it, the 10:02 extrapolation it corrected itself at 10:14, and the straddle
observation it raised against its own dose table. The judge accepted C-N407 and confirmed it
independently on the three champions' cards, accepted C-N406 and C-N410 as nulls, refused
the grouping on lane C's own C-N404 table, ran the title match, recorded the refusal,
corrected the champion record by appending, and qualified the structural consequence.
Robin's standing 16:35 rhythm ask is what the per-leg reading serves; the prior target it
implies, six legs at one period before alternation, is the judge's reading of his ask and
not his words. Robin stopped the fleet at 10:02 — "stop all ongoing research after you're
done; I will let you know when it's time to continue and how" — and lane C pushed its last
packet and closed at 10:18:55.
