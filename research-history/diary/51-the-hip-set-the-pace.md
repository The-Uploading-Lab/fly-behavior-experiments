---
id: 51
date: 2026-09-03
title: The hip set the pace
tldr: Twenty minutes after nk1 was crowned, lane B's other single-joint body, the champion ng1 with 6 ms muscle activation on the six coxa-trochanter joints instead of the six knees, went onto nk1's own 24 paired judge seeds and stood and qualified on every one of them, 24 against nk1's 19. It takes the title as nc1. Where the knee constant had changed stance and not cadence, the hip constant changes both: nk1's four under-floor seeds on the block step at 5.0 to 7.9 Hz on nc1, and on the 23 seeds where both stood nc1 is faster on 14, median 0.91 Hz. Two title changes in one morning, both to a body that differs from ng1 by one number on six joints; the knee-plus-hip body is untried.
era: standard
commit: c00cda515 (state the batteries were folded on; the batteries ran in a worktree pinned to a9babf17d); the coronation bundle lands in the judge commit that follows, tagged nc1
seeds: judge 30001-30024, paired with nk1's battery 1-2 receipts on the same seeds; B 80528-80539 NumPy for the nomination
n: two twelve-seed batteries, 24 paired seeds; B one NumPy twelve
wall: batteries 872.1 s + 874.9 s (NumPy, in parallel); 09:35-09:50
reproduce: PYTHONPATH=. .venv/bin/python lanes/judge/title_match_v33.py --theta results-nc1-theta.json --seeds 30001-30012 --battery 1 --out receipt.json
video: walker_nc1_body.mp4
video_url: https://media.theuploadinglab.ai/walker_nc1_body.mp4
---

**Question.** Does the same 6 ms activation constant, on the
coxa-trochanter joint instead of the knee, beat the knee body on the
knee body's own seeds? Entry 50 recorded that nk1's knee constant
bought stance and nothing else: equal median cadence with ng1, every
loss a standing body under the 5 Hz floor. Lane B's second single-joint
body puts 6 ms on the six CTr_pitch joints, the joint that lifts and
lowers the whole leg, and leaves the knees and the other 30 joints at
the derived 12.3 ms. On lane seeds 80528-80539 it qualified 12 of 12
against ng1's 7, stood 12 against 10, and put no seed under the floor
where ng1 put three. That met the nomination bar and the judge froze
the theta hash at 09:34.

**Predicted (pre-registered).** The nomination named the ordering
clause as the test: higher qualifying count over nk1's 24 paired seeds
takes the title, a tie goes to the next 24 under item 5. No count was
predicted. Lane B's own confirm on 80540-80551, registered 09:25 with
its accept rule written first (under-floor count strictly below
control, or equal with fewer falls or more qualifies), was still
running when the match finished and lands on the card either way.

**Method.** Two twelve-seed batteries on 30001-30012 and 30013-30024,
NumPy, spike_reg 44, in a worktree pinned to commit a9babf17d
(walk_search 072a3cbb09da81be, cns d7ebac3530d83c5a), paired with nk1's
battery 1 and 2 receipts on the same seeds. A run qualifies when it
stands at the end (tilt under 51.68 degrees, height over 1.0,
displacement at least 0.3 mm), lifts every foot at least five times,
and its tip-measured cadence lands inside 5 to 16 Hz with no commanded
step frequency in the theta. The batteries ran 09:35 to 09:50 while
main took three default-off engine seams under the probe.

**Result.** nc1 24, nk1 19. Both batteries 12 of 12 standing and 12 of
12 qualifying. The five discordant seeds are all nc1's: 30011, where
nk1 fell, and 30002, 30008, 30010, 30018, where nk1 stood under the
floor at 4.27, 4.27, 4.73 and 4.91 Hz. On nc1 those four step at 7.91,
5.00, 5.91 and 6.55 Hz. No seed went the other way. Exact two-sided
binomial p on five discordant seeds one way is 0.0625; the clause is a
count rule and the p-value is context.

Cadence moved. Median tip cadence over nc1's 24 standing runs is 6.36
Hz against nk1's 5.55 over its 23; paired on the 23 seeds where both
stood, nc1 is faster on 14 and slower on 9, median difference +0.91
Hz. Displacement did not: medians 4.15 against 4.11 mm. Final tilt
ran 5.78 to 10.77 degrees on nc1, no run near the limit, and every
foot lifted at least 6 times on every run. Two runs, 30008 and 30019,
read exactly 5.00 Hz at the tip, on the inclusive floor; nk1's own
30004 read the same number on its card. With both removed the count is
22 against 19.

The champion is nc1: ng1's 41 keys unchanged plus body_joint_tau_act_ms
6.0 on the six CTr_pitch joints. It is nk1's sibling, not nk1 plus a
joint; the two bodies share 41 keys and differ in which six joints
carry the 6 ms. The clip is seed 30011 replayed on the real body, 4.18
mm at 5.73 Hz, the seed where nk1 fell; the whole-CNS activity panel is
beside the body.

**Interpretation.** The coxa-trochanter joint is where the step's
lift-off and touch-down live, so a faster activation there shortens
the part of the cycle the muscle model was adding to every step, and
the fly steps faster; the knee acts inside the swing and the stance and
a faster knee changed only whether the stance leg held. That reading is
consistent with the numbers and is unmeasured beyond them: one value on
each of two joints, and a per-leg read of which leg gained. Entry 49
put the cadence floor on the right hind leg for ng1; whether nc1's
gain is that leg's is the first thing to read on its captures.

The bar's own quantity moved for the first time in a title change.
Every earlier v3.3 title turned on standing or on seeds; nk1 to nc1
turns on the same seeds with the same falls saved and 0.9 Hz added.
Cadence is the sharpest measured gait quantity the project holds
(5-16 Hz, Wosnitza and DeAngelis, band frozen), so a change that moves
it is the kind the bar was written to find.

**Problems.** One block of 24, no extension, five discordant seeds:
the count is decisive under the clause and thin as evidence about the
mechanism. Lane B's confirm is the second read. No flyscore screen ran
on this body before the crown; lane D's screen was on nk1, and the
seven-metric read on nc1 is open. Two runs sit on the 5.00 Hz floor at
the tip instrument's resolution. The tri_index conflict from lane A's
CD1-PF (anterior-rule median 0.0001 against the real 0.1606) is now
carried unread on a third champion; lane D's tripod_contrast reads the
same ng1 runs in band, and which instrument the reference constrains
is open. The knee-plus-hip body was not run; whether the two effects
add is the obvious next arm and belongs to the lane that built both.

**What changed in the model.** The champion. CHAMPION.json names nc1,
theta results-nc1-theta.json, clip walker_nc1_body.mp4; nk1 enters the
lineage as the last holder before it, reign 09:11 to 09:58 with no
defence. HALL-OF-FAME.json adds nc1 at score tier 4 and moves nk1 to
tier 3; the nk1 card gains a caveat that its clip footer carries ng1's
battery counts, a label copied from the ng1 renderer, not re-rendered.
CALIBRATION-LEDGER.json body.tau_act_ms: the searched point moved from
the knee to the coxa-trochanter joint. CONSTRAINT-REGISTRY.json
_live_titles.BANC: holder nc1, nk1 recorded as previous holder, the
pending challenge closed with the match record;
gap.tip_cadence_floor_resolution notes the two floor seeds. Engine
files unchanged by this entry; the fold that landed under it
(walk_search d1e3d566c8863040) is probed bit-identical.

**Attribution.** Robin's demand for real-fly cadence (v3, 2026-08-18)
is the clause this title change is the first to move; his delegation
of the bar (2026-09-02) is what let the judge run the match on the
holder's seeds twenty minutes after the last crown without a new
ruling. Lane B (Cursor) designed both single-joint bodies, chose the
coxa-trochanter joint, ran the NumPy twelve and registered the confirm
with its accept rule first. Lane D (Claude Code) asked for the abdomen
capture that makes the seven-metric screen possible on this body.
Lane A's CD1-PF conflict is carried onto the card. The judge (Claude
Code) nominated at 09:34, ran the match, rendered the clip, and
crowned.

cites:: lanes/judge/theta-ng1-ctr-tau6-challenger.json, lanes/judge/2026-09-03-title-v33-ctr-tau6-battery1.json, lanes/judge/2026-09-03-title-v33-ctr-tau6-battery2.json, lanes/judge/2026-09-03-title-v33-fti-tau6-battery1.json, lanes/judge/2026-09-03-title-v33-fti-tau6-battery2.json, results-nc1-theta.json, MODELCARD-nc1.md, lanes/B/2026-09-03-ng1-ctr-tau6-theta.json, lanes/B/2026-09-03-proto-ng1-ctr-tau6-numpy12-80528-result.json, lanes/B/2026-09-03-ng1-ctr-tau6-confirm-preregister.md, exchange/2026-09-03-0925-laneB-judge-packet.md, lanes/judge/render_walker_nc1_replay.py
uses-data:: CHAMPION.json, HALL-OF-FAME.json, CONSTRAINT-REGISTRY.json, CALIBRATION-LEDGER.json
