---
id: 52
date: 2026-09-03
title: The small cells were doing the work
tldr: Lane A read the champion's motor pools spike by spike and found the largest motor neurons in each pool silent and the small ones firing at 36 to 78 Hz, so under the measured 770-fold force range across a pool the cells doing the firing delivered almost no force and the hind legs could not lift. Compressing that range to 10-fold, one added number, put nc1's body inside the real-fly cadence band on the de-flickered read for the first time. On the holder's own Metal record seeds the new body qualified 22 of 24 against nc1's 0, both batteries passing on their own, and takes the v3.5 title as qg1, the first champion under a bar read with hysteresis on the Metal engine. Two falls of 24 and a walking speed still a third of the slowest real fly are what it did not fix.
era: standard
commit: 48e6f69ce (the state the batteries ran on, with the size_span seam ported under the bit-identity probe); the coronation bundle lands in the judge commit that follows, tagged qg1
seeds: judge 30001-30024, paired with nc1's Metal record receipts of 16:02 on the same seeds; A 201099-201116 Metal for the nomination
n: two twelve-seed batteries, 24 paired seeds; A eighteen paired seeds
wall: batteries 178.4 s + 176.1 s (Metal, in parallel); 17:31-17:34
reproduce: PYTHONPATH=. .venv/bin/python lanes/judge/title_match_v35.py --theta results-qg1-theta.json --seeds 30001-30012 --battery 1 --engine metal --out receipt.json
video: walker_qg1_body.mp4
video_url: https://media.theuploadinglab.ai/walker_qg1_body.mp4
---

**Question.** Why does the champion's hind foot drag? Entry 51 crowned
nc1 at 6.36 Hz on the threshold read; the 14:59 hysteresis instrument
(bar v3.4) then read the same runs at 3.50 Hz, every one under the 5 Hz
floor, and lane A's CD1-PU had shown the hind feet moving forward while
loaded in every run. Lane A's CD1-QE (17:00) asked the question one
level down: which motor neurons in each leg pool fire under nc1's
drive, and what force does the body assign them. The body follows
Azevedo 2020, where per-spike twitch force spans about 770-fold across
a pool, so each motor neuron's gain is 770 to the power of minus its
rank in the pool. The read on six seeds: the largest cells, gain 1.0,
are silent; the small cells, gain 1/250 to 1/770, fire at 36 to 78 Hz.
The hind levator pool summed to 0.001 activation. The legs were
commanded and the command reached cells the body had priced at nothing.

**Predicted (pre-registered).** CD1-QF (17:03, six seeds) and CD1-QG
(17:13, twelve seeds) registered a `size_span` dial that compresses the
range, 770 to 10 and to 30, with the clause that cadence fires if the
compressed body qualifies on the v3.5 read where nc1 does not. The
nomination at 17:20, hash frozen, predicted 18 to 22 of 24 qualifying on
the judge's seeds for span 10, 10 to 16 for span 30, and 0 for the
holder, whose Metal record already stood at 0 of 24.

**Method.** Lane A ran both spans against nc1 on 18 fresh Metal seeds
(201099-201116): span 10 qualified 16 of 18 and stood 16 of 18, two falls
over 130 degrees; span 30 qualified 10 of 18 and stood 18 of 18, its
misses at 3.8 to 4.9 Hz on the floor; nc1 qualified 0 of 18. The judge
ported the 13-line seam onto `walk_search.py` (absent key = the old
arithmetic; `lanes/A/probe.py` bit-identical) and ran the v3.5
instrument: two twelve-seed batteries on 30001-30024, Metal, spike
regularity 44, paired against nc1's 16:02 Metal record on the same seeds
and engine. A fall on the primary was expected somewhere in 24; the
match was decided on the count.

**Result.** qg1 22, nc1 0. Battery 1 standing 11 of 12, qualifying 11 of
12; battery 2 the same. The two losses are falls, 30010 at 77.6 degrees
and 30020 at 155.7, on seeds where nc1 stood; nc1 fell on 30003 and
30014, where qg1 stood, so standing is 22 against 22. On the 20 seeds
where both stood qg1 steps faster on all 20, median 6.77 against 3.64 Hz
on the hysteresis read, and every one of its 22 standing runs is inside 5
to 16 Hz where every one of nc1's is under 5. Every foot lifted at least
17 times in every standing run; nc1's record had two runs with a foot
under 5. Travel rose from 4.31 to 6.14 mm over six seconds. Lane A's
prediction of 18 to 22 met at 22. Seed 30003 is the clip: the first seed
of the block where the old body fell and the new one stood, 7.37 mm,
7.27 Hz.

**What it means.** The body's muscle model assigns force by the size
principle and the connectome's drive recruits the small cells first, as
the size principle says it should. The two facts are each measured, and
together they were silencing the legs, because the first-order activation
model between them has no tetanic summation: real muscle driven at 36 to
78 Hz holds a fused force many times a single twitch, and the model held
one twitch's worth. The 10-fold span is a searched stand-in for that
missing summation, recorded as such in the ledger row `mn.size_gain_span`
with the force-frequency curves that would replace it. Two points were
tried; the space between and beyond them, and nc1's `drive_scale`, which
was fitted with the hind pools muted, are open.

What did not change: the fly still travels 1.0 mm/s against a real 2 to 45
(a 42 percent gain on nc1 and still a conflict), two seeds in 24 fall,
and every open row on nc1's card (FeCO flexion population, tri_index,
posture with the abdomen off, the 12:24 re-grade's six failures, the
hind-foot drag itself) is carried unread onto this body. The mechanism
read that started this, CD1-QE seed 201060, does not replay exactly on
the judge's engine while two judge runs agree with each other; open with
lane A, and the title rows are the judge's own.

**Contribution ledger.** Lane A: the mechanism read (CD1-QE), the dial,
the two registrations, the prediction, the nomination with a frozen hash,
and the seam. Judge: the port under the probe, the match, the clip, the
coronation bundle, and the ledger row. Robin, 15:4x: the engine
instruction that put the match on Metal, which is why two batteries took
three minutes and the port, the match and the clip fit inside one
half-hour pass.

**Judge-seat update, 19:22 EEST.** No later result changed the champion. The
first qg1 card put footfall separation inside its reference band on 10 of 10
standing runs and put the median hysteresis swing at 16.25 ms, but both row
promotions are held for an isolated Metal replay. Five conflicts remain on
that card, including front-knee range at 74.4 degrees, net speed at
1.43 mm/s, and straightness at 0.3689. Adding abdomen gain 0.3 put posture variability inside its band on
five of five Lane D runs, with four standing and about one leg of cadence lost.
That is the only current extension with a shared-model signal.

The earlier replay discrepancy is now explained. Six concurrent Metal runs of
seed 201060 split between two exact trajectories because relaxed float-atomic
scatter depends on device contention. Routine BUILD runs may retain shared GPU
use with that caveat. Decision-grade runs now request an exclusive
repository-wide Metal lock, and the title runner aborts on any run exception.
qg1's 22 to 0 title result is unchanged, but exact isolated replay remains due
on a Metal-capable judge.

Arie's consultation changed the search route, not the current title. Neural
parameter grouping is cell-type-first rather than hemilineage-first. Blanket
inhibitory downweighting was rejected; CT1's single point state was recorded as
a visual representation gap and assigned a one-seed silencing screen. Lane C
now moves upstream to the native MaleCNS R1-R6 to T4/T5 to
visual-projection-neuron path, with selective T4/T5 ablation deficits as the
acceptance test.
The open recruiting decision is whether Robin wants Arie's warm introduction
to Philip Shiu at Eon.
