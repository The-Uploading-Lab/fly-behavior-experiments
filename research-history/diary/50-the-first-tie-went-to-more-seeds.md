---
id: 50
date: 2026-09-03
title: The first tie went to more seeds
tldr: Lane B's challenger, the champion ng1 with one added number, a 6 ms muscle activation time constant on the six knee joints where every joint had shared 12.3 ms, tied ng1 19 to 19 on the holder's 24 paired judge seeds at 08:32, the first tie of the record. The ordering clause said a tie keeps the holder, and the engine is deterministic per seed, so a tied challenger could never win a re-match. The judge wrote item 5 at 08:36, drew the next 24 judge seeds, and ran both flies on them. The knee body won 17 to 12, 36 to 31 over 48, and takes the title as nk1. It wins by falling less (three falls in 48 against seven), not by stepping faster: all nine of its standing losses are a body under the 5 Hz cadence floor. Lane B's own NumPy twelve on lane seeds tied 9 to 9, and lane B said do not crown on that; the count rule crowns.
era: standard
commit: b84b0a35a (state the extension batteries ran on; batteries 1-2 on 4e1c13494); the coronation bundle lands in the judge commit that follows, tagged nk1
seeds: judge 30001-30024 (tie) and 30025-30048 (extension), both arms; B 80516-80527 NumPy, 80492-80515 Metal; D 60520 Metal for the flyscore screen
n: four twelve-seed batteries per arm, 48 paired seeds; B one NumPy twelve and two Metal twelves; D one seed
wall: batteries 789.2 s + 786.5 s (base, NumPy, in parallel) and 1400.7 s + 1373.8 s challenger, 1375.9 s + 1381.6 s ng1 (extension, four batteries in parallel); B 1936.6 s; 08:14-09:11
reproduce: PYTHONPATH=. .venv/bin/python lanes/judge/title_match_v33.py --theta results-nk1-theta.json --seeds 30025-30036 --battery 3 --out receipt.json
video: walker_nk1_body.mp4
video_url: https://media.theuploadinglab.ai/walker_nk1_body.mp4
video_receipt: vision/results-c-v70-cell-relative-command-build.json
---

**Question.** Does a faster muscle activation on the knee beat the
champion on the champion's own seeds? The body's single activation
time constant, 12.3 ms, is Azevedo 2020's 8.5 ms spike-to-half-force
divided by ln 2, and their 8.5 ms includes nerve conduction and
neuromuscular delay, so true activation is faster than 12.3 ms on every
joint. Lane B put 6 ms on one joint, the femur-tibia pitch (the knee)
of all six legs, and left the other 36 joints at 12.3 ms. Two Metal
twelves on lane seeds beat ng1 8 to 6 and 11 to 6, which met the
nomination bar, and the judge froze the theta hash at 08:14 before any
judge seed ran.

**Predicted (pre-registered).** The nomination named the ordering
clause as the test: higher qualifying count over the holder's 24
paired seeds takes the title, a tie keeps the holder. No prediction of
the count was written. Lane B's own NumPy twelve on 80516-80527
(preregistered 08:2x, run in parallel with the match) predicted the
knee would reduce falls and named an under-floor accept condition,
which it missed.

**Method.** The match followed the ordering clause: two twelve-seed
batteries on 30001-30012 and 30013-30024, NumPy, spike_reg 44, on the
holder's engine of that hour (walk_search f5c83dc63c877e3b, cns
ebf0eb7bdfa726e7), paired with ng1's 2026-09-02 receipts on the same
seeds. Each run qualifies when it stands at the end (tilt under 51.68
degrees, height over 1.0, displacement at least 0.3 mm), lifts every
foot at least five times, and its tip-measured cadence lands inside
5 to 16 Hz with no commanded step frequency in the theta. The count
came back 19 to 19 at 08:32. Item 3 of the clause held: ng1 kept the
title, its eighth defence.

The tie exposed a defect in the clause. The engine gives the same
result for the same seed every time, so a challenger that ties on the
24 returns the same tie on every re-match, and no fresh-seed evidence
about it can ever crown it. Item 5, written at 08:36 and committed
before any extension seed ran: on an equal count over the 24, the
judge draws the next 24 judge seeds and runs BOTH arms; the higher
combined count over 48 takes the title, an equal count over 48 keeps
the holder; the challenger must still pass each fresh battery on its
own; one extension per nomination. Both arms went onto 30025-30048 at
08:39, four batteries in parallel, and the receipts landed at 09:02.

The pre-coronation screen was already on file: lane D ran the
challenger through flyscore on Metal seed 60520 at 08:2x, tripod
contrast 0.010, gait signal quality 0.179, left-right asymmetry 0.223,
all inside the real-fly bands, and the judge's replay of that seed on
main was exact.

**Result.** Extension: the knee body 17, ng1 12, on 30025-30048. Its
two fresh batteries pass on their own, 8 of 12 qualifying with 11
standing and 9 of 12 with 11 standing; ng1's were 6 and 6. Combined
over 48: 36 against 31. Twelve seeds the knee body qualifies and ng1
does not (30012, 30015, 30020, 30023, 30025, 30030, 30035, 30038,
30039, 30041, 30042, 30043); seven the other way (30002, 30008, 30010,
30011, 30029, 30037, 30045). McNemar's test on the 19 discordant seeds
gives a two-sided p of about 0.36; the clause is a count rule and the
p-value is context.

Standing is where the difference is: 45 of 48 against 41 of 48, three
falls against seven. On the seven ng1 falls, the knee body stood on
five. Cadence is not: the two flies' median tip cadence over standing
runs is the same number, 5.45 Hz, and every one of the knee body's
nine standing non-qualifying runs is under the 5 Hz floor, at 4.27 to
4.91 Hz (30002, 30008, 30010, 30018, 30028, 30029, 30036, 30037,
30047). Displacement medians 4.17 against 4.35 mm over the 48;
displacement orders nothing.

Lane B's NumPy twelve on 80516-80527, delivered at 08:45 while the
extension ran: 9 to 9 qualifying. The knee saved three of ng1's falls
(80517, 80520, 80525), added one (80522), and put two seeds under the
floor (80518, 80521). Same shape as the judge seeds, at a tie. Lane B
wrote "Do not crown" on that result. The title does not turn on a lane
twelve; it turns on the clause, and the clause crowns.

The champion is nk1: ng1's 41 keys unchanged plus
body_joint_tau_act_ms 6.0 on the six FTi_pitch joints. The clip is
seed 30012 of battery 1 replayed on the real body, 5.43 mm at 5.36 Hz,
one of the seeds where ng1 stood and stepped under the floor at 4.73
Hz; the whole-CNS activity panel is beside the body.

**Interpretation.** A faster knee changes how often the fly stays up
and not how fast it steps. That is consistent with the time constant
acting on the stance leg's ability to hold the body when the swing
legs move, and with cadence being set upstream, in the cord, where the
knee constant does not reach. It is also consistent with the 5 Hz floor
being the same one-leg floor entry 49 measured on ng1: the knee body's
losses are the same kind of loss, a right hind leg stepping under 5 Hz
while the rest of the fly walks. Nothing in this match moves that leg.

Whether 6 ms on the knee is the right number, or the knee the right
joint, is unmeasured. One point in a 42-joint space was tried because
one lane tried it. The measured-structure directive's shape rule
applies here in reverse: activation time varies per muscle in the
animal, the body now has a per-joint parameter, and the other 36
values are still the derived 12.3 ms nobody chose.

The tie itself is the finding of the morning for the bar. Two flies
that differ by one number on six joints scored equal on 24 seeds and
5 apart on the next 24. A 24-seed match has a resolution of roughly
that size, and the two-battery rule of v3.1 was written for exactly
this seed luck. Item 5 gives the clause a remedy that is more evidence
rather than another quantity.

**Problems.** The paired difference over 48 is 12 to 7 on discordant
seeds; at p about 0.36 it is not by itself strong evidence that the
knee constant matters, and the title changes hands on a count rule
that says nothing about significance. Lane B's Metal screens ranked
the two flies further apart (8 and 11 against 6 and 6) than NumPy does
on any block; every Metal count is a different fly per seed, and lane
B's own 80504 replay said so first (stood on Metal, fell on NumPy).
Batteries 1-2 and 3-4 ran on different execution commits (4e1c13494
and b84b0a35a) with the same engine hashes; the judge's probe on the
fold between them was bit-identical. The main engine has since taken
lane C's cell-relative command seam (walk_search 072a3cbb09da81be, cns
d7ebac3530d83c5a), probed bit-identical and champion smoke unchanged;
the nk1 render on that engine reproduced the battery row exactly,
5.4324 mm against 5.43.

**What changed in the model.** The champion. CHAMPION.json names nk1,
theta results-nk1-theta.json, clip walker_nk1_body.mp4; ng1 enters the
lineage as the last holder before it, with its eight defences.
HALL-OF-FAME.json adds nk1 at score tier 4 and moves ng1 to tier 3;
the v3.3 bar's clause text gains item 5. CALIBRATION-LEDGER.json
body.tau_act_ms: provenance now derived on 36 joints and searched
(one point, 6.0 ms) on the six knees. CONSTRAINT-REGISTRY.json
_live_titles.BANC: holder nk1, ng1 recorded as previous holder with
its defences, the pending challenge closed with the match record.
Engine files unchanged by this entry.

**Attribution.** Robin's delegation of the bar (2026-09-02, "the bar
is your's, have agency") is what let the judge write item 5 in the
four minutes between the tie and the extension; Robin's standing
demand for real-fly cadence (v3, 2026-08-18) is the clause every loss
here fails, and his rule that every clip carries the CNS panel beside
the full body shapes the record's video. Lane B (Cursor) designed the
knee body, ran the two Metal screens and the NumPy twelve, and wrote
the "do not crown" reading of its own tie. Lane D (Claude Code) ran
the pre-coronation flyscore screen. Lane A's 04:47 read of the FeCO
flexion population is carried onto the new card as an open conflict.
The judge (Claude Code) nominated at 08:14, ran the match and the
extension, wrote item 5, rendered the clip, and crowned.

cites:: lanes/judge/theta-ng1-fti-tau6-challenger.json, lanes/judge/2026-09-03-title-v33-fti-tau6-battery1.json, lanes/judge/2026-09-03-title-v33-fti-tau6-battery2.json, lanes/judge/2026-09-03-title-v33-fti-tau6-battery3.json, lanes/judge/2026-09-03-title-v33-fti-tau6-battery4.json, lanes/judge/2026-09-02-title-v33-ng1-battery1.json, lanes/judge/2026-09-02-title-v33-ng1-battery2.json, lanes/judge/2026-09-03-title-v33-ng1-battery3.json, lanes/judge/2026-09-03-title-v33-ng1-battery4.json, results-nk1-theta.json, MODELCARD-nk1.md, lanes/B/2026-09-03-ng1-fti-tau6-theta.json, lanes/B/2026-09-03-proto-ng1-fti-tau6-numpy12-80516-result.json, exchange/2026-09-03-0845-laneB-judge-packet.md, lanes/D/2026-09-03-ftitau6-challenger-flyscore-result.json, lanes/judge/render_walker_nk1_replay.py
uses-data:: CHAMPION.json, HALL-OF-FAME.json, CONSTRAINT-REGISTRY.json, CALIBRATION-LEDGER.json
