---
id: 53
date: 2026-09-03
title: The abdomen steadied the fly
tldr: The champion's abdomen had never moved. Lane D drove it from its own 168 abdominal motor neurons through a guessed map and found that a small gain, 0.2, puts the posture measure inside the real-fly band and, on twelve paired fresh seeds, stands the vigorous body up on four seeds where qg1 fell. On the judge's 24 paired seeds the new body qualified 23 against qg1's 22, both arms run one battery at a time under the exclusive GPU lock, and takes the v3.5 title as qh1 by one seed. The same match replayed qg1's recorded batteries in isolation and found them equal on every field of every seed, so the champion record's load caveat is discharged by measurement. What it did not fix is the speed, 1.1 mm/s against a real 2 to 45, and the cadence moved down toward the floor.
era: standard
commit: b22495039 (the state the batteries ran on; the coronation bundle lands in the judge commit that follows, tagged qh1)
seeds: judge 30001-30024, both arms; D 60626-60640 for the nomination
n: two twelve-seed batteries per arm, 24 paired seeds; D twelve paired seeds
wall: qh1 352.7 s + 106.5 s, qg1 129.0 s + 105.8 s (Metal, exclusive lock held per run, three lane processes interleaving); 20:34-20:46
reproduce: PYTHONPATH=. FLY_WBE_METAL_LOCK_MODE=exclusive .venv/bin/python lanes/judge/title_match_v35.py --theta results-qh1-theta.json --seeds 30001-30012 --battery 1 --engine metal --out receipt.json
video: walker_qh1_body.mp4
video_url: https://media.theuploadinglab.ai/walker_qh1_body.mp4
---

**Question.** Can the one flyscore conflict every champion since nk1 has
carried, posture variability 0.015 radians against the real 0.026 to
0.086, close without costing the walk? The measure is the variability of
the abdomen-thorax angle, and the abdomen of every champion was rigid:
BANC has 168 abdominal motor neurons and the walker actuated none of
them. The judge built an adapter at 12:30 (lane B's 11:01 census, a
neuromere fold onto the body's five abdomen links, one gain, a 30 degree
clip, all ten signs +1) and lane D took the search.

**Predicted (pre-registered).** Lane D's 18:12 card at gain 0.3 put
posture in band on five of five seeds with four standing, and gain 1.0
toppled three of five. Its 19:17 five-seed screen at 0.15 to 0.30 read
standing at 3 or 4 of 5 at every gain and the route was dropped as a
standing cost. At 19:54 lane D froze a nomination theta anyway, qg1 plus
`abd_gain` 0.2 (sha 4bc4ba31), and ran the bar's own instrument on
twelve paired fresh seeds, single worker, isolated: 11 of 12 qualifying
against qg1's 8 of 12, four seeds rescued and one lost. The five-seed
screen had used a cruder standing read than the clause. The nomination
predicted that the abdomen stabilises rather than topples at this gain.

**Method.** The judge ran the v3.5 instrument on the holder's 24 judge
seeds for both arms, one battery at a time under
`FLY_WBE_METAL_LOCK_MODE=exclusive`, because lane E had shown at 19:54
that one qg1 seed flips from standing (7.51 mm) to fallen (3.44 mm)
between three-worker and isolated execution, twice identical in
isolation. The holder's arm was therefore not read from its record but
replayed under the same condition as the challenger. Four receipts,
`lanes/judge/2026-09-03-title-v35-qg1abd02-battery{1,2}.json` and
`...-qg1-isolated-battery{1,2}.json`.

**Result.** qh1 23, qg1 22. Battery 1 standing 11 of 12 and qualifying 11
of 12 (30009 fell at 171.5 degrees); battery 2 standing 12 of 12,
qualifying 12 of 12, including 30020 where qg1 falls at 155.7. qg1's
isolated replay: 11 of 12 and 11 of 12, falls on 30010 and 30020,
identical to its 17:31 record on every field of every seed. Discordant:
qh1 stood on the two seeds qg1 fell on and fell on one seed qg1 stood
on. On the 21 seeds where both stood qh1 steps slower on 16 (median 5.82
against 6.77 Hz on the hysteresis read, four runs within 0.5 Hz of the 5
Hz floor, 30017 at exactly 5.00) and travels further on 13 (median 1.14
against 1.12 mm/s). Every foot lifted at least 16 times in every standing
run. Seed 30020 is the clip: qg1 falls, qh1 stands and steps at 7.00 Hz.

**What it means.** A body walking this vigorously is close to falling,
and two of qg1's 24 title seeds did. A small mass swing at the back,
driven by the cord's own abdominal motor output, damps those falls at a
gain where it does not yet cause new ones; the same swing at gain 1.0 is
what toppled three of five seeds in the afternoon. The mechanism is the
reading of one lane on two dozen seeds, not a measured fact, and the
margin is one seed on 24 paired: two discordant against one. The clause
is a count rule and crowned on it; item 5's 48-seed extension applies
only to an equal count. The map from abdominal motor neuron to body
joint is a guess (no muscle label exists for any of the 168 cells), the
range and signs are guesses, and now all three are load-bearing, so
their replacing measurements move up the list.

What did not change: speed, 1.14 mm/s against 2 to 45, the oldest
conflict, unread on this body beyond the title receipts; cadence moved
down, not up; the champion-level closure of the posture row is the
judge's card read of the title runs, pending, so the row stays CONFLICT
tonight on lane D's five-seed reading alone; and every open row on qg1's
card, the bilaterally in-phase gait lane A measured at 20:19, the hind
left-right asymmetry, the FeCO flexion population, body height, R1i, the
front-knee sweep, is carried unread onto qh1.

One thing did change that is not about this body. The 19:12 correction
had said exact replay of qg1's parallel title batteries was unproven. It
is now proven for that load: 24 seeds, every field equal. The caveat
stands for heavier loads (seed 201060 six-wide, lane E's seed 100273
three-wide), which is why decision-grade runs keep the exclusive lock.

**Contribution ledger.** Lane D: the abdomen route from the 18:12 signal
through the 19:17 drop and its own reversal, the frozen nomination, the
paired twelve-seed card, and the mechanism reading. Lane B: the 11:01
census (168 abdominal motor neurons, none with a muscle label) and the
11:38 neuromere map guess the adapter folded. Lane E: the 19:54
contention finding that made the judge replay the holder's arm instead
of reading its record. Judge: the 12:30 adapter, the match with both
arms isolated, the coronation bundle, the clip. Robin, 15:4x: the engine
rule that put every arm on Metal, which is why four batteries took twelve
minutes; 2026-08-10: the zero-conflicts aim, which is the reason a
posture row existed to chase.

**Judge-seat update, 21:24 EEST: the champion's card, read on the champion.**
Robin's goal for the judge was an honest card: the registry had been graded
on nc1, two crowns ago. Twelve fresh judge seeds on qh1 under the exclusive
lock (receipt `lanes/judge/2026-09-03-card-qh1-metal.json`), eleven standing.
Closed on the champion: posture variability, the row the abdomen was built
for, 0.035 radians median with all eleven inside the real 0.026 to 0.086;
swing duration 22.5 ms and footfall separation 1.094 body lengths, both held
on qg1 for an isolated replay and now read isolated; the front hip and the
four middle and hind joint sweeps; tripod contrast, gait signal quality and
left-right asymmetry on the fixed scorer; cadence 5.82 Hz, duty, lifts, the
load-stance rules. Open on the champion: speed 0.98 mm/s against 2 to 45,
straightness 0.15 against a 0.50 floor, body height 1.22 mm against 0.80 to
0.89, front knee sweep 75.5 degrees against 81 to 112, the tip-stance R1i
rule, and the three contralateral phase rows lane A measured tonight: qh1
steps each segment's two feet within 24 to 57 degrees of in phase where the
real fly is 178 to 180 degrees apart. The left hind steps under the 5 Hz
floor at the median (4.36 Hz) while the right hind steps at 6.36. Three
assumed quantities carry the title, the abdomen map, range and signs, and
the card now says so at the top.


**Judge-seat update, 2026-09-04 00:45 EEST: the title is vacant again.** Lane
A found that a third of the champion's counted lifts are one-tick hover taps
on the hysteresis read, and the judge's re-read of qh1's own 24 title seeds
with a 10 ms minimum swing episode gives 4.45 Hz where the hysteresis read
gave 5.82, three of 24 qualifying where 23 did. Bar v3.6 (AGENTS.md) reads
cadence and lifts with that minimum; the v3.5 title closes with qh1 as its
last champion, one defence held, and the v3.6 title starts vacant. The same
logic made v3.4 in the afternoon; the instrument that survived nc1 did not
survive the body it crowned.

**Judge-seat update, 2026-09-04 03:32 EEST: a cut on vision, and the shape of
the missed step.** Lane C ran the male connectome (MCNS) in the engine, matched
its optic-lobe firing rate to the female's, and found that the wiring offset
the female lacks and the male has (148 to 168 degrees) gives no direction
selectivity in T4/T5, and neither does a real 30 or 60 ms delay on the
delayed arm added on top of it: the two families respond more to a moving
grating with the delay in, but never in opposite signs. In linearly summing
point neurons a delayed input and an undelayed one deliver the same total
whichever way the grating moves, so the cut is a model class, not a
parameter: the missing ingredient is a coincidence nonlinearity. Registered
before each run, both animals, and the earlier "140-fold attenuation" number
withdrawn by the lane itself the same night when it found the trace rows were
sorted and carried the motor neurons. On the walking side, lane A traced the
missed step to its end: the joint tracks its command, the foot rises 0.3 to
0.8 mm in the body frame on missed and completed cycles alike, and the world
clearance is that rise less a 0.4 mm dip of the whole body a quarter cycle
after all six legs flex together. The lock between segments is carried by
between-segment excitation, its inhibitory half fires and is inert, and no
gain on it, on the commissural class or on load feedback moves any pair
toward the alternation a walking fly has.
