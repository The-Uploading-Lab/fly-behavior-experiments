---
id: 86
date: 2026-09-11
title: The same budget on three wirings, the shuffles stand and none of them steps
tldr: Robin asked for the fairer half of the reader's question, the same tuning budget on the measured wiring and on both shuffles. Thirteen dials searched from scratch, 156 runs per wiring, pre-registered. The measured wiring's winner stands 12 of 12 and steps on every foot at 4.7 Hz, qualifying 3 of 12; the degree shuffle's winner stands 12 of 12 by sliding 3.9 mm with a foot that never lifts; the type shuffle's stands 9 of 12 at 2.2 Hz with feet that never lift. Both registered predictions held. Clips of all three winners are on the page.
era: standard
budget_exception: Fourth entry because Robin asked for the same-budget run and its report the same day, its result closes the half that entry 83 recorded as owed, and its three clips carry exact receipts that the generator requires an entry to attach on the day they are published.
video_receipt: publication-receipts/walker_wi1_null_real_seed30555_samebudget.sha256-45ddfb7620401f97448ad920dc903852d64d71301e111ca79bccc502a33be873.publication.json, publication-receipts/walker_wi1_null_degree_seed30555_samebudget.sha256-4fb930747db4791e7fdb4a28e0cc4489b0b82c1b9e19782f59744cbe37fb4156.publication.json, publication-receipts/walker_wi1_null_type_seed30555_samebudget.sha256-ed87d1001e1f8c4257c6248081ce88a152926096fc1b972322f9c07517697e02.publication.json
commit: the commit this entry lands in, on main; engine at c7d79be7e; registration in 992bb54f3 before the first run
theta: results-wi1-theta.json (sha256 ba18676b65b3db0747532d344cd2a2847ac8e38335bb77d3c34b29e9b7802fe7) as the fixed mechanism set, spike_reg 44; thirteen dials overridden per candidate
seeds: search seed 30900 for every candidate; grading on 30544-30555; rng seed 20260912 on every arm; shuffle draw 20260911
n: 120 candidates and 36 grading runs per arm, three arms, 468 runs
wall: 9 to 14 s per run on the deterministic Metal engine, one worker, rising to 86 s while the machine paged; about 3 h 20 min end to end including one restart
reproduce: PYTHONPATH=. .venv/bin/python lanes/field/2026-09-11-null-examples/same_budget_search.py lanes/field/2026-09-11-null-examples/same_budget_search.json real,degree,type
---

**Question.** Entry 83 answered the reader's question at fixed parameters
and said the symmetric half was owed. Robin, an hour later: *"can you do a
run where the tuning budget is the same for this exact example we have on
the site now?"* So: given equal freedom to fit, at a small budget, does the
measured BANC wiring walk better than a degree-preserving shuffle and a
cell-type-preserving shuffle of it?

**Predicted** (`lanes/field/2026-09-11-null-examples/PREREG-same-budget.md`,
committed at 992bb54f3 before any run): *"1. The measured arm's best graded
candidate qualifies on more of the twelve seeds than either shuffle's best.
2. At this budget the measured arm does NOT recover the champion's 8 of 12.
3. Whether a shuffle stands at all is not predicted."* Falsifier: *"A shuffle
arm whose best graded candidate qualifies on at least as many seeds as the
measured arm's best."*

**Method.** wi1's theta of record is the fixed mechanism set on every arm.
Searched from scratch, per arm and identically: the thirteen continuous
dials with declared bounds in `walk_search` (SPACE: campaniform gain, force
reference, drive scale, joint time constant, adaptation increment, command
rate; SPACE_NETWORK without short-term depression: delay, adaptation time
constant, balance target, synaptic scale, graded gain, graded midpoint,
graded slope). 40 uniform draws, then 80 hill-climb steps from the running
best with `walk_search.neighbour`'s step (normal, sd 0.15 of the range,
clipped), one 6,000 ms run per candidate on search seed 30900, scored by
displacement if the run stands (v3.2 clause) else 0. The rng seed is the
same on every arm, so the 40 random candidates are the same settings on all
three wirings. The three best distinct candidates per arm were graded on
30544-30555; the arm's result is the best by qualifying, then standing, then
median travel. Nothing starts from the champion's values.

**Result.**

    arm                          stands  qualifies  travel   cadence  every-foot-lifts  standing in search
    measured wiring               12/12     3/12    9.92 mm  4.73 Hz     12/12 seeds        62/120
    degree-preserving shuffle     12/12     0/12    3.93 mm  0.77 Hz      0/12 seeds        51/120
    cell-type-preserving shuffle   9/12     0/12    6.55 mm  2.18 Hz      0/12 seeds        46/120

Prediction 1 held (3 against 0 and 0). Prediction 2 held (3 against the
champion's 8). The measured winner (candidate 87) stands on every seed,
travels 1.7 to 15.0 mm, and lifts every foot 10 to 17 times per run; it
misses qualifying on nine seeds on cadence alone, 3.8 to 5.3 Hz around a
floor of 5. The degree winner (candidate 56) travels 3.8 to 4.1 mm on every
seed with at least one foot that never lifts, its drive scale at the box
ceiling and its campaniform gain at the box floor: a shove, not a gait. The
type winner (candidate 81) falls on three seeds (179, 179 and 75 degrees),
travels 2.6 to 8.9 mm on the rest, and never has all six feet lifting.
Per-seed rows, dials and search traces:
`lanes/field/2026-09-11-null-examples/same_budget_search.json`.

**Interpretation.** At this budget the shuffles are not dead, which the
fixed-parameter clips could not show: given their own tuning both stand and
both move. What neither reaches is a step cycle in which every leg lifts,
which the measured wiring reaches on every seed from the same starting
draws. The type shuffle, which keeps type-to-type counts, recovers more
than the degree shuffle, which keeps only degrees; that ordering is the one
the null family was designed to expose. This is one search per wiring, one
search seed and 156 runs each: a screen, not the stage 5 experiment at the
lineage's scale, and the page says so.

**Problems / what bit us.** The machine paged during the degree arm (lane D
and this seat both holding Metal engines, then this seat alone with the
compressor at 8.8 GB), and each run slowed from 21 to 86 seconds. The search
was stopped at candidate 85 of the degree arm, the runner changed to a fresh
memo per run so its memory stopped growing, and it resumed from its logged
rows without re-evaluating any; the pace returned to 11 to 12 seconds. The
restart is a runner change, not a protocol change, and is recorded here.
"Stands" admits a sliding fly, since its travel clause is 0.3 mm; the
every-foot-lifts column is what separated the arms.

**What changed in the model.** Nothing. No theta, registry row, bar or
champion record is touched. The renderer gained a dial-override option to
film the winners.

**Attribution.** Robin, 2026-09-11 morning: asked for the same-budget run on
the page's example, then "just deploy to the site and send me a report on
telegram when done". The Field Engineer seat (Claude Code, on the Pro) wrote
the registration, the search runner and the clips, ran the three arms, and
wrote the page section and this entry. The judge did not grade this; it is
not a title event.
