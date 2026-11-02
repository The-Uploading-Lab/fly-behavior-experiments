---
id: 11
date: 2026-08-17
title: Five collapses, one cause — the chamber grades on its own dice
tldr: A challenger that stood on 4 of 4 tune-up runs stood on 2 of 6 fresh ones. It is the fifth time that has happened; we finally named the cause and changed the search rather than the fly.
era: standard
commit: 303c6b8
theta: challenger = "winner" in results-round19.json; incumbent = results-p150-theta.json
seeds: battery 830-835 (fresh; nothing in the search touched them); chamber and rerank seeds drawn at random
n: chamber 180 candidates x 3 seeds; rerank 10 candidates x 4 seeds; battery n=6
wall: not recorded in the round-19 results files
reproduce: .venv/bin/python3 regression_walk.py --theta results-round19.json --out results-regression-walk-round19.json
---
**Question.** The champion crowned yesterday — the phasic fly — had
pinned its new sensing dial at the ceiling we allowed. Round 19
raised that ceiling to 3.0 and re-searched from the champion. Does
more phasic sensing buy a better fly?

**Predicted (pre-registered, quoted).** From the round-18
registration: "round 19 raises the phasic ceiling (search (0, 3.0) —
the chamber voted for more), seeds from the phasic fly", with a
coordination secondary "with teeth: report mean-PLV-excess and
tri-index on the winner's standing seeds." The bar was written
before the run and left unchanged: a challenger takes the title only
if its battery beats the champion on standing (> 3/6) **AND** speed
(> 1.03 mm/s), with all-feet lifts >= 5.

**Method (jargon defined inline).** The search ("chamber") drew 180
parameter sets and scored each on 3 randomly chosen simulation seeds
— a seed is one roll of the dice the noisy network is run under. The
top 10 were then re-run ("rerank") on 4 fresh seeds each, and the
best of those went to the battery: the fixed constraint checklist,
re-measured on 6 seeds no part of the search had ever seen. That
last step is the practice the same day's reporting directive names
in plain words — never grade the fly on dice rolls it practiced on.

**Result (numbers, with variance).** The rerank winner was an
extreme animal: phasic 2.85, command 156 Hz, campaniform gain 48,
joint f0 6.0 Hz, adhesion threshold 0.1. On its 4 rerank seeds it
stood **4 of 4** — upright 0.99 / 0.93 / 0.97 / 0.94, displacement
5.15 / 5.50 / 5.69 / 5.32 mm, median **5.41 mm**. That is the
strongest rerank in the project's history.

Battery, per seed (seed: mm, upright): 830: 6.52, 0.99 | 831: 3.88,
0.16 | 832: 0.65, 0.19 | 833: 4.00, 0.37 | 834: 7.63, 0.73 | 835:
5.58, 0.97. **Standing 2/6, speed 1.10 mm/s** (1.19 and 1.01 on the
two standing seeds). The speed clears the bar; the standing does
not; the bar is an AND. **R19 FAIL — the phasic fly keeps the
title.**

What stands from the chamber anyway: it pinned the phasic dial at
the raised 3.0 ceiling *again* — at the halfway mark the leader was
a candidate passing all 3 of its seeds with all-positive
coordination excess (+0.007 / +0.033 / +0.058) at exactly phasic
3.00, and 25 of the 180 candidates ended within 0.01 of the ceiling
(only 4 of 180 passed all three seeds at all). And the challenger
scored the first contralateral swing-avoidance pass on both standing
seeds in recent memory (R1c_tip p = 0.01 and 0.005).

**Conflicts carried, stated with the fail:** cadence 2.68 Hz against
the real 5-16 Hz band; median swing 55 ms against a 60 ms band edge;
speed 1.10 mm/s against a real-fly floor of 2.0.

**Interpretation.** Five of these collapses now exist, and they have
the same shape: m050xDNg100, coact, the round-16 winner (battery 1/6,
speed 0.19), t025+phasic1.0 (battery 0/6), and this one (2/6). Score
180 candidates on 3 dice rolls each, take the best, and you have
selected partly for the parameters and partly for the dice. A
4-seed rerank over the top 10 does not undo that — reranking a
shortlist the same dice produced is another selection step, not an
evaluation. The enemy is not the fly, not the ceiling, and not the
bar: robustness was only ever tested after the winner was chosen.

**Problems / what bit us.** (1) The rerank looked like the best
evidence the project had produced, and it was worth nothing on its
own. "Strongest rerank ever" is a statement about a selection stage.
(2) Speed genuinely improved — 1.10 against the champion's 1.03 —
and there is an obvious reading in which this round half-won. The
bar was written as an AND before the run precisely so that reading
is unavailable afterwards.

**What changed in the model.** Nothing in the fly; the change is in
how the search is graded. Round 20 is registered and committed
before any run: **4 seeds per candidate inside the search**, phasic
ceiling raised a third time to 5.0 (two consecutive ceiling pins),
anchors p150 / p150@ph3.5 / p150@ph4.5, N = 240, same bar against
the same champion.

**Attribution.** Human: the standing rules this verdict is enforced
by — fresh-seed evaluation and pre-registration, both named in the
reporting-standards directive of the same day; and the governing aim
that makes a published failure worth the words, "every measurement
is a constraint that might just make our goal vector space narrower"
(2026-08-10). Agent (Claude Fable 5, per the commit trailer): the
round-19 chamber, rerank and battery; the diagnosis that five
scattered collapses share one cause; the round-20 registration that
acts on it.
