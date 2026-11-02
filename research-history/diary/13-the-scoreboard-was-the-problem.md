---
id: 13
date: 2026-08-18
title: The scoreboard was the problem
tldr: Three times in one day we went looking for a better fly and found a broken scoreboard instead. Six-seed titles turned out to be coin flips, so verdicts became rates on twelve dice. The search had been scoring distance while the title measured balance, so eight straight collapses were an objective mismatch and not variance — fixing it produced the first championship change in this project's history. And every chamber ever run had picked the slowest joint it was offered, because we scored standing and a slow joint is a stable joint. Score cadence instead and the search walks to 30.1 Hz, within 1.3 Hz of a number measured in fly muscle by a completely independent road.
era: standard
born: an objective mismatch looks exactly like variance until you fix it — and a search cannot walk toward something you never scored
commit: arc 4d3e868..7c9c21a (86fc21e the championship, acfb2ca the power fix, 7c9c21a cadence)
theta: results-p150-theta.json (deposed); results-f23-theta.json (champion: f0 15.0, zeta 1.08, cmd 62.9 Hz, adh 1.74, k_gain 141, phasic 1.5, camp 25); results-round-f25.json and results-round-f26.json (cadence chambers)
seeds: v2 re-grades female 838-865 and male 943-954; F22 title 2100-2113; F23 chamber 130 x 6, title 2201-2213, paired decider 3000-3012 (identical for both arms); F25 chamber 130 x 6; F26 chamber 200 x 6, title v3 5100-5113 — 12 fresh each, none reused
n: 24 + 12 + 780 + 12 + 24 + 780 + 1224 + 12 runs, 6 s each; F27 manipulation test 3 x 3 s
wall: midday to 22:59, 2026-08-18
reproduce: .venv/bin/python3 walk_search_f26.py   # then regression_walk.py --theta results-round-f26.json --seeds 12
---
**Question.** Three questions, and they turned out to be one. Is six
seeds enough to decide a title? Eight times a chamber winner had been
reranked, looked strong, and then failed its title battery — is that
variance we have not beaten, or are the chamber and the title measuring
different things? And why does every chamber we run, given any range of
joint frequency, pick the slowest value in it?

**Predicted (pre-registered, quoted).** Each fix registered before it
ran. On power: "both animals sit at 4/12 standing on all seeds graded
to date, so the likeliest result is that **NEITHER holds a title
tonight**... Recorded before the runs so a double failure cannot be
quietly reframed." On the objective: "if the objective was the problem,
F23's winner should hold its rate into the twelve-dice match where
seven predecessors did not... A chamber rate of 0.7-0.87 predicts
roughly **8-10 of 12**; anything near 4/12 says the collapses are
variance we have not beaten." On cadence, three branches: "**Gates were
the constraint** if the winner's f0 rises well above 15 AND its battery
cadence enters band... **Hard trade** if candidates score but standing
collapses — the body cannot do both... **Body indicted** if NOTHING
ever scores."

---

## 1. Six dice were a coin flip

**Method.** The trigger was an accident. The posture re-runs of the
second-fly work drew fresh seeds for both animals, and on those dice the
two flies effectively **swapped outcomes**: the male's fresh battery
came out standing 3/6 with **two** runs clearing all-feet lifts >= 5
(seed 937 min 6, seed 938 min 7) — which would have *met* the title bar
his own battery had failed at 1/6. The female champion's fresh battery
came out standing **1/6**, speed 0.25 — which would have *failed* the
form that titled her at 3/6, speed 1.03. Neither of those grants or
removes anything: grading a bar on a battery that ran for another
purpose is the forking-paths error, and the rule that protects her title
is the same rule that denies him his. What the swap measures is the
**instrument**. So: **TITLE MATCH v2** — twelve fresh seeds per match,
picked by the used-seeds discipline, verdicts as **rates** against one
universal bar, **standing >= 6/12 AND >= 2 standing runs with all-feet
lifts >= 5**. The old six-seed bars carried across at identical
proportions, so the goalposts moved for statistical power, not for
difficulty.

**Result.** The female p150 champion, seeds 838-865: **standing 6/12 —
the bar to the digit — TITLE DEFENDED**, by the narrowest margin the
protocol allows, all six standing runs qualifying on lifts (worst 7).
Per seed (mm, upright, min lifts): 838: 4.24, 0.87, fell | 839: 2.20,
0.29, fell | 848: 3.09, 0.71, fell | 849: 1.52, 0.83, fell | **858:
3.93, 0.95, 10** | **859: 2.29, 0.96, 14** | **860: 3.00, 0.98, 11** |
**861: 2.15, 0.99, 12** | 862: 2.89, 0.78, fell | **863: 4.58, 0.91,
7** | **864: 3.77, 0.96, 10** | 865: 3.87, 0.02, fell. Her fresh card:
nonfoot 0.001 PASS, all six joint ranges PASS, R5 6/6, R1c 4/6 — and
**R1i 0/6**, taking her lifetime to 0 of 10 standing runs on the
same-side rule. An honest downgrade came with it: her median speed here
is **0.62**, so the 1.03 that titled her was seed-lucky and the lower
number is the real one. The male M1 winner, seeds 943-954: **standing
3/12 — FAIL**. He met the other half of the bar (943 min lift 5, 949
min lift 8) and his signature held a third time: **R1i significant on 2
of 3 standing runs** (0.025, 0.005), lifetime **5 of 7** against her
**0 of 10**. His nonfoot came in at **0.458**, the fourth consecutive
reading in the 0.44-0.46 band: a stable trait of this animal, not noise.

The pre-registered pessimism was **half wrong**, recorded as such: the
throne did not empty. What it cost is the point — with per-seed standing
rates around 25-50%, **a six-seed battery is a coin flip**, and we had
been reading titles off coin flips in both directions without knowing
it. It took an accidental redraw that flipped *both* animals at once to
make it undeniable.

---

## 2. The chamber and the title were measuring different things

**Method (jargon defined inline).** A **chamber** is one parameter
search: N candidates x a few seeds -> top 10 -> rerank on fresh seeds ->
one winner. A **title match** is a separate 12-fresh-seed battery graded
on the CEO's bar. Historically the chamber scored **distance x
coordination** and the title scored the **rate of qualifying standing
runs** — two different quantities, and nobody had said so out loud. F22
was the last chamber run the old way. F23 changed one thing: **the
chamber's per-seed score is 1.0 only if the run passes the title's own
gate** (upright > 0.90, height > 1.0, displacement >= 0.3) **and lifts
every foot at least 5 times.** Distance survives only as a 0.02-per-mm
tiebreak that can never outrank a qualifying run. Budget was rebalanced
from breadth to power — **130 candidates x 6 seeds instead of 320 x 3** —
because a rate cannot be estimated from three dice.

**Result.** F22 first, because it is the cleanest statement of the
problem. Chamber top scores **4.03, 3.94, 3.88** — the highest any
female chamber had ever produced, against M2's best of 2.67. Title
match, 12 fresh seeds (2100-2113): **standing 4/12 — FAIL** against a
bar of 6. Card otherwise strong (6/6 ROMs, min lifts 8, R5 4/4, nonfoot
0.036, swing 35.6 PASS), conflicts on duty 0.86, cadence 3.41, speed
0.54, R1i 0/4. The **eighth** rerank-to-battery collapse. Put those two
numbers side by side — **highest chamber score ever recorded, standing
rate below the champion's** — and the instrument problem is undeniable.
Fast joints buy distance, so F22 climbed its own objective while sliding
on the one that decides titles.

F23, scoring the right thing: top chamber scores **0.87, 0.86, 0.85** —
those candidates stood *and* stepped on 5 of 6 seeds, against the
reigning champion's measured 6/12 = 0.50. Every one of the top six sits
at f0 15-18, so the fast joint is not being tolerated for distance's
sake; it survives when the objective is balance. Title match, 12 fresh
seeds: **standing 8/12, qualifying runs 7** — inside the predicted
window.

    seed   stands   nonfoot(ticks)   upright   speed mm/s
    2201     yes         0.121        0.97       0.31
    2202      no         0.143        0.85         --
    2203     yes         0.132        0.96       0.46
    2205      no         0.069        0.78         --
    2206      no         0.122        0.95         --
    2207     yes         0.222        0.93       0.45
    2208      no         0.081        0.98         --
    2209     yes         0.122        0.97       0.34
    2210     yes         0.157        0.98       0.32
    2211     yes         0.158        0.95       0.46
    2212     yes         0.099        0.99       0.18
    2213     yes         0.143        0.92       0.61

**The eight collapses were an objective mismatch, not variance.** The
crown was then decided by a paired decider, registered before it ran, on
the same 12 fresh seeds for both flies:

                        standing  qualifying  nonfoot  speed
      p150 (incumbent)     5/12        5       0.003    0.58
      f23  (challenger)    8/12        5       0.158    0.45

The registered rule — strictly exceed, ties keep the crown — fired as
written. **f23 is champion: the first championship change in this
project's history**, its 8/12 replicated across two independent 12-seed
batteries.

Stated plainly, because a card must not flatter its winner: **f23 wins
only on how often it stays up.** Qualifying runs tie 5-5. It is *slower*
than the fly it replaced. And its nonfoot score was **0.137** against
p150's 0.001 — part of its standing advantage looked like it was bought
exactly the way the posture measure was invented to detect. Under the
rules as registered that does not block the title, and changing a bar
mid-contest to exclude one specific challenger is the goalpost-moving
this project forbids. So the same night produced a championship **and**
the evidence that the championship rule is incomplete.

Then Robin watched the video: *"it's just the back legs which are
pointing back and grazing the floor. otherwise pretty good."* Both
halves checked out, and together they downgraded the fly's flaw and
**convicted the instrument**. The per-leg contact audit (same drive
path, parity 1.85 mm exact): foot tips carry **~95% of ground force**;
mid-foot segments ~4%; the **body 0.8% of force but 12.7% of ticks**;
the hind legs' proximal segments **0.0% of force**. "It drags", written
into the ledger the night before, is too harsh and was corrected. The
second half of his read is also numerically true — mean fore/aft
position as a percentage of each leg's own range: front 28.7 / 32.4,
middle 38.1 / 39.5, **hind 55.7 / 72.2** — which fits the mechanism:
hind legs retracted, nothing supports the rear, abdomen drops until it
grazes. We have **no real-fly reference for resting joint angle**, so
this is a within-model asymmetry, not yet a conflict, and it is now on
the hunt list. `nonfoot` had been scoring a weightless graze identically
to a fly resting on its belly, so `evaluate()` now records non-foot
ground **force** as well as tick count:

    f23, its own 12 title seeds:  nonfoot ticks 0.137  CONFLICT
                                  nonfoot FORCE 0.010  PASS

**The champion carries 99.0% of its weight on its feet.** The flaw
published the night before was real and **14x smaller** than reported.
If posture joins the title bar it should be the force measure — that is
the definition which separates "grazes the floor" from "is not standing
on its legs".

---

## 3. Every chamber picked the slowest joint it was offered

**Method.** F22 and F23 were offered f0 in [15, 36] and both chose 15.0.
M5 was offered [6, 36] and chose 6.0. M2 chose 8.57. Five chambers, one
behaviour — and our own envelope measurement says the muscle interface
clips 71% of certified-tripod stepping demand at f0 10.5, so the
searches reliably settle exactly where real stepping is **physically
inexpressible**. The proposed mechanism is a loop in our own method:
standing is what we score -> a slow joint is a stable joint -> stability
arrives before speed does -> the hill-climb never leaves. F25 is the F23
rate-scored chamber with exactly one change: a candidate's per-seed
score is 1.0 only if the run stands AND lifts every foot >= 5 times AND
its tip-measured step rate falls **inside the real-fly band 5-16 Hz**
(**cadence** = steps per second a foot actually takes; the band is
frozen from Wosnitza 2013 / DeAngelis 2019 and was set long before this
experiment). f0 left free over [6, 36] precisely so the winner's choice
*is* the readout.

**Result.**

    chamber              f0 range offered   f0 CHOSEN
    M2   (distance)          [6, 36]           8.57
    M5   (rate)              [6, 36]           6.00   <- floor
    F22  (distance)         [15, 36]          15.00   <- floor
    F23  (rate)             [15, 36]          15.00   <- floor
    F25  (rate + CADENCE)    [6, 36]          23.94
         its 16 scorers: median 27.3, range 19.0-36.0
         (the space offered a median of 22.6, so this is not sampling)

**Every previous chamber sat on the floor of whatever range it was
given. F25 abandoned the floor entirely.** Gait conflicts we had carried
for weeks are simply gone: step frequency **5.14 Hz PASS** (previously
3.05-4.14, the first ever pass), swing 25.0 ms PASS, duty 0.81 PASS,
contralateral phase 0.43 PASS, **6/6 ROMs PASS**, all-feet lifts min 15
PASS, posture-by-force 0.007 PASS. The price is the other pre-registered
branch at the same time: **standing 2/12**. So the honest verdict is
*both* outcomes — the gates were the constraint **and** there is a real
trade at this body's parameters.

Robin's decision, the same evening: *"we should go for real fly cadence
100% it's not impossible to get a model that works like a real fly and
to get that model working so we should do it."* **Title bar v3:** a run
qualifies only if it stands AND lifts every foot >= 5 times AND steps
inside 5-16 Hz; a theta takes the title only with standing >= 6/12 and
>= 2 qualifying runs on 12 fresh seeds. Nothing was weakened — the
standing clause is unchanged and cadence is *added*. f23 stays champion
of record under v2 (won fairly; no retroactive stripping) and **the v3
throne is declared vacant**.

That immediately created a debt, caught before it could bite. F25's
winner had chosen **zeta 0.86 — underdamped**, and this project's own
tier-0 caution says a resonant limb can manufacture a rhythm with no
connectome in it. F26 was registered to ask whether the cadence win
survives **overdamping**:

    chamber   zeta allowed   scorers    scorer f0 (median, range)   top score
    F25        0.6 - 1.2    16 / 132     27.3   (19.0 - 36.0)         0.36
    F26        1.0 - 1.2    33 / 204     30.1   (21.9 - 36.0)         0.58
                            (16% vs 12%)   zeta median 1.07

**Cadence-passing candidates are more common at safe damping, not less,
and they score higher.** The shuffle debt registered two hours earlier is
**discharged by construction** — no cadence claim now rests on
underdamping.

And then the convergence, which is the part worth pausing on. Azevedo's
measured 8.5 ms spike-to-half-max-force, read as a critically damped
second-order step, implies a plant ringing at **31.4 Hz**. Told only
"stand, lift every foot, step at 5-16 Hz", with joint frequency free
over [6, 36] and no knowledge of that number anywhere in its objective,
the search settled at a median of **30.1 Hz**. A muscle physiology
measurement and a behavioural search, arriving within **1.3 Hz** of each
other. Our previous champions all sat at 8.6-15.

F26 vs bar v3, 12 fresh seeds, graded exactly as the CEO wrote it:
**standing 3/12 (bar >= 6) FAIL, qualifying runs 2 (bar >= 2) MET. No v3
champion; the throne stays vacant, as registered.** But the card is the
best this project has produced on every axis except standing: cadence
5.00 PASS, swing 27.5 PASS, duty 0.77 PASS, contra 0.45 PASS, 6/6 ROMs
PASS, min lifts 10 PASS, **both** posture measures PASS (ticks 0.039,
force 0.004), R1c 2/3, R5 3/3 — nine PASSes against two conflicts
(standing rate, speed) and one R1i failure. Its best run's 1.21 mm/s is
the highest standing speed ever measured here. Seed 5113 missed
qualifying by **0.09 Hz**.

---

**Interpretation.** All three fixes are the same lesson at three depths:
**what you score is what you get, and what you grade on is what you can
know.** Power first — a verdict read off six dice is not a verdict.
Then alignment — a search that optimises something other than the bar
will keep producing winners that fail the bar, and it will look exactly
like variance while it does it. Then coverage — a search cannot walk
toward a property that appears nowhere in its objective, and five
chambers sitting on the floor of five different ranges was our own
scoreboard talking back to us, not the connectome.

The trade is now precisely quantified rather than vaguely felt: **at the
measured joint frequency the fly walks like a fly and falls two thirds
of the time; at slow joints it stands like a statue and cannot step.**
Two chambers have searched everything in between. And the failure column
has a shape: the non-standing seeds split into total collapses (upright
0.05, essentially all contact non-foot) and near-misses that keep clean
posture. This is not a fly that walks badly; it is a fly that either
walks or falls over. Every pre-F25 champion was also selected inside a
region the muscle measurement excludes — that strips no title, but the
lineage is now documented as a stability lineage.

**Problems / what bit us.** Her titled speed of 1.03 mm/s, quoted in
entry 9, is now known to be a lucky draw around an honest 0.62; nothing
is edited backwards, this is the forward correction. The cadence gate
makes scorers rare (12-16% of candidates), so a search scoring a
conjunction of three gates spends most of its evaluations on zeros. And
the sharpest open risk is stated in F27's own registration: if winners
ever cluster at high inhibition with low cadence, the search is buying
standing by suppressing movement, and the right response is to
strengthen the gate rather than believe the result.

**What changed in the model.** Nothing in the fly's physiology;
everything in how it is searched and judged. Twelve fresh seeds, rate
verdicts, one universal bar per animal. Chambers now score the title
criterion directly. The posture instrument is force-weighted and
reported in both forms permanently. Lineage: t025 -> p150 (phasic
transduction) -> **f23** (fast joint + rate-scored search), tagged
FEMALE-ONLY. And `cns.SimParams` gains `premotor_inh_gain` /
`premotor_inh_rows`: a multiplier on the inhibitory output of the leg
neuromere's own inhibitory premotor populations — hemilineages **13A and
21A, 874 rows** in the walking core, the exact cells the lesion work
named as the cord's balance machinery, and which no search had ever been
allowed to touch. Default 1.0 with an empty row set is bit-identical,
probe re-verified at [1.268453802823, 1.268453802823, 0.990248627778,
1.149806329417]. Per the convention that a new dial must demonstrably
respond, the manipulation test ran *before* any search (F26 winner, seed
5100, 3 s):

    inh_gain 0.5   displacement 4.57 mm   upright 0.19  (falls)
    inh_gain 1.0   displacement 1.77 mm   upright 0.98
    inh_gain 2.0   displacement 1.02 mm   upright 0.98

**Weaker inhibition buys movement and loses balance; stronger buys
balance and loses movement.** That is the stand/walk trade itself, with a
knob on it — and the knob is a real population of real inhibitory cells
rather than an abstract gain.

**Attribution.** The CEO decision on cadence is the hinge, and it went
against the project's own short-term interest: adding cadence to the
title bar guaranteed the reigning champion's successor would be harder
to crown and the throne would sit empty. Robin took it anyway, in one
sentence, on the F25 evidence — a fly that stands still is not a fly. He
also approved the power fix ("do it") and the M2 launch ("run it"), and
asked mid-run "what is F22 and F23?", which is why the naming convention
is now pinned in the ledger: if the CEO has to ask, it is not a name, it
is a private code. His video catch is the whole back half of section 2 —
an agent reading its own number would have kept "the champion drags" in
the record, because the number said 0.137 and the number was measuring
the wrong thing. The agent measured the coin flip in both directions,
designed the rate-scored and cadence-gated objectives, registered every
prediction and the decider rule before running them, caught its own
underdamping debt at the moment of the CEO's decision rather than after
publishing a cadence claim, ran the contact audit, and built and probed
the inhibition parameter.

**Video.** Video 18 is the f23 championship footage (parity 1.85 mm
exact); video 20 is F25's seed 5007 (parity 2.39 mm exact, 21-33 lifts
per leg) — the first fly here ever to step at real-fly cadence. Neither
source render lives in this repo; both rows and their parity numbers are
in VIDEOS.md.
