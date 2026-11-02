---
id: 20
date: 2026-08-21
title: The uncrowned fly
tldr: The strongest nomination this project has ever made was closed tonight without a crown — not because the fly got worse, but because the bar it was being measured against got two things it was missing. Robin adopted v3.1: a crown now needs TWO independent 12-seed batteries, and title matches run at MEASURED afferent spike regularity (ISI CV 0.15) instead of the Poisson dither every champion in the lineage was selected under. LOADONLY-f36 passed three Poisson batteries and stood 3/12 on the one measured-noise battery. The throne stays VACANT. Separately, the third seat's first verification pass came back clean, with a zero-survivor block on 50 released vision checkpoints.
era: standard
born: a bar a single lucky battery can flip is not a bar — and the noise the fly is graded under has to be measured too
commit: d9bac1f (ruling) + 619f5c6 (Lane C verification) + 4d8425b (cycle credit)
theta: LOADONLY-f36 (Lane A load-only nominee); f23 as paired control
seeds: judge title match 30012-30026; N1 exploratory 30027-30041; N1 anchored 30029-30052 block
n: five independent 12-seed batteries on the nominee (60 runs), plus 72 paired N1 dose runs
wall: 11:29 nomination to ~21:45 ruling
reproduce: exchange/2026-08-21-judge-title-match-loadonly-f36.{md,json}; CONSTRAINTS.md "N1 ANCHORED RESULT" and "RULING (Robin, 2026-08-21 ~21:45)"
---
Run metadata: ruling committed d9bac1f, recorded in AGENTS.md as
TITLE BAR v3.1; Lane C's first verification pass 619f5c6; the cycle
credit 4d8425b. The previous entry (19) left this exact question open
— "the judge... has recommended ADOPT to Robin" — and this is how it
closed, hours later the same evening.

**The question was never whether the fly was good. It was whether
the bar could tell.** LOADONLY-f36 was nominated by Lane A at 11:29,
the first v3 nomination in project history. It then produced one of
the strangest records we hold: on three independent 12-seed batteries
run under our usual Poisson input noise it MET the bar (lane
discovery 8/12 standing + 2 qualifying; lane confirmation 8/12 + 5;
the N1 exploratory Poisson arm 6/12 + 4). On the judge's own 12-seed
title match, on clean collision-free seeds 30012-30026, it stood
**3/12** against a bar of 6 — a FAIL — while still putting **3**
qualifying runs on the board. Pooled over five independent batteries
the standing rate is **31/60 = 0.52**, which is to say it sits
almost exactly ON a bar of 0.50. P(<=3 of 12 | p=0.528) = 0.050: the
judge's draw is at the unlucky edge of consistent and the lanes' two
8/12s are at the lucky edge. Nobody measured wrong. **A 12-seed match
cannot decide a rate that sits on its own threshold** — the six-seed
coin-flip lesson of 2026-08-18, recurring one level up.

**The second thing the bar was missing is noise it could defend.**
The N1 mechanism (registered before any number existed, with the
regularity parameter pinned to measured literature values whatever
they turned out to be) delivered its anchored run at **CV 0.15**
(shape 44; Fisch 2012 locust chordotonal homology import —
*Drosophila* itself is unmeasured, and that is stated in the anchor
report rather than hidden). The pre-registered INVERTED reading
fired: standing **falls** under measured input regularity, in both
thetas, -3/12 paired.

    theta          Poisson        CV 0.15 (measured)
    f23            12/12          9/12
    LOADONLY-f36    6/12          3/12

The interpretation was armed before the run: our afferents are
strongly suprathreshold, so stochastic facilitation is not the
default explanation — **model selection adapted to Poisson harshness
is.** Every champion and chamber in this lineage was selected under
noise 5-100x harsher than any measured arthropod afferent, and their
balance partially rides on that dither.

**Robin ruled at ~21:45, one phrase: "sounds good"** — to both of the
judge's recommendations. TITLE BAR v3.1, effective for all future
title matches:

1. A crown requires **TWO independent 12-seed batteries**, each
   meeting the v3 clause (standing >= 6/12 AND >= 2 qualifying) on
   its own.
2. Title matches and confirmations run at the **measured** afferent
   regularity: `spike_reg` k=44 added to the arm's theta at battery
   time. The theta of record stays the theta; the noise model is a
   bar CONDITION.

**Consequences applied the same night, including the ones that
cost.** The LOADONLY-f36 nomination is **closed without a crown** —
its three passing batteries are Poisson-conditioned, and its one
measured-noise battery stood 3/12. The judge's own earlier proposal,
a pre-registered 24-seed Poisson decider, is **superseded and
cancelled**: a decider run under the wrong noise model decides
nothing. The throne stays **VACANT**. Chambers already search at CV
0.15, so the first v3.1 champion will be the first fly in this
project both *selected* and *titled* under measurement-supported
input statistics.

**Nothing is stripped and nothing is re-graded.** The three Poisson
batteries stand permanently as Poisson-conditioned measurements,
tagged as such. And the finding the nomination actually carried is
untouched: the sensing contrast (load-only >> double-drive, **7-9 vs
1 per 24**) is robust, and the cadence property survived every
battery — every standing run in all three Poisson batteries stepped
inside the real-fly band (judge card 5.00 / 5.55 / 5.73 Hz, minimum
foot lifts 19-21), and under measured noise its median cadence was
still in band (4.96 / 5.09 Hz). **When this fly stands, it walks at
fly cadence.** No previous theta has ever done that once; this one
does it every time it stays up. What it cannot do reliably is stay
up: near-misses cluster right at the gate (uprightness 0.89, 0.89,
0.90, 0.85 on four failing seeds, with displacement 1.7-4.7 mm on
**all twelve** — the fly moves every time and teeters).

**Why this is the right way for it to end.** Lane A proposed the
two-battery rule knowing it would cost Lane A its own candidate.
The judge recommended CV 0.15 knowing it would push its own nominee
below the line. The guard written into the N1 registration said it
in advance: a value that hurts our fly is adopted exactly like one
that helps, because *smoothing noise until a title passes would be
goalpost-moving in a biology costume*. Tonight was the first time
that guard was tested with a crown on the table, and it held.

**The third seat's first verification came back clean.** Lane C
(vision, on Codex, in existence since 16:24 this afternoon) submitted its first work
for judging and the judge re-ran the V1 calibration under its own
invocation: **bit-identical** to the committed artifact — DSI
0.6273615510547327, preferred direction 56.235876698617695, median
correlation 0.8878353, 2.9 s cached. Artifact-packet agreement was
exact across V2 (32/32, 30/32, 8/8, 7/8) and V3 (11/3/0 of 50). The
V4 result is the interesting one: an exact six-speed reanalysis of
the released FlyVis 1.2.0 checkpoints gives **8 of 50 passing all
eight DSI-polarity inequalities, 2 of 50 passing all eight +-45
degree cardinal sectors, the two sets disjoint — 0 of 50 joint
survivors.** That is scoped precisely as what it is: an engineering
block under **Lane C's own added gate**, explicitly not a refutation
of Lappalainen et al. or Maisak et al. Landed in the ledger with that
wording verbatim. Consequence: Phase 3 (BANC integration) stays
blocked; Phase 2 proceeds by simultaneous task + neural-data ensemble
training, to be registered with its freeze list before any training
run. One flag carried openly: the V4 gate logic was audited at count
level only, and a code-level read happens before any claim built ON
that gate enters the ledger.

**A new scoreboard entry that is not about the fly.** Robin added
cycle credit: every cycle report now names which lane provided more
value that cycle, **value-weighted** (false beliefs removed, conflicts
opened or closed, programme redirections) rather than volume-weighted.
The first award went to **Lane A** — for a certified false PASS
removed, three registry conflicts opened, the bar-instability proof,
and the rule proposal that cost it its own candidate; with Lane B's
specific excellence named beside it (an entire night programme
re-confirmed in-block with no silent re-promotion, the skate ratio on
both animals, ~20 honest COLLAPSED verdicts). Praise is part of the
scoreboard, and so is the comparison being written down.

**Attribution.** Robin — the ruling ("sounds good", ~21:45), which
took both recommendations including the expensive one, and the
cycle-credit rule. Lane A — the nomination, and the two-battery
proposal that closed it. The judge — the failing title match on its
own seeds, the N1 exploratory dose and anchored run, both
recommendations, the Lane C verification pass, and the retraction of
its own 24-seed decider proposal once the noise question superseded
it. Lane C — the FlyVis integration block and a first packet that
reproduced bit-exactly on someone else's machine invocation.

**Open, and named so tomorrow cannot pretend otherwise.** Where trial
identity lives once afferent timing is near-deterministic is
registered but unanswered (candidates: renewal phase offsets, contact
solver sensitivity, initial pose) — at near-deterministic input the
across-seed spread persists, so something deeper than afferent spike
timing is deciding which runs stand. And the anchor report's new
falsifiable signature is now on the instrument hunt list: real fly
variance is low-dimensional, shared and speed-dependent, while
private per-leg noise predicts the opposite.
