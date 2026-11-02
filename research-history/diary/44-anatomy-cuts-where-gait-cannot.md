---
id: 44
date: 2026-09-02
title: Anatomy cuts where gait cannot
tldr: Lane D's posture card found the selection's front legs sweeping 26 degrees short of measured fly anatomy at 04:41; by 08:23 both range conflicts were closed by two verified mechanisms and the left-turn margin had widened 36-fold. In between, a lane-versus-lane dose war ended with both sides reproduced to the digit on one canonical engine, the first search candidate survived its out-of-sample test, and the judge was corrected twice on the record. Every measured row the walking battery holds now passes on the selection.
era: standard
video: selection-rom-closure-seed60341.mp4
commit: 6dd42d9e1
seeds: D 60316-60355, B 80267+, A 200470-200481
n: two ROM closures, one confirmed search survivor, one resolved coordination conflict, two judge corrections
wall: one night shift, 04:41-08:54
reproduce: git log --format='%h %s' eaef30866~1..6dd42d9e1
---

**Question.** What happens when a fresh anatomical measurement disagrees
with a body that already passes its gait card?

**Method.** Lane D ran the battery's posture and Haustein joint-range
rows against the live selection, the first time those rows were graded
here. The conflicts went into the registry within minutes; three lanes
then attacked the same dial from different directions, and the judge
replayed every disagreement on one canonical engine before ruling.

**Result.** The front femur-tibia swept 55 degrees against the fly's
measured 81 to 112, the front coxa-trochanter 67 against 72 to 110 — on
a body whose gait card is clean, which is the point of hunting
constraints the model has never been graded on. Front-leg step gain 1.6
closed the coxa-trochanter row the same morning. The femur-tibia row
resisted: every template-level dose that reached the range destroyed the
static window (-29 degrees per second at the worst), and one lane's
"close this dose" at gain 1.4 contradicted another's "push further" at
1.3 and 1.6 until the judge merged the dial into the canonical engine
and reproduced BOTH lanes' numbers to the digit — the dose landscape is
genuinely rugged, with two failure modes sitting between passing
neighbours a tenth of a gain step apart. The mechanism that closed the
row: gate the extra gain to ticks where the image moves, so the static
segment keeps the selection's amplitude. Four-seed gate 4 of 4, femur-
tibia at 86 degrees, five clear of the floor, left-turn margin 13.5
degrees per second where the morning started at 0.376. Separately, the
first clause-shaped search candidate survived twelve fresh paired seeds
(p = 0.0327, standing 9 to 11 of 12, a 26% displacement cost taken as
the registered trade), and the selection passed all three Cruse
coordination rules 6 of 6, resolving a female coordination conflict
recorded against the old body in August.

**Interpretation.** A measured constraint the model had never been
graded on deleted a region of parameter space the gait card could not
see, and the deletion was productive within four hours. The rugged dose
landscape is a property of the deterministic vision loop worth its own
question. The two species-level conflicts now standing (motor neurons
silent at rest against a measured ~30 Hz; no recruitment ordering under
drive in either animal) both point at one missing mechanism: tonic
excitatory input at rest.

**Problems.** The judge was wrong twice and is on the record for both:
a hold claimed no search scorer was committed when it had been for a
day, and an arbitration note called a root-committed module missing.
Both holds stood on their deciding point — runs with no committed
runner and no engine hash — and both misstatements were caught by a
lane's disagree-first line, which is the system working. A first Cruse
replay compared a committed file with itself before the guard caught
it; the rerun was fresh and matched. Receipts now require an engine
hash fleet-wide.

**What changed in the model.** `selection-theta.json`: template front
gain 1.6, moving-tick front gain 2.8, brake deadband 1.5. Registry rows
29 and 30 CLOSED; coord.R1i resolved on the selection; rows 27 and 28
stand as the sharpest open target. Video:
https://theuploadinglab.ai/fly/selection-rom-closure-seed60341.mp4

**Attribution.** Robin's high-agency directive is one day old and this
is what it bought: lane D found the conflicts, named the decoupling
mechanism, and killed one of its own closures at the gate on the way;
lane B built the gain dial, measured the 1.4 failure honestly on a
dirty tree, and took the process correction; lane A shipped the first
out-of-sample search survivor and corrected the judge twice with
evidence; the judge merged two engines, arbitrated one dose war, and
folded two closures the same morning they were found.

cites:: lanes/judge/2026-09-02-front-gain-arbitration.md, exchange/2026-09-02-2010-laneD-judge-packet.md, exchange/2026-09-02-2230-laneD-judge-packet.md
uses-data:: selection-theta.json, CONSTRAINT-REGISTRY.json, lanes/D/2026-09-02-front-row29-gate-result.json
