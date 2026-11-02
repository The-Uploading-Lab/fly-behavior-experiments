---
id: 71
date: 2026-09-08
title: The head row was the right-hind tarsus
tldr: FlyGym fuses the static head out of its compiled MuJoCo model. The trajectory exporter retained that name with body ID -1, so NumPy filled the head row from the last body, the right-hind tarsus. The scorer then used that moving foot as the fly's heading. Replacing it with the retained eye midpoint changes the known-good CPG benchmark from 3 of 7 fields inside the real-fly range to 2 of 7 and exposes posture variability as a conflict. The champion card uses captured body orientation and is unaffected.
era: standard
commit: the commit this entry lands in, on main
seeds: no new seed; frozen 6 s FlyGym CPG trajectory and spent wi1 seed 108520
figure: fig_fused_head_adapter_2026-09-08.png
---

00:45 EEST.

## Question

Why were five of the twelve projected within-leg links unreadable on the
standing wi1 replay, and did the same coordinate path still support the
documented known-good CPG comparison?

## Predicted

The selected R555 read was expected to separate missing target cadence from
joint tracking and body-axis projection. The judge instruction was: "Separate
a body-axis projection loss from absent wi1 command before any theta change."
No adapter failure was predicted.

## Method

The existing 6 s CPG trajectory and spent wi1 seed 108520 were read through the
same 32-landmark adapter used by R554. Body names were checked against their
compiled MuJoCo IDs before the kinematic surfaces were calculated. The repair
uses the midpoint of the retained left and right eye bodies for the horizontal
heading. Fifteen focused tests cover the R554 reader, R555 localization, an
arbitrary poisoned `c_head` row and frame invariance under global rotation and
translation.

The final wi1 read held each leg's cadence bin fixed and followed the 42 joint
targets through achieved actuator coordinates, MuJoCo forward kinematics with
a fixed body, the recorded free root, the recorded non-leg hinges and the full
pose. Twelve observed poses supplied a 42-coordinate central-difference
Jacobian. No altered pose advanced physics or entered the CNS.

## Result

`c_head` is byte-identical to `rh_tarsus5` on all 60,000 CPG frames. Its
thorax-relative vector points 159.53 to 168.62 degrees away from the eye
midpoint heading. The retained eye midpoint stays 0.2575 to 0.2603 mm from the
thorax.

The corrected CPG remains readable on 12 of 12 projected-order rows, with 9 of
12 inside the real intervals instead of 8. The wi1 replay remains readable on
2 of 12 rows, but they are both on L3 rather than R2. R554's real-fly split
still passes 11 of 12 rows, its fixed-membership disruption moves 12 of 12, and
all seven scores remain exactly invariant to replacing the 18 non-tip leg
landmarks.

The known-good CPG benchmark changes from 3 of 7 fields inside the real-fly
central 90% to 2 of 7. Posture variability changes from 0.032681 rad, inside
[0.025567, 0.085916], to 0.002051 rad, below the measured range. Tripod contrast
and left-right asymmetry remain inside.

All five unreadable wi1 links carry target cadence. Same-leg coordinate
composition cancels three projected middle links. Adding the recorded root
attitude removes one left-front proximal and one left-middle projected link.
No link is lost through actuator tracking, and the 18-by-42 Jacobian has zero
cross-leg terms.

## Interpretation

The canonical CPG control has an additional measured conflict: its abdomen to
thorax angle varies too little. The earlier in-range value measured right-hind
foot motion in the coordinate frame, not body posture. The correction does not
regrade wi1 because the title card derives heading from the captured body
quaternion rather than `adapt_and_score.py`.

The one wi1 replay localizes missing projected order after the joint target. It
does not estimate a population rate and does not justify a theta change.

## Problems

The exporter preserved a documented body name after MuJoCo fused that body out.
The internal index was negative, and ordinary NumPy indexing made the failure
look like valid finite data. The old Lane D adapter-equivalence checks used the
same false heading on both sides, so numerical agreement did not validate the
quantity's identity.

R556 reconstructs the historical comparison and restores numerical parity only
after both paths use the eye midpoint. The source exporter can still emit the
fused name, and four male-body audits compare whole position arrays. Their exact
changed-element counts may count the right-hind tarsus twice; their qpos and
foot evidence was not affected, but those counts remain to be re-read.

## What changed

The shared adapter now uses the retained eye midpoint and rejects a non-finite
or zero-length heading. The documented CPG count is reduced from 3 of 7 to 2 of
7. R554's simulated rows are replaced by the corrected table. The champion,
theta and title bar do not change.

## Attribution

Robin selected walking observability and prohibited a new seed. Lane E built
the R555 kinematic reader, exposed the fused-body fault before committing its
first result, repaired the adapter and returned the corrected replay. The judge
reproduced the seven-field change on the frozen CPG trajectory and added the
frame-invariance gate.

## 03:06 integration addendum

The source guard found one more live route only after the five branches were
combined. Lane E's branch contained thirteen body-position consumers. Main also
contained Lane A's new CD2-GZ reader, which reconstructed poses with
`data.xpos[body_ids]` and therefore read the fused head through ID -1. The
eye-midpoint adapter keeps CD2-GZ's projected-order values independent of that
row, but the executable path still bypasses the guard. The consumer-surface
closure was held and returned to A and E.

Removing the alias from retained male artifacts changes the deterministic
body-position count from 81,972 to 80,784. The actual-FeCO cut-30 and cut-10
counts change from 80,937 to 79,764 and 81,765 to 80,580. Each non-inert verdict
still has changed qpos and foot-tip values, so only the whole-array counts
change.

The same pass tested whether projected within-leg order was already recoverable
from tip and tripod summaries. A fixed nonlinear predictor gave five nominal
rows in one real-data split and seven after swapping the halves. All first five
repeated, but 200 permutations cannot clear the twelve-row familywise test: the
smallest adjusted p is 0.059701. That result sets the next computation, a
higher-resolution fixed replay, and does not add a metric or title row.
