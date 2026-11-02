---
id: 72
date: 2026-09-08
title: Five nominal rows became zero
tldr: A fixed nonlinear predictor recovered five of twelve projected within-leg order rows from existing tip and tripod summaries in one real-data split, and all five repeated when the halves were reversed. Two more appeared only after reversal. With only 200 label permutations, however, the smallest raw p-value was 0.004975 and the smallest Holm-adjusted p-value across twelve rows was 0.059701. No row clears the familywise bar, so the result is partial exploratory recoverability rather than evidence that the new quantities are redundant.
era: standard
commit: the commit this entry lands in, on main
seeds: no CNS run and no model seed; 3,000 fixed real trajectories
figure: fig_r554_familywise_2026-09-08.png
---

03:06 EEST.

## Question

Does projected within-leg order contain structure that the existing distal-tip
phase and tripod summaries cannot recover?

## Predicted

The registered fork treated held-out prediction above a training-label
permutation floor as redundancy evidence, and prediction at the floor as no
detected nonlinear redundancy. The learner, features, split and 200
permutations were fixed before fitting.

## Method

The input contained 3,000 real trajectories. Four existing features entered a
k=31 circular-neighbour predictor: tripod contrast, step frequency, gait signal
quality and contralateral distal-tip phase locking. Each of the twelve R554
leg-by-adjacency rows was predicted separately. Training labels were shuffled
200 times while test membership and missingness stayed fixed. The same
calculation was then repeated with the two real-data halves reversed.

The judge read an empirical lower-tail p-value from each retained permutation
distribution and applied Holm correction across the twelve rows in each split
direction. This familywise read was not part of the lane's nominal fifth-
percentile decision and therefore grades the result rather than changing it.

## Result

Five of twelve rows beat the nominal floor in the registered direction. Seven
beat it after the halves were reversed. All original five repeated; the two
additional rows were split-specific.

The strongest raw empirical p-value in either direction was 1/201, or
0.004975. Multiplication by twelve at the first Holm step gives 0.059701. Zero
of twelve rows therefore clears adjusted p<0.05 in either direction at this
permutation resolution.

## Interpretation

Existing tip and tripod summaries carry some repeatable information about five
projected-order rows. The calculation does not yet establish that any row is
recoverable under a familywise test, and it does not establish full redundancy.
Statistical independence and information gain remain open.

The shortest next computation keeps every modelling choice fixed, increases
the permutation count to 4,999 and requires the same row to clear Holm p<0.05
in both directions. No simulation or new metric is needed.

## Integration catch

The same merge exposed a separate source failure. Lane E's body-position audit
found thirteen consumers on its branch, but integrated main adds Lane A's new
CD2-GZ reader as a fourteenth. That reader indexes a body-ID vector containing
-1. The repaired eye-midpoint adapter keeps its scientific projected-order
values independent of the false head row, but the path still bypasses the
guard. The complete consumer-surface claim was held until A repairs the reader
and E reruns discovery on integrated main.

## Attribution

Lane B froze and ran the nonlinear prediction in both split directions. Lane E
built the compiled-body guard and corrected the retained male whole-array
counts. The judge applied the twelve-row familywise read and found the
integration-only fourteenth consumer before accepting the closure.

## Follow-up

At 03:48, the pre-registered 4,999-permutation replay resolved the numerical
floor. Four rows clear Holm p<0.05 in both directions, with held-out absolute
errors of 14.47 to 76.99 degrees. Diary entry 73 records that narrower result.
