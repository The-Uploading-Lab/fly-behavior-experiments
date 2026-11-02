---
id: 73
date: 2026-09-08
title: Four projected rows carry predictable structure
tldr: With 4,999 training-label permutations, four of twelve projected within-leg order rows beat the same fixed nonlinear predictor's shuffled-label error with Holm-adjusted p below 0.05 in both real-data split directions. The strongest gains are 8.37 and 9.07 degrees for the right-hind middle-to-distal row. Held-out absolute errors across the four rows remain 14.47 to 76.99 degrees, so the result establishes repeatable predictable structure, not reconstruction or a reason to remove a physical constraint.
era: standard
commit: the commit this entry lands in, on main
seeds: no CNS run and no model seed; 3,000 fixed real trajectories
figure: fig_r554_recoverable_2026-09-08.png
---

03:48 EEST.

## Question

Do any projected within-leg order rows remain recoverable from the current tip
and tripod summaries after enough label shuffles to apply a twelve-row
familywise test?

## Predicted

The learner, k=31, four features, missingness handling, two split directions and
permutation streams were committed before the higher-resolution replay. A row
would count as recoverable only with positive error gain and Holm-adjusted
p<0.05 in both directions.

## Result

Four of twelve rows clear that bar. The error improvements over shuffled
training labels in the registered and reversed directions are:

- R3 middle-to-distal: 8.3743 and 9.0693 degrees.
- L3 middle-to-distal: 4.8370 and 7.6788 degrees.
- L1 proximal-to-middle: 7.1450 and 6.3646 degrees.
- L2 middle-to-distal: 0.4822 and 0.3476 degrees.

All eight adjusted p-values are below 0.05. The largest is 0.0414 for the
registered L1 proximal-to-middle row. The registered direction clears four
rows and the reversed direction clears six. R2 middle-to-distal fails the
two-direction bar because its adjusted p-values are 0.088 and 0.0054.

## Interpretation

The current tip and tripod summaries contain repeatable predictive structure
for four projected-order rows. The prediction errors remain large: 54.54 and
54.94 degrees for R3, 63.71 and 63.52 for L3, 76.39 and 76.99 for L1, and 14.47
and 14.83 for L2. Those four physical observations therefore remain useful
constraints. The calculation does not establish full redundancy, statistical
independence or information gain.

The other eight rows have no repeatable familywise recovery. The next model
decision comes from paired interventions that move these measurements, not
from fitting another real-data predictor.

## Attribution

Lane B froze and ran the 4,999-permutation replay. The judge repeated the
result, applied the registered decision rule and retained all twelve physical
rows in the walking coverage gap.

## Parameter rank in the same pass

Lane D then compared wi1's 6 ms activation time on the six CTr-pitch joints
with the shared 12.2629 ms value on spent seeds 60524-60526. Zero of twelve
projected-order rows met the frozen parameter-link rule. The two R2 rows moved
53.007 and -95.271 degrees on seed 60526, but that was their only paired-readable
seed against a required minimum of two. Wi1 stood on 60524, while the shared-
time arm stood on 60525 and 60526, so no seed had both bodies standing. The
specified activation-time contrast was dropped without a model change.
