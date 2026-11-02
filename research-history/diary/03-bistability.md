---
id: 3
date: 2026-08-09
title: The network that was either dead or screaming
tldr: One global synapse gain made the network bistable. A per-neuron balance rule opened the regime a brain lives in.
era: rough
---
Every setting of the global synaptic gain gave one of two networks: one
that dies the instant input stops, or one that latches at ceiling
forever like a metronome (spike-time variability 0.06 — a clock, not a
brain). At one gain value, 4 of 12 seeds latched and 8 died: both
states coexist and noise picks. No fine-tuning could find the middle,
because there was no middle.

The fix that worked was homeostatic: each neuron scales its own
incoming inhibition to hit a target balance. The middle regime went
from 0% of runs to 36%, firing at 7.5 Hz with brain-like irregularity.

A correction is preserved here on purpose: we first wrote that
per-class gains were the fix. Measuring before building showed the
imbalance lives WITHIN cell classes, not between them — the fix we'd
already written down would not have worked. This kind of preserved
correction later became diary policy.

**Human:** patience while the network screamed.
**Agent:** the sweep, the wrong first fix, the measurement that caught
it, the homeostatic rule.
