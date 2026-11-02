---
id: 2
date: 2026-08-06
title: Counting our guesses
tldr: 48% of the model's numbers are guesses. We measured which guesses the output actually depends on.
era: rough
---
We tiered every scalar in the model by how much we believe it:
measured (448), principled (42), arbitrary (460). Then we perturbed
the live ones and re-ran the whole 188k-neuron simulation to see which
matter.

The winner was embarrassing: the single biggest lever in the model
(196x the noise floor) is `nt_sign_glutamate` — whether glutamate is
treated as inhibitory — a convention inherited from a published model,
not measured, applied to 25,099 neurons. Meanwhile two pure guesses
covering 33,452 neurons turned out to matter less than the difference
between two random seeds.

The lesson that stuck: tier alone tells you nothing — a wild guess the
answer ignores is harmless; a "principled" value the answer hinges on
is a research program.

**Human:** the demand that everything be justified or labelled a guess.
**Agent:** the ledger and the sensitivity runs.
