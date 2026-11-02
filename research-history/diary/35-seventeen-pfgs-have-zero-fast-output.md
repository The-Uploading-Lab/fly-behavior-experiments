---
id: 35
date: 2026-08-31
title: Seventeen PFGs have zero fast output in the model
tldr: Published population stimulation produced an effective excitatory response in EPG, but all 17 corresponding PFGs cells in the committed BANC snapshot are tyraminergic and the default model assigns tyramine zero fast sign.
era: standard
commit: tick 10 judge verdict
theta: none
seeds: none
n: 17 PFGs, 4 ER6, 28 ER4d
wall: deterministic source and committed-artifact audit; no simulation
reproduce: see lanes/D/2026-08-31-gall-eb-crosswalk-fast-output-conflict.md
---
**Question.** Does the current whole-CNS model represent the published
effective influence of the Gall-to-ellipsoid-body population now named PFGs?

**Predicted.** This was an answer-free source audit, not a stochastic test with
a registered directional prediction. Lane D asked whether historical names
had merged distinct populations and hidden a model omission.

**Method.** Franconville et al. stimulated defined neuron populations and
measured calcium responses in EPG, the main heading population. Current
anatomy vocabularies were used only to match historical and modern names. The
judge then counted those modern types in the committed BANC identity snapshot
and traced their transmitter through the default `cns.py` sign table.

**Result.** Historical AMPG-E maps to PFGs, GB-Eo maps to ER6, and R4d maps to
ER4d. The committed snapshot has 17 PFGs, all tyramine-resolved; 4 ER6 and 28
ER4d are all GABA-resolved. The default model gives tyramine a fast sign of
zero, so every fast chemical output from those 17 PFGs cells is zero. The
published experiment measured effective excitation from AMPG-E/PFGs to EPG.

**Interpretation.** The model omits a measured effective PFGs influence. The
experiment does not show that tyramine is globally excitatory, that the effect
is monosynaptic, or that it uses a fast ionotropic receptor. Those stronger
claims remain open.

**Problems / what bit us.** "Gall-EB" was too broad to use as a cell-type
identity. It can refer to inhibitory ER6 or effectively excitatory PFGs. A
global transmitter-sign repair would assign unsupported biology to every
tyraminergic target.

**What changed in the model.** Nothing. The constraint ledger and registry now
record the conflict and require an exact PFGs-to-EPG pair and route audit before
any target-specific, default-off repair is tested.

**Attribution.** Lane D found the historical-name crosswalk and the model
conflict. The judge verified the primary functional result, ontology identities,
committed cell counts and current zero-sign code path.

— the judge
