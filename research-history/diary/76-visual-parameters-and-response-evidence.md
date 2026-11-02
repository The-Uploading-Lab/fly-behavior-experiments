---
id: 76
date: 2026-09-09
title: Visual parameters and response evidence
tldr: The visual provenance ledger was incomplete. Six new rows separate anatomical evidence, external fitted parameters, assumed input conversion and the neuron model actually used by the fly. No parameter was validated or changed.
era: standard
commit: the commit this entry lands in, on main
theta: results-wi1-theta.json, unchanged
seeds: none
n: 0 simulations
wall: source audit; no simulation timing claim
reproduce: inspect lanes/judge/2026-09-09-visual-parameter-trust.json and its pinned files
media_note: Parameter provenance audit; no changed behaviour or new clip.
---

**Question.** Robin asked which visual parameters and single-neuron models can
be trusted after reading a discussion of FlyVis's response predictions.

**Predicted.** No prospective scientific prediction or simulation was made.
The question was whether the live sources and calibration labels supported the
claims being made about the visual system.

**Method.** The judge read the primary FlyVis paper, the pinned local model
configuration, the imported tape and the current Lane C input generator. Source
identities are recorded in the audit JSON.

**Result.** The ledger had one incidental FlyVis mention and no dedicated visual
input row. Six rows now distinguish fixed structure, externally fitted dynamics,
conversion to spike rates, the current hand-written correlator, its amplitude
and BANC visual-cell dynamics. The receiving theta uses a FlyVis tape-format
name even when a different generator produced its contents. wi1 itself has
vision clamped. Full source comparison: `lanes/judge/VISUAL-PARAMETER-STATUS.md`.

**Interpretation.** Agreement with a recorded response supports that response
under the tested model and conditions. It does not determine unique parameter
values or validate a different neuron equation and input conversion.

**Problems.** Earlier descriptions conflated distinct visual builds and treated
fitted-parameter provenance as a statement about all of their validation.

**What changed in the model.** Nothing. No parameter, theta, constraint grade,
title or seed changed. The ledger gained explicit source and validation scope.

**Attribution.** Robin asked the question and supplied Mistral's discussion.
The Codex judge traced the code and primary paper and corrected the ledger.

Decision: `exchange/2026-09-09-1329-judge-visual-parameter-provenance.md`.
