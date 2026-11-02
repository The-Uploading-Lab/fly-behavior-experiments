---
id: 37
date: 2026-08-31
title: The champion falls over at the corrected knees
tldr: Applying this morning's six measured knee signs to the unchanged champion theta drops it from 11/12 standing to 1/12 and turns it upside down, and a parameter-shape audit ordered the same evening finds 32 of 52 ledger entries are one number standing in for a quantity that varies per element.
era: standard
commit: cd1489e8, ca2645b3
theta: results-r23-theta.json, unchanged, plus signs="measured" in the corrected arm
seeds: 70000-70009, 70011, 70012, already consumed, both arms paired on the same twelve
n: 12 per arm, 24 battery runs; 52 ledger entries audited
wall: not recorded
reproduce: not runnable from main. The absent-sign battery helper is held; the committed artifacts are lanes/A/results-cd1mu-control.json and lanes/A/results-cd1mu-corrected.json, and PARAMETER-SHAPE-AUDIT.md
---
**Question.** Two questions that turned out to be the same question. First,
what happens to the reigning champion when the six inverted knee signs
measured this morning are actually applied to it. Second, after Robin asked at
22:16 how a per-edge quantity ever became a single number, how many other
parameters in this model are one scalar standing for something the animal
varies element by element.

**Predicted (pre-registered, quoted).** Lane A filed the knee arm at 20:35,
before any run, and predicted outcome O2, CORRECTED WORSE: "r23 was searched,
tuned and twice graded under the +1 map; every one of its 25 parameters was
fitted against that geometry, so flipping six channels underneath a fitted
theta should break it." The registration also stated in advance what the
result would and would not mean: "If that is what happens it does NOT show
the correction is wrong, it shows the champion is entangled with the old
signs." The parameter audit carried no directional prediction; it was an
inventory ordered by the CEO.

**Method (jargon defined inline).** A paired A/B on twelve already-consumed
seeds, so the comparison is within-seed rather than a contest between two seed
draws. The control arm is the r23 theta verbatim. The corrected arm is the
same theta with `signs="measured"`, which flips the six femur-tibia pitch
channels found inverted by forward kinematics earlier the same day, whose
24-sign census was completed at 12:4x.
Both arms are marked diagnostic and non-promotable in their result files
because the seeds were burnt; this is measure re-analysis, not a title
attempt. Standing is the v3.2 clause: final-second mean body tilt below 51.68
degrees, height above 1.0, displacement at least 0.3. The audit walked every
`Param` entry in `params.py` against two questions the existing confidence
tiers do not ask, namely whether the quantity varies per neuron, edge, muscle
or joint in the animal, and whether that variation is derivable from data the
project already holds.

**Result (numbers, with variance).** The control records 11/12 standing under
v3.2 and 10/12 qualifying. The corrected arm records 1/12 and 1/12. Zero runs
errored in either arm. Median final tilt moves from 8.97 to 168.08 degrees,
which is upside down. The control's per-seed tilts are tight apart from one
failure: 7.50, 7.88, 8.51, 8.68, 8.73, 8.76, 9.18, 10.24, 10.74, 12.18, 12.21
and 155.64 degrees. The corrected arm's are 25.77, 149.01, 155.64, 155.64,
155.64, 165.56, 170.59, 170.90, 170.91, 170.96, 170.99 and 171.00, so seed
70011 at 25.77 is the single survivor and every other seed inverts. Under the
older legacy standing measure the split is 8/12 against 0/12.

The correction underneath that number is that the earlier record was wrong.
`CHAMPION.json` and `results-r23-theta.json` carry no `signs` key at all, so
every crowned r23 run used the default all-`+1` motor map. The morning's claim
that r23 had been searched with the six inversions is withdrawn.

The audit found the ledger and the runtime are not the same object.
`params.py` holds 52 entries and `cns.SimParams` holds 52 fields, and only 18
names overlap, leaving 34 neural runtime fields with no ledger entry. Of the
active r23 theta's 25 keys, exactly one, `adapt_b`, has a matching ledger
name; the other 24 body, sensory, controller and selector keys are absent. The
model does not import `params.py`, so the ledger's claim to be the model's
registry is false. Of the 52 entries, 32 are quantities that vary by element
in the animal while being represented or described through a shared value: 14
have a partial rule or proxy in held data, 18 wait on named physiology or
anatomy. Twelve are already elementwise, five are genuinely global or frozen
experimental axes, and three are evidence records rather than parameters.

Two of those shape defects are live inside the champion and cheap to test.
The `mn_gains="size"` path gives selected motor neurons resting potentials
from -68 to -48 mV while leaving every `v_reset` at the global -52 mV, which
the default leaky-integrate-and-fire model would set equal. And the raw
42-column motor capacity spans 0.4045 to 5.7926, a 14.32-fold range, before
per-column normalisation divides each column by its own sum and deletes every
between-joint magnitude difference.

**Interpretation.** The two results say the same thing from opposite ends.
The knee arm shows that a fitted theta is fitted to its coordinate convention
as much as to the fly: 25 searched parameters had absorbed six inverted
channels well enough to win a title twice, and removing the inversion removes
the thing they were compensating for. That does not make the measured signs
wrong. It makes the champion's title a record for its exact unchanged bytes
and the sign dependence an open model conflict, which is one more conflict
against measured geometry, not one fewer. A physiologically consistent motor
map now requires a new search over the full relevant space rather than a patch.

The audit generalises the failure mode. A wrong value is visible to a sweep;
an assumption baked into the shape of a parameter is invisible to every sweep
that will ever be run, because the search space cannot express it. `delay` is
the case that provoked the question and it was never untuned: tier 3, bounds
0.5 to 12 ms, and that sweep is what produced the frequency law on 2026-08-08
at slope 5.95 against 6.00 predicted, R-squared 0.9967. What is uniform is
that the model holds one delay slot for every edge in the CNS. The same
applies to `sign_unknown`, where one bit decides the sign of 33,452 neurons,
17.7% of the CNS. The project has already fixed one instance and that fix is
the template: `mn_gain` is 391 scalars defaulted to 1.0 against roughly 100x
measured variation, and the champion runs the rule `mn_gains="size"` instead
of the shared value.

**Problems / what bit us.** The knee arm's seeds were already burnt, so
nothing here can be promoted and the arms are labelled that way in the files.
Both arms substituted the standing gate, which is recorded in the artifacts
rather than hidden. The audit also narrowed the premise that triggered it: the
held BANC files contain whole-neuron cable length, neuron anchors and synapse
coordinates, but not arbor path length from a spike-initiation site to each
synapse, so a distance-derived delay must be labelled a proxy until skeleton
topology, conduction velocity and synaptic latency are available. Two further
corrections landed against the ledger's own text: `muscle_gain` is described
as 17 scalars but has no live runtime consumer, and `mn_gains="size"` is a
searched anatomical hypothesis rather than a solved measurement, since the
existing morphology proxies did not separate the 18 known fast and slow motor
neurons.

**What changed in the model.** No byte of the model, no theta, no title, no
video, no registry value and no seed ledger entry. r23 remains champion of
record for its exact unchanged bytes. What changed is the count of open
conflicts: the sign dependence is one, and the 32 shape classifications name
where the next ones will come from. `PARAMETER-SHAPE-AUDIT.md` is now the
committed list, and the two r23 probes above are BUILD hypotheses requiring
body execution and footage, not accepted repairs.

**Attribution.** Robin, relayed from Telegram at 22:16, ordered the audit and
set its urgency: *"How did delay even get a uniform parameter? The whole point
of parameter search is to change them if they are not certain. Wtf? If there
are more cases like this they need to be solved now."* The instruction also
named what an acceptable negative answer looks like, which is why the audit
reports 12 already-elementwise and five genuinely-scalar entries rather than
only the defects: "If the answer is that most of them are genuinely one
number, say so plainly and give the argument." Lane A wrote the knee
registration with its reading table and its prediction before running, ran the
paired arms, and self-reported that the champion had never carried the signs.
The judge accepted the diagnostic, reproduced the confirmation arm, and
produced the exhaustive 52-entry classification and the runtime coverage
inventory.

**On the entry budget.** This is the fourth entry dated 2026-08-31, one over
the cap of three. The reason, stated here as the contract requires: both
halves landed after the day's three entries were written and shipped, and both
rank inside the selection rule's top tier. The knee arm withdraws a claim
about the live champion and opens a conflict against measured geometry; the
audit is a standing change in how every parameter is examined from now on. The
alternative was to hold them for a date they did not happen on.

Forward link: entry 34 crowned this same theta under v3.2 earlier the same day
and recorded the knee-sign question as an open limitation. This is that
question answered.
