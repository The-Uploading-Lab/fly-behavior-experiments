---
id: 83
date: 2026-09-11
title: Null test examples, two shuffles at the champion's settings
tldr: A reader asked whether the wiring had been compared against shuffled connectivity with everything else held the same. At the champion's own 56 settings and on its own twelve title seeds, a degree-preserving shuffle stands frozen on every seed and a cell-type-preserving shuffle lands on its back on eleven, against the champion's 8 of 12. The clips and the counts are on a new public page. The symmetric test, giving each shuffle its own tuning budget, is stage 5 and still not run.
era: standard
video_receipt: publication-receipts/walker_wi1_null_degree_seed30555.sha256-8563b38fb43269bde0635b4b0808d23fab91b8cc631c85705e9190bbcbf3ae82.publication.json, publication-receipts/walker_wi1_null_type_seed30555.sha256-08c3299681a96eeeeddda3fd57f45277612f947a5d0e1aae169ed57079a29984.publication.json
commit: the commit this entry lands in, on main; engine at c7d79be7e
theta: results-wi1-theta.json (sha256 ba18676b65b3db0747532d344cd2a2847ac8e38335bb77d3c34b29e9b7802fe7) plus spike_reg 44, unchanged for every arm
seeds: 30544-30555, the second block of wi1's coronation match, all three arms; clips at 30555; shuffle draw 20260911
n: 12 seeds per arm, one shuffle draw per null
wall: about 10 s per seed on the deterministic Metal engine, one worker; the two clips about 2 min each to render
reproduce: PYTHONPATH=. .venv/bin/python lanes/field/2026-09-11-null-examples/null_screen.py lanes/field/2026-09-11-null-examples/null_screen_30544-30555.json real,degree,type
---

**Question.** Robin forwarded a reader's question on the evening of 10
September: how much does the biological wiring contribute to the walking,
and has it been compared with shuffled connectivity while keeping the
neuron models and the tuning budget the same? He asked for clips of the
champion beside two shuffles with every other parameter unchanged, and a
page on the site to hold them. This entry is the record behind that page,
[`/fly/null-tests`](https://theuploadinglab.ai/fly/null-tests).

**Predicted.** Nothing was registered before the run. This was built as an
example set on request, not as a test, and it is graded as one: the counts
below are a screen at one parameter set, one shuffle draw per null, one
seed block.

**Method.** The champion wi1's theta of record, plus the title match's
spike regularity of 44, on the deterministic Metal engine with exact joint
integration, six seconds per run, seeds 30544 to 30555. Three arms on the
same seeds: the measured BANC wiring; a degree-preserving shuffle (the
post column of the 13,620,865-row edge table permuted, so every neuron
keeps its in- and out-degree and its partners are random; 215
self-connections and 32,631 duplicates made, kept and counted); a
cell-type-preserving shuffle (each connection's target redrawn within the
target's cell type, so type-to-type counts survive; 11,504 groups,
11,352,175 connections rewired, 2,268,690 left alone because the target has
no type label). Both rules are `nulls.py`'s, copied into the screen script
so nothing that builds a simulation is imported on the side. Standing is
the v3.2 clause (tilt under 51.68 degrees, height over 1 mm, at least 0.3
mm travelled); qualifying adds five lifts on every foot and 5 to 16 Hz on
the ten millisecond hysteresis read. The champion arm reproduces its
coronation rows to the digit on all twelve seeds, which is the engine's
determinism doing what the 2026-09-07 merge said it would.

**Result.**

    arm                          stands   qualifies   median travel   median cadence
    measured wiring (wi1)         8/12       8/12         8.95 mm        5.09 Hz
    degree-preserving shuffle     0/12       0/12         0.03 mm        0.00 Hz
    cell-type-preserving shuffle  0/12       0/12         4.21 mm        1.18 Hz

The degree shuffle is upright on every seed (final tilt 2.4 to 3.3
degrees) and never moves: 0.007 to 0.072 mm of travel, no foot lifted on
eleven seeds and two lifts on one foot on the twelfth. The type shuffle
pushes off and lands on its back within the first second on eleven seeds
(171.9 to 173.0 degrees) and on its side on 30552 (105.3); its travel and
its lift counts come from a fly kicking on its back. Two clips at seed
30555, one per shuffle, sit beside the champion's published clip on the
page; receipts and per-seed rows are in
`lanes/field/2026-09-11-null-examples/`.

**Interpretation.** At these settings the recipe is specific to the
measured wiring, and the two nulls fail in different directions: keeping
degree and losing partners silences the cord, keeping the type summary and
losing the individual wiring keeps the drive and loses the balance. That
is the same shape as the 2026-08-14 dual battery, where the real and
shuffled cores separated on coordination and stability rather than on
tempo. What it does not show is that random wiring cannot be tuned to
walk: the shuffles ran the champion's numbers, not their own. The
symmetric experiment, real and shuffled trained under the same procedure
and budget, is stage 5 of the order of work and has not run. The reader's
question was in two halves and only the first is answered.

**Problems / what bit us.** The first render of the degree clip labelled an
upright, motionless fly "FALLS", because the verdict line only knew the
standing clause's pass and fail; it was re-rendered with a three-way label
and the page carries the second render. The 2026-08-14 shuffle control,
where a degree-preserving shuffle of the nerve-cord cut stood and stepped
in band on 5 of 8 seeds at that day's body, is stated on the page so the
new result is not read as if the question had always come out this way.

**What changed in the model.** Nothing. No theta, registry row, bar or
champion record is touched.

**Attribution.** Robin, 2026-09-10 evening: forwarded the reader's question
and asked for "a few videos of this latest walking champion and two
connectome shuffles with other parameters the same" and "one more page to
our company website which is named something like null test examples".
The Field Engineer seat (Claude Code, on the Pro) chose the two nulls,
wrote the screen and the renderer from the judge's champion renderer, ran
the arms, published the clips and wrote the page and this entry. The judge
did not grade this; it is not a title event.

**Amended forward, 2026-09-11 afternoon:** the same-budget half of the question was run the same day; see entry 86.
