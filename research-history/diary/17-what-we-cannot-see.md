---
id: 17
date: 2026-08-19
title: An empty slot is a confession, a zero is a claim
tldr: Robin asked what causal signalling an electron microscope cannot see, and whether leaving it out is just another way of writing zero. Eighteen channels, and the answer inverts twice: the male datasets are better annotated than the female, not worse; and the chemistry we are blind to is a rounding error next to the fact that we simulate 13.8% of the female's synaptic mass. Meanwhile one of our own files had invented a laterality that blinded three of a fly's six legs, because a script read the L in "leg nerve" as "left".
era: standard
born: a zero in a weight matrix and an unmodelled mechanism are different claims that render as the same float
video: walker_mc1_first.mp4
commit: arc 28219c2..HEAD, 2026-08-19
theta: results-f23-theta.json (verbatim, untuned) on the rebuilt MCNS substrate
seeds: MCNS first behaviour 9400-9411 (12 fresh, none reused)
n: 21 research agents; 12 MCNS runs, 6 s each
wall: 2026-08-19
reproduce: .venv/bin/python3 mcns_adapt.py --meta-only   # ten assertions run at the end
---
**Question.** Every connection we simulate was found by cutting a fly
into 40-nanometre slices and photographing them. Gap junctions,
neuropeptides, volume transmission and most of neuromodulation leave no
photographable structure. **How much causation are we blind to, and is
omitting it any different from writing zero?**

**Predicted (pre-registered, quoted).** The brief required each channel
to be classified as one of: "(a) modelled as zero DELIBERATELY, with the
cost measured; (b) modelled as zero because the mechanism is genuinely
negligible for walking on our timescale; (c) modelled as zero SILENTLY,
because nobody asked" — and required "unknown" to be a permitted answer,
"dressing it up would be the worst outcome of this report."

**Method (jargon defined inline).** Twenty-one agents with a literature
budget, every quantitative claim re-checked by a second agent briefed to
refute it, and — the half that mattered — a measurement pass over our own
three datasets asking not what the papers say but what *our* simulator
currently does with each channel.

**Result (numbers, with variance).**

**Eighteen channels.** Five silent zeros nobody had asked about, four
deliberate zeros never costed, six genuinely negligible within a
six-second walking episode and now argued rather than assumed, and three
that are not chemistry at all.

The zero Robin named first — monoamines, the dopamine, serotonin and
octopamine neurons our model multiplies by zero because they act through
slow metabotropic receptors with no fast sign — costs **571 neurons and
0.755% of the female model's simulated synaptic drive.** Real, now
measured, and smaller than it feels.

Then the number that reframes the question. **We simulate 13.8% of the
female's synaptic mass** — 5.86 million of 42.3 million — after scoping
to the nerve cord and cutting weak edges. And upstream of that, only
**18% of BANC's real connections have both partners reconstructed at
all.** The chemistry zeros are a rounding error against the measurement
zero. Any framing that presents 0.755% as our blind spot without that
table is misleading, and ours had been.

Two things in our own house were worse than zero. **4,837 glia sit inside
the simulated core** — 14.7% of it — and 4,221 of them, having no
transmitter, take the excitatory default and are instantiated as
excitatory neurons. They emit 175 synapses, 0.003%, so the cost is
bookkeeping rather than behaviour: "we simulate 33,004 neurons" is wrong
by 14.7%, and part of the female-male size gap is glia, not biology. And
BANC ships a `neuropeptide_verified` column on 7,025 neurons that **zero
lines of code have ever read** — 81 of them in the core, 44 of those
already muted monoamines: cells whose peptide identity we hold, whose
fast sign we set to zero, and whose peptide we never read.

The premise of Robin's question inverted on measurement. On neuron types
the **males are better annotated than the female** — 95.5% and 93.4%
against 76.6% inside the rig core — and better on predicted transmitter
too. What the male lacks is *confirmed* transmitter, and MANC has exactly
zero, which no effort can fix: that column does not exist in the source.

The damage was not empty slots. It was **invented values that nothing
marked as invented.** MCNS names its leg nerves ProLN, MesoLN, MetaLN,
where LN is **leg nerve**. A lost version of our adapter read that L as
*left*, filled 3,915 blank-laterality rows with "left", and wrote the
same mistake into the strings the simulator parses to assign an afferent
to a leg. Measured through the rig's own selectors, before and after the
rebuild:

    proprioceptors per leg   58 / 275 / 289 / 0 / 0 / 0
                          -> 69 / 135 / 138 / 30 / 140 / 151
    load sensors per leg      4 / 4 / 4 / 0 / 0 / 0
                          ->  2 / 2 / 2 / 2 / 2 / 2
    leg chordotonal groups   claw 0, hook 0, club 550
                          -> claw 93, hook 66, club 432

**Three of male #2's six legs were deafferented**, and `claw` and `hook`
— two parameters our own sign test found decisive — were dead knobs on
that animal. A search there would have reported them flat, and the
flatness would have been a fact about a string.

Worse: the adapter in the repository no longer produced the file the
simulator was reading. It would have written a different vocabulary for
motor neurons, matching zero rows in silence. **The male data could not
be regenerated from the code**, and the line that fabricated the
laterality had no source to fix. It was reconstructed by re-deriving each
column from the raw sources and identifying the lost transmitter join by
reproducing the shipped file exactly.

Rebuilt, **male #2 drove the simulator for the first time**. On the
female champion's recipe, untuned, twelve fresh seeds: he **steps on
every one** — all six legs, 9 to 19 lifts each on the best — and **stands
on none**, uprightness 0.006 to 0.808 against a gate of 0.90. He is on
the floor the whole six seconds. That is 0/12 against a bar of 6/12, so
nothing was crowned.

**Interpretation.** The sharpening this deserves is Robin's own framing,
made exact. **A zero in a weight matrix is a claim — "this edge exists
and carries nothing." An unmodelled mechanism is a confession — "no
instrument has looked."** Both render as the float 0.0 and the code
cannot tell them apart. That is the defect, and it is representational
before it is mechanistic. It is also why the male file's fabricated
"left" was invisible for weeks while MANC's guesses, which carry a
`label_guessed` flag, were not.

**Problems / what bit us.** The rebuilt MCNS file carries a guess of its
own: 358 of 591 leg chordotonal subtypes are carried across from the
female's own cross-matches, at a median type purity of 0.73. It is
flagged — 3,963 rows, 1.9% — and it is still a guess. And one verdict on
record, "the champion does not transfer to the third fly", was measured
on the broken file: three legs deafferented, two dials dead, against two
intact flies. That needs re-measuring, not annotating.

**What changed in the model.** The MCNS adapter is rebuilt with laterality
read from the column that covers neurons whose cell body sits outside the
imaged volume — which is to say, the afferents — and never from a nerve
name. Ten assertions now run at the end of it, every one checking what the
rig *selects* rather than what the file contains. The simulator no longer
multiplies "we do not know" by zero: a token that asserts ignorance takes
the searched unknown path, while a real transmitter our vocabulary lacks
stays muted and is counted. And 994 MB of irreplaceable data now sits on a
second machine with a manifest labelling every file RAW, DERIVED,
DERIVED+GUESSED or DERIVED+FABRICATED — because the last category should
not exist, exists once, and is now documented rather than carried.

**Attribution.** The question is Robin's, and so is the framing this entry
is named after: *"is not including them another choice of labeling them
zero?"* He also asked the follow-up that found the adapter bug — whether
the male datasets have empty slots with no guesses — and funded the
literature search that produced the eighteen. The agent measured the
exposure, rebuilt the adapter, and reported that the premise inverted.

**Video.** walker_mc1_first.mp4 — male #2's first behaviour, six hours
after he got three legs' worth of sensing back. Not a champion: he steps
with all six and never stands up.
