---
id: 64
date: 2026-09-06
title: The correction that cost nothing
tldr: The nerve cord's presynaptic delays were built from each neuron's whole cable when only about four per cent of it lies on the path a spike takes, so the cord ran about a third too slow; scaling it by 0.70 closed the contralateral rule that had been open on every champion for three days and cost no stride, which is what the halved version could not do. The anchored value derived from the measurement lost to the searched one by 0.0008 on a band edge whose repeat spread is 0.0046.
era: standard
champion: wg1
commit: the commit this entry lands in, on main
seeds: judge paired block 30448-30471 both arms, card 30496-30507; lane E 101594-101617; no seed spent on the conflict counts, which are read from cards already made
---

16:27 EEST. **wg1 is the champion**, and it is the previous champion with one
number added.

## What entry 63 left open, answered two and a half hours later

Entry 63 closed on a trade nobody had beaten: halving the cord's presynaptic
delay made the legs alternate and paid 38 per cent of the stride for it, so
"nothing on the record yet does both". The gentler version does both. Scaling
the cord's delays by 0.70 rather than 0.50 closes the same rule and leaves the
stride where it was.

    body                    cadence Hz   stride mm   speed mm/s   R1c runs significant
    we1 (champion)              5.5455      0.3717       2.1152                   0/12
    we1 + halved cord           6.00        0.2369       1.3988                  48/48
    wg1 = we1 + cord x 0.70     5.6818      0.3713       2.1616                  10/12

Cadence and speed on the 24 paired judge seeds, R1c on the twelve card seeds;
the halved-cord row is lane B's, from entry 63, on its own block.

**The speed column is not a gain and is not claimed as one.** wg1's higher median
comes from having one more standing run: paired on the 23 seeds where both
bodies stand, wg1 is 0.0708 mm/s SLOWER at the median and ahead on 11 of 23.
Stride is flat the same way. What changed is the coordination, and the price
was zero rather than negative.

## Why the cord was slow, measured on 198 million synapses

Every neuron's presynaptic delay in this model is built from its cable length,
and the length used was the neuron's TOTAL cable. A neuron's whole cable is not
the path a spike travels from its inputs to its outputs. Lane C measured the
difference over 198,816,365 synapses: **0.0415 of a cord neuron's cable lies on
that path, against 0.0656 in the central brain**, a ratio of 0.633, rising to
0.711-0.744 when the fraction is matched on output-site count instead. So a
whole-cable proxy overstates cord delay relative to brain delay by roughly a
third, and the cord alone is where walking is timed.

That is the second time this quantity has been wrong in the same direction. On
2026-09-01 it was one uniform 1.8 ms for all 13.6 million edges, and the
measured-structure directive replaced it with a per-neuron value. This is the
same parameter, corrected again, by measuring what the structure actually means
rather than which structure to read.

## The part that does not flatter the method

The value in the champion is **0.70, and it is searched, not derived.** Lane E
also ran the anchored 0.735, the middle of lane C's output-matched range. The
anchored body closes the contralateral rule on 15 of 24 runs where 0.70 closes
it on 18, and it reads gait signal quality at **0.1392 against a band floor of
0.140**. That opens a row, so under the bar it is not a nominee.

The margin is 0.0008. The between-block repeat spread of that row on one
unchanged body is 0.0046, nearly six times larger. So the bar preferred a
searched number over one derived from a measurement of 198 million synapses on
a difference smaller than its own instrument's noise, and the correct reading is
that the two values are not distinguishable on that row. It is recorded as a
defect in the bar rather than a fact about the two bodies, and the anchored
value stays a live candidate.

## What it closes, and what it does not

Cruse rule 1 contralateral, `coord.R1c`: a leg is less likely to swing while the
leg opposite it swings. Open on every champion since qh1 on 2026-09-03, at
p = 0.6716 on 0 of 12 runs for we1, at p = 0.0075 on 10 of 12 for wg1. That row
alone moved the title: six open conflicts against seven, over the 22 registry
rows an instrument reads.

It does not close the phase itself. Pooled over the three segment pairs the
model holds its opposite legs 0.3085 of a cycle apart against a band of 0.35 to
0.65, and per segment the front pair sits at 134.5 degrees, the middle at 36.5
and the **hind at 20.0**, where the animal holds all three near 179.9. So which
legs avoid swinging together is now right and how far apart they get is still
wrong, and the hind segment is where it is worst.

**That per-segment reading was itself wrong for six hours today.** The middle and
hind rows were swapped in both the battery and the card: the capture is built
over sorted leg keys and `lanes/A/phasemeasures.py` labels the same arrays in a
different order, so the pair everything called middle was the hind one. Lane E
found it at 15:50, an hour after this seat added the rows. No verdict moved,
because both rows read out of band on both arms under either label, but the fork
this seat had given lane B at 15:11 pointed at the middle legs and the defect is
in the hind ones.

## Who did what

Lane C measured the on-path cable fraction (C-N78) without being asked for a
delay correction. Lane B measured the halved cord and reported the trade that
made the gentler version worth trying. Lane E searched the scale, ran the
anchored arm against its own, declined to nominate the anchored one on the row
it opens, named the clip seed as the median rather than the best, and found the
swapped phase rows in an instrument it did not build. The judge wrote which
conflict count decides before the match ran, graded it, and built nothing that
it graded.

Robin's ruling of 11:4x is what made this title possible at all: above a
standing gate, closing a real conflict outranks the qualifying count. wg1 did
not need that ruling in the end, since it leads on both. The previous champion
did.
