---
id: 93
date: 2026-09-13
title: The model read against the papers, with nothing running
tldr: Robin ruled that every published measurement the model contradicts becomes a tracked conflict row and that every conflict row is read by the battery or marked unreadable with its reason and an owner. The first action under it spends no simulation at all. The registry goes from 79 rows to 110, 53 of them conflict rows that no instrument reads, and twenty rows read straight off the champion's parameters open conflicts nobody had looked for: the balance rule raises inhibition on 133,867 of 188,508 neurons by a median factor of 2.5, an olfactory receptor neuron drives its partner 1.01 mV where the animal's drives about 5, and the two named front leg motor neurons sit at -63 and -64 mV where they were measured at -68 and -48.
era: standard
commit: the commit this entry lands in, on main; the ruling at db16d2e0c, the consultant's sweep at 9b26640a3, the inventory and shift record CONSTRAINTS.md 16:09 at df81b8fa0, the registration 16:33 at 2ed79baba, the lanes' first reads 16:39 at da6653bfb, the static receipts 17:46 at 784e0b806
theta: results-wm1-theta.json, sha256_16 3a21ddb794be23cb, unchanged. Nothing in this entry sets a parameter; the static receipts read the champion's theta and the engine's own defaults
seeds: the inventory, the registration and the twenty static receipts spend none. Lane A's flight read used three fresh lane A seeds 203803-203805 plus smoke seed 203800; lane E's two reads and lane C's three archive reads spend none
n: 79 registry rows inventoried, 110 after registration, 20 rows given static receipts, 9 runs behind lane A's wing numbers, 211,577 male rows reproduced by lane E's reader
wall: 3.5 s for the static receipts, 328 s for lane A's nine runs, zero machine time for everything else
reproduce: lanes/judge/static_receipts.py writing lanes/judge/2026-09-13-static-receipts-wm1.json and .md; the inventory at lanes/judge/2026-09-13-registry-readability-inventory.md and .json; the row writes at lanes/judge/2026-09-13-register-sweep-rows.py, -register-lane-rows.py and -register-static-receipts.py; lane E's reader at lanes/E/read_named_motor_intrinsics_r827.py
media_note: Twenty of these rows were read without starting the engine, and the rest are conflicts opened rather than a fly that got better. Under the 2026-09-04 footage rule a result that claims no improvement carries its engine identity, its command and a committed reader instead of a clip; all three are above, and the champion the parameters belong to has its own published clip.
---

16:09 EEST. **A published measurement the model contradicts is now a tracked row, and a
row nothing reads is a debt with a name on it.** Robin, after the hackathon, recorded as
consultant ruling R15 at db16d2e0c:

> "we found many conflicts with reality and I realized we need to have specialized cell
> models for many special cells that have real measurement data that contradicts our
> model. Make sure you or one of the researchers finds as many of those special cells or
> populations or just special functions and write them down to our conflicts, and that
> all conflicts are being tracked in the battery tests."

The state it was ruled on was already on the record and already ignored. The judge's own
audit of 2026-09-06 found **55 of 74 registry rows read by no instrument, 24 of them
graded CONFLICT**, and opened
`gap.the_conflict_count_reads_only_what_an_instrument_computes` for it. That gap was
never closed. Since v3.9 the title is ordered on the count of open conflicts, and since
v3.9 that count has read only what the instruments happen to compute, which is why the
headline figure in every report for two days has been seven.

## The inventory, which is the part that costs nothing and hurts most

The first action under the ruling is a census, written from the instruments' own row
lists and the 129 per-run fields the judge's card records. Of **79 rows, 33 are read
today**: 27 by the title conflict count, one by the qualifying clause, five by the
successor gate as held-out rows. **46 are unread, and 25 of those are CONFLICT or
CONFLICT-CANDIDATE.**

The 46 split by what it would take to read them: 9 this shift from fields the card
already records, 5 once the card records any neural channel at all, 6 as static receipts
with no run, **22 UNREADABLE because their protocol is not walking** (owners lane B 7,
lane C 11, lane D 3, lane A 1), and 4 not applicable to this body. The 22 are the honest
floor. A row measured on a fly responding to an odour puff cannot be read off a walking
capture whatever instrument is built, and marking it so is the difference between a
tracked debt and a document nobody consults.

## Twenty-six rows arrive from the literature, and five citations were wrong

The consultant's sweep (9b26640a3) covers 89 measured quantities across intrinsic
properties, population activity, behaviour timing and synaptic physiology, each with its
PMID. Twenty-six became registry rows at 16:33, each carrying the model's value today and
a status of CONFLICT, or CONFLICT-CANDIDATE where the paper itself had not been read.
The registry went 79 to 105 rows, then to 110 once the lanes' own reads landed:
**53 conflict rows with no reader.**

The sweep also corrected this project's own citations on five rows. The monosynaptic
latency row is Augustin 2017 and Kadas 2019, the cord-period conduction row is Kadas,
Duch and Consoulas 2019, and the 16 Hz top of the cadence band is Mendes 2013. Two
numbers in the source were flagged as having no source at all: the 598 MOhm at
`cns.py:505` is not in Gouwens and Wilson 2009, and the -48 and -74 mV at
`walk_search.py:5088` are unsourced.

## What the champion's own parameters say, read in three and a half seconds

The static receipts (784e0b806) execute the engine's defaults, the champion's theta and
the size rule straight from source, with no simulation. Twenty rows, 3.5 s. Ten findings
worth stating, all on the reigning champion wm1:

1. **The balance rule rescales inhibition on 157,219 of 188,508 neurons.** Each neuron's
   inhibitory input is multiplied by s = E_in / (I_in x 0.435): median 2.5, 5th to 95th
   percentile 0.33 to 10.0, **raised on 133,867 neurons and cut on 23,352**. The measured
   quantity it was written against is a unitary current matched to the postsynaptic
   cell's arbor size, not a fixed ratio, so the row is a conflict in kind.
2. **The giant fibre's excitatory drive onto its wing motor pool is far too weak.** Ten
   PSI to DLMn connections of 15 to 64 synapses give linear unitary potentials of **0.66
   to 2.80 mV against 4.71 measured**, and none of the ten crosses the 7 mV threshold gap
   alone, where the measured synapse is spike-triggering.
3. **Olfaction is weaker still.** Twenty-three receptor neuron synapses onto a projection
   neuron give **1.01 mV against about 5 mV measured**. The median such connection in
   BANC is three synapses, which is 0.13 mV.
4. **The reset sits above rest on 267 of the 391 leg motor neurons** (median +6 mV, max
   +16), equal on 19 and below on 105, because the reset stays at the global -52 mV while
   the size rule sets rest per pool.
5. **The two named front leg motor neurons are both at the wrong end.** The fast tibia
   flexor gets -63 mV against about -68 measured and the accessory slow flexor -64 mV
   against about -48. The rule that assigns them ranks by incoming synapse count, and that
   rank **places the slow cell next to the fast one**, which is the error behind both.
6. **Photoreceptors R1 to R6 have no BANC rows at all.** The 1,846 R7 and R8 rows that do
   exist are spiking units with a 20 ms membrane time constant against about 2 ms
   measured, resting at -52 mV against a dark rest of -60 to -77.
7. **The giant fibre's visual input is short by an order of magnitude**: 171 and 183 LC4
   synapses and 199 and 377 LPLC2 synapses per DNp01 against 2,442 and 1,366 per giant
   fibre in FAFB.
8. **The 16 insulin-producing cells run on engine defaults nothing overrides**: threshold
   -45 mV against -36 +/- 1.4, rest -52 against -62 +/- 4, refractory 2.2 ms against a
   9.3 +/- 1.8 ms spike.
9. **There is no electrical coupling anywhere in the champion.** The measured coupling
   coefficients are 0.023 +/- 0.003 for the MN1-2 and MN3-4 pairs and 0.010 +/- 0.003 for
   others; the model's is zero, since the gap keys adopted this afternoon are default-off.
10. The two rows that cannot be read at all are the input resistances of those same two
    named cells, 150 and 700 MOhm measured. **The model has no megohm quantity**, so they
    are marked UNREADABLE with the reader owed rather than graded.

Two rows pass and are worth recording as passes: the body's joint frequency at 20.55 Hz
inside 15 to 40, and its damping at 3.0 against a floor of 1.0.

## The lanes' first reads under the rule

Lane A flew the champion under its flight command and read the wing power pool
(nine runs, three fresh seeds, 328 s). **The power motor neurons fire 50.50, 54.67 and
32.58 Hz per cell in flight against 2 to 10 measured, and 11.13, 12.92 and 12.03 Hz at
rest where the animal's are silent**: a conflict on 3 of 3 seeds in both windows. The
first basalar steering neuron passes, at 226 to 229 Hz against a band of 150 to 250. Lane
A's own qualifier on that pass is the more useful half: the animal's basalar fires once
per wingbeat with the haltere's input carried in the spike's phase, and a rate stand-in
cannot carry a phase, so a passing rate is not evidence the code is right.

Lane E read Azevedo 2020 with its preparation attached (young females, restrained in
vivo whole-cell, right front tibia, immobile intervals) and produced the numbers in
finding 5 above. It then resolved the slow cell's male counterpart by anatomy, from BANC
720575941526437988 through MANC 20605 to MaleCNS 1050268065, where the native male
control assigns -58 mV and the cell fired zero spikes in the commanded window.

## What bit us, including a conflict row written against the wrong criterion

`mn.reset_below_rest` was a CONFLICT on the grounds that the model's reset sits above
rest. Lane E's counterexample is accepted: **a reset above rest and below threshold is a
legal state for this kind of neuron model**, so the row as written graded an arrangement
rather than a disagreement with the animal. It is downgraded to CONFLICT-CANDIDATE and
the physiological reset target is owed. A conflict row is only worth the count it enters
if its criterion is the measurement.

Three limits on the static receipts, stated because the numbers above read stronger
without them. The balance figures use synapse counts and signs and fold in neither the
per-type output gains nor the size factor. Every unitary potential is a linear
current-based estimate of one spike with no shunting, which is the model's own arithmetic
and not a simulation of it. And `net.edge_density` has no whole-graph counterpart at all:
the band was defined on a core cut of the graph, and the champion keeps every edge, 72.3
per neuron, so the row is a note rather than a grade.

## What changed in the model

Nothing. No parameter moved, no theta changed, no verdict was re-graded, and the
conflict count that orders the title still reads what its instruments read. What changed
is the ledger: 31 new rows, a readability mark and an owner on all 110, twenty static
receipts, and five corrected citations. One engine change landed in the same shift and is
default-off and bit-identical with the keys absent: the hackathon's gap junction keys are
on main, with the Metal engine still refusing them, so a body with electrical coupling
runs on NumPy and cannot be matched for a title today.

Whether any of these rows enters the count is a bar change and belongs to Robin, not to
the judge. He answered the first instalment of that question at 22:04 on the same day,
for the four gait rows; entry 94 carries it.

## Attribution

**Robin** ruled R15 in the words quoted above and set its scope: find the special cells
whose measured data contradicts the model, write them down as conflicts, and track every
conflict in the battery. He also declined, by leaving it to the judge, to say what
happens to the count, which is what forced the proposal rather than a quiet enactment.
**The consultant** ran the 89-row sweep with a PMID on every number and caught the five
wrong citations. **The judge** wrote the inventory, the registration scripts and the
static receipts, and graded the day's packets. **Lane A** measured the wing pool and
supplied the qualifier that a rate cannot carry a phase code. **Lane E** read Azevedo
with its preparation, resolved the male counterpart, and produced the counterexample that
downgraded one of the judge's own conflict rows. **Lane C** and **lane D** closed three
routes by their own results in the same shift, none of which moved a number.
