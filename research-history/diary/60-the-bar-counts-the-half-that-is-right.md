---
id: 60
date: 2026-09-06
title: The bar counts the half that is right
tldr: Three separate changes made the fly cover more ground overnight and the title beat all three, because walking speed is stride times cadence and the bar counts only cadence, which is the half already inside the animal's band. The champion's step is about a fifth of a millimetre where its step rate is fine, so the deficit is arithmetically in the stride, and every challenger got faster the same way: it lengthened the step. Widening a joint's range does that and raises the body's vertical bounce monotonically, so it swaps one measured conflict for another; lowering the adaptation does it and costs nothing but a little cadence. The judge published three wrong readings in four hours and lanes refuted two of them within five minutes each, including one built on a denominator the judge had checked in only one direction. The lever nobody has pulled is the drive scale, which uses the range the joints already have.
era: standard
champion: wc6
commit: the commit this entry lands in, on main after the fifth defence
seeds: judge 30280-30303 (wc4, both arms), 30316-30339 (wc5, both arms), 30304-30315 and 30268-30279 reserved and unspent; lane D's dose blocks, lane A 202810-202821, lane E 101438-101449
---

02:25 EEST. **Champion: wb4**, unchanged since 2026-09-05 17:05, five defences.

## The night's arithmetic, in one line

Walking speed is how far each step carries the fly times how often it steps.
The champion's step rate is **5.7 per second against a real band of 5 to 16**,
so it is fine. Its speed is **1.2 mm/s against a band that starts at 2.0**, so
it is not. Therefore the deficit is in the step LENGTH, and it is about
**0.21 mm**, and nothing in the bar reads it.

That was computed from the receipts of five title matches with no new run, and
it reframes everything that happened before it. Three bodies were beaten on the
count overnight and all three had got faster the same way:

| body | speed | cadence | stride |
|---|---|---|---|
| wb4, champion | 1.216 | 5.727 | 0.208 mm |
| the middle coxa-trochanter cap | 1.637 | 5.909 | 0.269, +27% |
| adaptation at half | 1.446 | 5.364 | 0.259, +35% |
| adaptation at a quarter | 1.777 | 5.273 | 0.336, +61% |

Two of the three paid a little cadence for the longer step and lost the match on
that payment. The bar counts the quantity that is already inside the animal's
band and does not count the one outside it.

## Two ways to lengthen a step, and one of them is a swap

**Widening the joint's range works and is not free.** Lane A read every range
cap it had run, from receipts already committed and with no seed: the vertical
bounce rises monotonically with the cap, +0.053 of body height at 45 degrees,
+0.093 at 91.2, +0.209 at 180, unanimous or near-unanimous on every block, while
the adaptation dose does not move it at all. The body's height oscillation is
itself an open conflict with the animal, about 70 percent of body height against
roughly 10. So the correction that closes the middle-leg range conflict widens
the body-height one while it does it. Lane A withdrew its own nomination on
finding this, before any judge seed had run on it.

Then it asked whether the bounce was buying the speed, registered the thresholds
first, and found it was not: within a fixed body across its own seeds the median
correlation between speed and bounce is **-0.109** over 33 arms. So the cap does
two independent things, and removing the bounce while keeping the speed is a
real target rather than a contradiction. It also tried to break the bounce
conflict itself, suspecting the five-second window was collecting drift, and
refuted its own suspicion: 96 percent of the statistic survives a 2 Hz cut and
the fly bounces 0.763 mm within the stride.

**Lowering the adaptation works and costs almost nothing.** The dose curve had
been sitting on disk since the afternoon, unread by anyone including the judge
who had spent the evening routing experiments around it: forward speed 1.965,
1.358, 1.059 at adaptation 0, 2 and 4, monotone, with cadence rising once and
then flattening. At zero the body is within two percent of closing the speed
conflict and misses the cadence floor by 0.09 per second. **The coronation that
made wb4 champion traded 0.09 of cadence for 0.6 mm/s of ground**, on the row the
fly is furthest from the animal on, and won 24 to 17 on a count that reads
neither.

## What the judge got wrong, three times

**"This fly buys ground with posture."** Written from three lanes' packets
without checking whether the arms shared a base body. They did not.

**The correction to that.** A pooled correlation over 136 arms mixed bodies and
reversed sign under stratification, which is Simpson's paradox, published twenty
minutes after correcting a lane for less. Within the current champion speed,
bounce and tilt do rise together; two champions back none of the three is
coupled to any other.

**"The model's legs sweep 54 percent of the animal's."** The most useful mistake
of the night. The project's own frozen reference holds leg excursion and body
length for 3,000 real flies, both written off in `AGENTS.md` as uncomparable
because one side is camera pixels and the other millimetres. That is true of
each alone and false of their ratio, so the judge took the ratio and found a
two-fold conflict. Lane A refuted it in five minutes: the two body-length
landmark pairs do not span the same share of their animals, 0.360 of the
tracked fly against 0.551 of the simulated one, a factor of 1.531 and all of it
in the denominator. Corrected, the fly sweeps about 83 percent and the row is a
near miss.

The instructive part is not the error, it is how it was missed. The judge had
explicitly asked at 02:11 whether the denominator might be too SHORT, worked out
that this would inflate the ratio and so could not manufacture a deficit, and
stopped. The real error was a denominator too LONG, the one direction that does
manufacture one. Checking one sign of one artefact was treated as having checked
the artefact.

None of it touches the stride finding, which uses no landmarks at all.

## The falls that decide the matches still have no cause

The 171-degree inversion has now survived four attempts to explain it. It is not
a loss of support: the fly is never airborne for a single tick before it goes
and still has two to five feet on the ground. It is not a slow topple: 30 to 150
degrees in 27 to 75 milliseconds, over a diagonal. It has no precursor in tilt,
feet-down count, unsupported interval or tilt slope across 146 runs. A load
precursor found on two falls failed its own registered bar on eight. And the
legs are not pushing it over: contact pitch torque before departure is 35.7
against the standing champion's 39.9 on the same seeds.

One number came out of the last of those by accident and is the only open thread
on it: both the falling and the standing body carry a persistent nose-up pitch
torque of about 40 units from the hind legs. An inversion may be a failure to
balance a standing torque rather than an impulse that arrives, and nothing
measured so far would tell those apart.

## One cell, correctly identified, that the model cannot represent

BANC types one CT1 and leaves the other untyped and transmitter-blank, so the
build signed it excitatory and the left optic lobe ran with an excitatory giant
where the right had the real inhibitory one. The judge proposed it as the single
cause of the day's left-right vision chain. Lane C patched it to CT1 and GABA
and read the chain: left tangential-cell separation rose nearly fivefold and it
did so **by removing the detector**, left T4 rate to 17 percent of baseline,
left T5 to zero on all three seeds, the horizontal-system asymmetry 3.2 times
worse and its direction flip reversed.

The identity is not in doubt. What the model cannot carry is the identity. The
point-neuron graph is one node with 6,980 sources and 7,221 targets where the
animal has one cell whose hundreds of terminals act locally, so the correct
transmitter turns that node into a correct global inhibitor of an entire lobe.
**The right type in the wrong shape is worse than the wrong type in the wrong
shape**, because the wrong type was excitatory and therefore weak in a lobe that
is mostly excitatory anyway. Compartments are now the blocker, and this is the
first time the representation defect this project has carried as a caveat since
early September has had a measured behavioural cost.

## The lever nobody has pulled

The joint's equilibrium in `body.py` is the mid-point plus the half-range times
the hyperbolic tangent of the drive over a drive scale. The half-range is a
**gain**, not a clip, which the judge got wrong on 2026-09-05 and was corrected
on. So the step's amplitude can be raised two ways: widen the half-range, which
is the cap and costs bounce, or push the tangent closer to one by lowering the
drive scale, which uses the range the joint already has and never touches the
limit. Lane A measured the middle knee commanding **0.559 of the half range it
has**. If that is typical the model is using a little over half the sweep its own
joints allow, and the correction argued about all night widens a limit that is
not binding.

## Attribution

Robin: away; the decision on whether covering more ground should win a close
match is waiting on him with its default, and two clause changes are his to
veto. Lane A: the cap-bounce law, the mediation test that separated bounce from
speed, the landmark audit that killed a judge's row in five minutes, the
window audit that confirmed a conflict it had tried to break, the seed read on
the deciding fall, and two nominations withdrawn on its own evidence before a
judge seed ran. Lane D: the dose curve that had the answer nine hours before
anyone read it, two nominations filed on request, and the heave measurement that
changed the bar within half an hour of the row existing. Lane E: the inversion
read tick by tick, and two registered self-refutations, the seeds-cluster
reading and the load precursor. Lane C: stopping on an unstable gate rather than
proceeding, the NumPy evidence that overturned the judge's engine diagnosis, and
the CT1 chain read. Lane B: a self-correction of its own prose against its own
table, the descending census, the second hop on hemilineage 19A, and a Metal
refusal filed rather than worked around. The judge: five defences held, two
clauses written and three declined for favouring bodies it had routed, three
wrong readings published and corrected, and the two findings that mattered most
taken from files that had been sitting unread on disk.

## Addendum, 2026-09-06 03:45 EEST: the first title decided on a conflict rather than a count

**Champion: wc6**, crowned 03:45 under bar v3.8, ordering items 5 and 6. Tag
`wc6`. Nominated by lane E at 03:17, hash claimed and verified. Seeds
30340-30387 for the match and its extension, 30388-30399 for the card. Receipt
`lanes/judge/2026-09-06-title-v38-wc6-coronation.json`.

### What happened

wc6 is the previous champion plus three labelled biology corrections and **not
one searched dial**: the campaniform channel fed the leg's matched ground load,
a six-cell remotor transfer in the second thoracic segment, and one transmitter
map. Four keys, all additions.

It tied wb4 **23-23** on twenty-four fresh seeds. The tie-break drew twenty-four
more and both arms ran again: **23-23**. Over all forty-eight paired seeds,
**46 each**, with one discordant seed either way in each half.

Dead level. So the bar asks its next question, which it has never had to ask
before: **which of the two carries fewer unresolved conflicts with measured
data?** wc6 carries one fewer. Its campaniform sensors are fed the leg's actual
load. Every champion before it fed them **muscle activation**, a stand-in
adopted in August when the model had no ground, and kept for a month after the
body closed and gave it ground. A campaniform sensillum reports cuticular strain
under load; it does not report what a muscle is doing.

The card veto was then read, as the clause requires on every route to a
coronation, and returned **nothing either way**: no row moved three of twelve in
either direction, and the floor rule found no change on either row that is out of
band for both flies.

### Two things decided before the numbers, which is the only reason this counts

**At 03:20 the block rule changed.** Three holder records existed by then, at 24,
23 and 22 of 24 on three different blocks, and reusing one was the judge's
choice. In a match decided by one or two seeds, and three of the night's five
were, that choice is the margin. So each nomination now runs **both arms on a new
block**, at a cost of twenty-four extra runs, and the discretion is gone. It was
written before this match's first seed.

**At 03:21 the tie-break standing was fixed.** The judge determined that wc6
closes the campaniform gap and wb4 does not, read from the theta with no run,
against a gap registered sixteen hours earlier. That determination was made
**before any count from this match existed**. Had it come afterwards, with the
numbers visible, this coronation would be worth nothing, and the judge had
already declined five clause changes tonight for exactly that reason.

### What it is not, said as plainly as what it is

**It does not walk better.** Forward speed 1.50 mm/s against a real band that
starts at 2.0. Vertical bounce 0.71 of its own body height against about 0.10.
Contralateral phase 0.266 against a band from 0.35. Every one of those is
inherited from wb4 at the same magnitude. On the bar's own counting quantity it
is exactly as good as the fly it replaced, and not one step better.

It won because it stopped pretending one of its senses was something else.

### The night around it

Five defences held before this one fell, three of them against flies that walked
measurably faster and lost on the step count. The judge published five
corrections to its own claims and had three hypotheses refuted by lanes, twice
within five minutes. Lane E's diagnostic under this nomination found that
reverting **any** of six dials rescues all four of a body's falls, so no dial
causes them; the fly sits at a point in dial space that any axis leaves. Lane B
drove a premotor hemilineage and turned the body for the first time, then moved
the contralateral phase toward the animal for the first time, with a confound in
the phase measure the judge raised and neither has yet closed.

## Attribution, this addendum

Lane E: the nomination, the corrections-only body its goal asked for, and the
rescue diagnostic under it. Lane A: four registered self-audits in one hour,
including one that verified a judge note it had been repeating on the judge's
authority, and one that refuted its own post-hoc. Lane B: the turn, the phase
movement, and a steering test reported as failed when a looser reading was
available. Lane D: the withdrawal of its own lead on its own rule with the
medians in its favour. The judge: two rule decisions taken before the numbers
that made the result mean something, five corrections to its own claims, and a
coronation it did not expect when the match began.

### Correction, 2026-09-06 04:30: "it does not walk better" was wrong

The addendum above says, twice and at length, that wc6 does not walk better than
the fly it replaced. **That is withdrawn.** It was written from the qualifying
counts, which tied at 46 of 48, and a qualifying count is the bar's counting
quantity rather than a measure of walking.

Lane A re-based its whole bounce line onto the new champion within forty minutes
of the coronation, raising the disagreement rule in its own first line because
its goal file still named the old one. Paired over twelve reserved seeds with
both arms standing and qualifying 12 of 12, it found **forward speed +0.545 mm/s,
up on ten of twelve, sign test p 0.039**. Lane E's nomination evidence had given
+0.51 on nineteen of twenty-three pairs, and the two cards read 1.5031 against
1.2523. Three independent reads, one of them paired and pre-registered.

So the correct sentence is: **wc6 qualifies on exactly as many runs as the fly it
replaced, and it walks measurably faster.** On the largest conflict this project
carries, the coronation moved the fly toward the animal by half a millimetre per
second.

What genuinely did not move, and this half of the original claim stands: the
vertical bounce is unchanged, paired at -0.0113 with its 5.841 Hz peak identical
to the digit, and the contralateral phase is unchanged. Lane A's entire bounce
line therefore transfers to the new champion without re-measurement, which is the
result it set out to get.

The entry above is left as written, per the no-retroactive-polish rule. The judge
wrote the wrong sentence four times in four places within ten minutes of the
coronation, and it took a lane forty minutes and twelve seeds to find it.
