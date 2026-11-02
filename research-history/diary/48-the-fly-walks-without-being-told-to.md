---
id: 48
date: 2026-09-03
title: The fly walks without being told to
tldr: Three results on the crowned fly in one night, all replayed exact. Lane A set the descending command to zero and ng1 still stood and travelled on every seed, at about 3 Hz instead of the 5 Hz the command gives it: the command is not what makes it walk, it is what sets the pace. Lane D re-ran the one manipulation that ever stood the fly, DNg93 driven at full rate, against a command the connectome can now inhibit, and it still did not stop a driven walk; DNg93 gates self-starting and nothing else. Lane E drove the two premotor 13B cells that stand in for a published optogenetic driver and the fly sped up 11 percent where the real fly slows by 40 to 80: a measured sign the model gets wrong, registered as a conflict candidate. Lane B's 30-cell walking map was title-matched on ng1's 24 paired seeds and lost 14 to 19; ng1 keeps the title for the second time in three hours.
era: standard
commit: d57139de7 (folds, engine, rows), match record in the following judge commit
seeds: judge 30001-30024 paired with ng1's; A 200867-200869; D 60472-60474; E 100125 (reused)
n: two title batteries, one paired comparison; A four arms by three seeds; D two arms by three seeds; E two arms by one seed
wall: 1,018 s + 996 s battery wall (NumPy, batteries in parallel, about 85 s per seed); A 302 s; D about 7 min compute; 00:46-01:21
reproduce: PYTHONPATH=. .venv/bin/python lanes/judge/title_match_v33.py --theta lanes/judge/theta-ng1-walkingmap-challenger.json --seeds 30001-30012 --battery 1 --out receipt.json
video: walker_ng1_body.mp4
video_url: https://media.theuploadinglab.ai/walker_ng1_body.mp4
---

**Question.** Four questions from four lanes. Lane A: does the crowned
fly walk at all with no descending command, now that lane D has shown the
command is delivered as a forced spike? Lane D: does DNg93, the one cell
pair whose activation stands the fly, stop a walk once the command is
delivered through the membrane and can be inhibited like any other cell?
Lane E: does driving the two BANC cells homologous to a published 13B
optogenetic driver slow the fly the way the driver slows a real one?
Lane B: is the 30-cell walking map, which took 10 of 12 of its own seeds
against ng1's 5, a better fly on the title instrument?

**Method.** Every arm ran the whole female CNS (188,508 neurons) on
NumPy, ng1's own engine and parameters, and every result below was
replayed by the judge on main before it went on a row: one seed of each
lane's result, every field exact. Lane A ran four arms on three fresh
seeds, 6,000 ms: the crowned spike command, the command at 0 Hz, the
same under membrane delivery, and a 14 mV membrane command. Lane D ran
a 14 mV membrane command with and without DNg93 forced at 94.26 Hz on
three fresh seeds. Lane E paired a sham with two left-T1 13B rows
(IN13B010, IN13B013) released continuously from rest toward threshold
for 720 ms, the stimulus length of Agrawal et al. 2020's Figure 8, on
one reused seed, stopping at the first gain under its registered rule.
The title match followed the ordering clause: a frozen theta hash
written at 00:46 before any judge seed ran, two twelve-seed batteries on
ng1's 24 paired seeds, spike regularity k = 44, higher count takes the
title and a tie keeps the holder.

**Result.** With the command at 0 Hz ng1 stood on 3 of 3 seeds and
travelled 1.84, 2.76 and 3.07 mm, tip cadence 3.27, 3.55 and 2.82 Hz;
under the command the same seeds gave 5.55 Hz, a fall, and 5.00 Hz. The
command cells still fired 546 to 651 times in six seconds from network
input alone, a third of their commanded 1,535 to 1,661. Spike and
membrane delivery with nothing to deliver were bit-identical. Lane D:
DNg93 at full rate against the membrane command left 7.59, 7.69 and
7.19 mm/s of an 8.16, 9.77 and 9.30 mm/s walk, 0 of 3 standing. Lane E:
sham 8.819 mm/s over the stimulus window, structured 9.810, plus 11.2
percent, standing kept at 9.96 degrees of tilt; the measurement is a
decrease of about 0.65 cm/s. The title match: the 30-cell map qualified
14 of 24 against ng1's 19. Both its batteries passed on their own (7
and 7 qualifying, standing 11 and 10 of 12). It took exactly the five
seeds the 3,376-row union took at 23:25 and lost ten: six standing
runs at 4.45 to 4.82 Hz tip cadence, three falls, one foot lifted four
times.

**Interpretation.** The walk is the network's; the command is a pace
setter. That reorders the v3.3 question. The bar asks for cadence the
model produces itself inside 5 to 16 Hz, and the model already produces
about 3 Hz with no generator anywhere in it; the search is for what
raises that rhythm, not for whether one exists. DNg93 fits the same
picture from the other side: it prevents ignition of the uncommanded
walk on every seed tried and cannot stop a driven one under either
delivery, so in this model it is a gate on starting. Command-cell
inhibition, which lane A showed works under membrane delivery, is the
only stop route on record, and it works by removing the pace setter,
not by braking the legs. Lane E's sign error is the sharpest of the
three because a sign has no tolerance to hide in: everything that
raises 13B output on this body (the resting-potential dial of R149 and
R150, the direct activation of R153) raises speed or cadence, and the
real fly slows. Either the two rows are not the driver's cells, or the
13B rows in the graded map are wired to the wrong sign downstream. The
second is checkable from the connectome without a run. The title match
says the membership dial is exhausted at this parameter set: two
challengers built from the evidence map have now lost the same way, on
cadence a fraction of a hertz under the floor, with membership changing
which seeds those are and not how many.

**Problems.** Lane E's conflict rests on one reused seed, one gain with
no measured conductance behind it, and a homology hypothesis for which
BANC cells the driver expresses in; it is a candidate until the sign
holds on three fresh seeds across a gain bracket with the permuted twin
beside it. Lane A's zero-command cadence of 2.8 to 3.6 Hz is measured by
the same tip estimator whose resolution near the 5 Hz floor is an open
instrument row. Lane D's DNg93 still takes 0.6 to 2.1 mm/s off the
driven walk on every seed, so "no brake" is a statement about standing,
not speed. The challenger's three falls against the wide union's one
are on the judge block only; its own twelve Metal seeds fell less. The
battery receipts name the merged engine in their provenance because they
hash the file on disk at completion, and the two engine changes landed
tonight were applied to the tree at 01:06 while the batteries, launched
at 01:02, ran on the previous engine; the match record says which
engine actually loaded. A first launch of the same batteries at 00:47 on
that engine was killed from outside at 01:00 with ten seeds each done,
and the relaunch reproduced all twenty seed for seed.

**What changed in the model.** Nothing in the champion. Registry:
`brain.spontaneous_walking` now holds ng1 beside nl1 with the command
reading; `cmd.DNg93_stop` gained the membrane column and the gate
reading; new `premotor.13b_activation_slows_walking`,
CONFLICT-CANDIDATE, FEMALE-ONLY; the second defence under the BANC live
title. Ledger: `cns.command_delivery` carries both results; new
`cns.graded_13b_command_gain`, searched, with the driver's conductance
curve and expression map named as the measurement that would replace
it. Engine: lane E's 13B command mechanism and lane C's actual T4/T5
tape into the graded HS body, both default-off, probe bit-identical,
one merge conflict resolved by hand. Video: none new; ng1's clip
stands, https://media.theuploadinglab.ai/walker_ng1_body.mp4

**Attribution.** Robin's standing instruction from 21:45 (enable the
lanes, do not direct them) shaped every feedback file tonight. Lane A
designed and ran the zero-command arms and named the next probe
(`mn_tonic_slow_hz` on the uncommanded body). Lane D built the
membrane-command DNg93 arm on lane A's seam and drew the separation of
the two stop routes. Lane E built the 13B command mechanism, chose the
homologous rows, registered the stopping rule and reported the wrong
sign as the result. Lane B built the 30-cell map and declined to crown
from its own Metal card. Lane C landed the T4/T5 bridge. The judge
nominated the challenger, ran the batteries, replayed one seed of each
lane's result, resolved the engine conflict, and recorded the defence
and the rows.

cites:: lanes/judge/2026-09-03-title-v33-walkingmap-battery1.json, lanes/judge/2026-09-03-title-v33-walkingmap-battery2.json, lanes/judge/theta-ng1-walkingmap-challenger.json, lanes/A/2026-09-03-cd1oh-ng1-whole.json, lanes/A/2026-09-03-0046-cd1oh-result.md, lanes/D/2026-09-03-dng93-stop-membrane-result.json, exchange/2026-09-03-0102-laneD-judge-packet.md, lanes/E/2026-09-03-ng1-13b-slowdown-r153-result.json, exchange/2026-09-03-0104-laneE-r153-judge-packet.md, exchange/2026-09-02-2340-laneB-judge-packet.md
uses-data:: CHAMPION.json, CONSTRAINT-REGISTRY.json, CALIBRATION-LEDGER.json
