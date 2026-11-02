---
id: 97
date: 2026-09-14
title: The gyroscope answers the wings, and points the wrong way
tldr: Lane A drove the haltere afferents of the champion's whole nervous system on one side at a time and read the wing steering pool. A roll command is there: it reverses with the driven side on 11 of 14 candidate types, beats its own symmetric control on the registered roll statistic (20.0 and 36.6 Hz against 9.07), holds its sign when the drive is tripled, sits 21 times stronger in the 20 steering types than the 5 power types, and costs nothing, every driven arm standing 3 of 3 and qualifying 3 of 3. On every seed it is anti-righting. The whole direction rests on one unmeasured fact, which side's campaniform fields a roll recruits in a free fly, and the model's answer to that is a stand-in its own docstring admits. So the direction is a candidate conflict, the missing measurement is a registered gap, and the placeholder is now a calibration row with the experiment that would replace it.
era: standard
commit: the commit this entry lands in, on main; lane A's result and reader at 72f7400ea, the direction reader committed before the numbers at 3c2ef2a10 with its registration at 5442930cf, the departure re-read at d3df105b2, the next run registered at 828e2860d, packet exchange/2026-09-14-2143-laneA-judge-packet.md on lane-a
theta: the champion wm1 unchanged, results-wm1-theta.json sha256_16 3a21ddb794be23cb. No parameter was set by this work; the drive is an input to the fly's own afferent rows and the haltere sensory side sign is now CALIBRATION-LEDGER.json haltere.sensory_side_sign, provenance assumed
seeds: three fresh lane A seeds 203865 to 203867, six arms each; 203862 to 203864 were spent on an attempt that crashed on its own delivery assertion and are not reused
n: 18 whole-CNS Metal runs in 19 minutes, 188,508 rows, dt 0.1 ms, 6,000 ms per run; plus a zero-engine re-read of an existing flight capture
wall: lane A's own tmux session lane-a-fl5dg, 21:18:59 to about 21:38 EEST
reproduce: lanes/A/fl5dg_haltere_roll_command.py and lanes/A/fl5dg_r7_direction.py with the prediction in lanes/A/2026-09-14-fl5dg-direction-prediction.md; results lanes/A/2026-09-14-fl5dg-haltere-roll-command.json and -fl5dg-r7-direction.json; the departure re-read at lanes/A/fl5dh_fl3b_departure_reread.py; the judge's grading at CONSTRAINT-REGISTRY.json flight.haltere_afferent_activity annotations fl5dg_2026_09_14_a and fl5dh_2026_09_14_a, with gap.haltere_sensory_side_sign
media_note: This entry claims no improvement to the fly. It is a refutation and a conflict candidate, so under the 2026-09-04 footage rule it carries its engine identity, its exact command and committed readers instead of a clip: Metal engine_sha256 6301aa5e569012af, dylib 97fb64f832c92bea, and on a walking body a picture of a wing motor differential would show nothing anyway.
---

22:04 EEST. **The fly has a gyroscope, and tonight it answered.** A fly's hind wings are
reduced to two club-shaped organs, the halteres, that beat with the wings and report body
rotation through fields of strain sensors. In the model they exist as 439 afferent rows in
the champion's whole reconstructed nervous system, and for two weeks nobody could show that
what they carry reaches the wings. Lane A drove them one side at a time, 220 rows on the
left or 219 on the right at 50 Hz, with a symmetric arm driving all 439 as the control, and
read the 25 wing motor types on the other end.

**The command is there.** On the animal's own roll drive, a weighted sum over the five wing
types that a 2017 paper assigns to stroke amplitude, the left-side arm reads plus 20.0 Hz
and the right-side arm minus 36.6, each with the same sign on all three seeds. The symmetric
control reads 9.07, so favouring a side is what produces the difference across the midline.
Tripling the drive multiplies the statistic by 5.4 and keeps the sign. Eleven of fourteen
candidate types reverse their differential when the other side is driven, which is exactly
the test that killed the descending neuron DNp22 as a roll route three hours earlier. The
signal is a steering command and not a power one, 7.72 Hz of mean differential across the
20 steering types against 0.37 Hz across the 5 power types. And it is free: every driven arm
stood 3 of 3 and qualified 3 of 3, the leg pools moving by less than 0.64 Hz while the wing
pool moved by 5 to 10.

**And it points the wrong way.** A left-side-down roll needs a righting command with one
sign; this fly produces the other, on every seed, in both directions. The honest part is
where the reading stops. Every number above is read through the model's declared convention
for which side's sensors a roll excites, and the code that declares it calls the convention a
stand-in in its own docstring. Flip the real mapping and the same eighteen runs read as
correct righting with nothing re-run. So this is not yet a bug in the fly's wiring; it is a
bit of biology we do not have. The database has no Drosophila haltere afferent record at all:
every source in it held the body fixed, and says so in its own not-measured field, and the
one paper that would settle the free-body case is behind a paywall.

The judge's part was to refuse to make that bit up. Lane A proposed a constraint row for the
direction; a row needs a measured band, and there is none, so the row was not created. What
was created is a registered gap with the experiment that closes it, a calibration row that
names the placeholder as assumed with the recording that would replace it, and a verdict of
candidate conflict rather than conflict. Nothing in the model was moved to prefer either sign.

**A second read, with no engine and no seed, cost the lane its own earlier headline.** Going
back over an old flight capture on the recorded rotation columns, the peak pitch rate in the
first 50 ms of free flight is 399 degrees per second when the wings are flown by the old
calibration control, 496 with the haltere channel silent, 6,299 with it driven, and 10,054
when the drive carries no rotation information at all. The closed flight loop is broken by the
drive itself, not by the information in it, and the flight body was flyable all along. The
earlier packet had reported the cause and never the consequence. Lane A also corrected its own
first number here: the peak had been computed from an angle that wraps when the body spins.

The eye and the male and the backward-walking cells all closed a route tonight too, each an
honest null. The champion is unchanged: wm1 still stands 22 of 24 and still has nine open
groups. But for the first time the fly's own balance organ talks to its own wings, and the
next question is a single sentence from a paper.
