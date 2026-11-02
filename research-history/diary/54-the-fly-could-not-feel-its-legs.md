---
id: 54
date: 2026-09-04
title: The fly could not feel its legs
tldr: The champion's leg position sensors delivered exactly zero to its nerve cord on four to five of its six legs, because a high-pass filter tuned above 1 clips a steady signal to nothing. Twelve explanations for a left-right stepping asymmetry were killed first, all of them about the connectome, because the fault was upstream of all of them. Turning the senses back on was expected to slow the fly, since every sensory channel reaches the motor pools through majority inhibition and the literature says feedback masks a leg's own rhythm; it did the opposite, and the judge's written corollary was contradicted seventy-five minutes after it was recorded. The one key that had to be right was reading each claw cell's tuning from BANC's own SNpp50/SNpp51 labels rather than splitting them by row parity: the same restoration flips from 0 of 10 seeds up to 10 of 10 on that key alone, at less drive. The result, wa1, is the first fly to clear the v3.6 bar, on two judge batteries with standing 12 of 12 and 12 of 12 and qualifying 8 of 12 and 6 of 12. Speed is unchanged and the other conflicts are carried over unread.
era: standard
champion: wa1
commit: the coronation commit this entry lands in, tagged wa1
seeds: judge 30171-30194 both batteries; A 202036-202047 and 202048-202059 for the nomination; E 100684-100695 for the hair arm
---

2026-09-04 23:34 EEST. **Champion: wa1**, crowned tonight under v3.6, the first fly to clear that
bar. It replaces qh1, which held the record since 2026-09-03.

The day began with a puzzle about two slow legs and ended with a champion, and
the thing in between was not a discovery about the fly. It was a discovery about
what we had been simulating.

## The hunt

The champion's left front and left hind legs stepped at about half the rate of
their partners. Two lanes measured it on three separate batches of seeds and got
the same numbers to the second decimal, so it was real and it was not noise.

Lane A then killed twelve explanations for it. Which premotor cell types reach
the pool, how much input volume each leg gets, when that input arrives, how hard
those cells fire, whether the cells were muted, whether their encoded position
had saturated, whether the signal they carry was uninformative. All null, all
about the connectome, over about four hours.

The sharpest of them swapped the two sides' inputs so that the right sensors were
reading the left legs' joints. They still did nothing. So it was not what those
cells were saying.

## What it was

Every claw afferent's output passes through a high-pass filter that subtracts the
cell's own running average and re-amplifies what is left. The champion's setting
for that filter was 1.967. Above 1, the subtraction is larger than the signal,
and the result is clipped at zero.

A claw cell is a position sensor. It sits at its own average by construction.

So the output was not small. It was **exactly zero**, on every tick, on four to
five of the six legs. Turning the gain up multiplied zero. Deleting the cells
removed nothing. Halving their connections changed nothing. Twelve explanations
had to fail, because the fault was upstream of everything they examined.

Then lane E asked the question one level wider and found it was not only the
claw: the same filter gates all five sensory channels, and two hair-plate
channels were dead as well.

## The prediction that was wrong, and whose

Turning the senses back on was expected to make the fly slower. Two reasons, both
good ones. Lane E measured that every proprioceptive channel reaches the leg
motor pools through majority inhibition one synapse out. And the literature has
said since Sapkal that proprioceptive feedback masks a leg's own rhythm rather
than generating it, so a fly with its feedback deleted should look faster than
one without.

This diary should record that the judge wrote that reasoning down as a corollary
at 22:52 and it was contradicted seventy-five minutes later by the arm it asked
for. Lane A put the corollary into its registration as the thing to falsify,
which is why the answer arrived so quickly.

A body with all five channels live stands 12 of 12 and steps **faster**: +0.682
Hz paired, up on 11 of 12 seeds, with a quarter more lifts on 12 of 12, and the
left-right asymmetry cut to a third. It replicated on a second fresh block the
same hour.

## The one key it turns on

The champion assigned each claw cell to the flexion or extension population by
whether its row number was even. BANC labels those cells itself, SNpp50 and
SNpp51, and that labelling is mirror-asymmetric.

Restoring the claw under row parity: lifts fall, **0 of 10 seeds up**.
Restoring the claw under BANC's own labels: lifts rise, **10 of 10 seeds up**.

Same channel, same restoration, less drive in the arm that works. The asymmetry
in the annotation is not a defect to be corrected. It is load-bearing, and the
symmetric split was what broke the leg.

## The bar

wa1 clears v3.6 on two independent twelve-seed judge batteries: standing 12 of 12
and 12 of 12, qualifying 8 of 12 and 6 of 12, tip cadence 5.045 and 4.955 Hz on
the ten-millisecond read. The bar asks for 6 of 12 standing and 2 qualifying.

That bar was raised twice this morning, both times against the fly that held the
title: at 00:45 because a quarter of the champion's counted steps turned out to
be one-tick hovers, and at 01:05 because the joint integrator had a numerical
mode in it. The throne sat empty for twenty-three hours.

https://theuploadinglab.ai/fly/walker_wa1_body.mp4

## What it does not fix

Speed is flat. The fly still travels about 1 mm/s where a real one does 2 to 45.
Tilt is slightly worse. Its other open conflicts were carried over from qh1 and
have not been re-read on this body yet.

## Who did what

Robin set the goal: make the scoreboard measure the copy, and fill the vacant
title. He also asked, on seeing the clips, why the legs went through the ground,
which turned out to be a renderer replaying commanded joint angles instead of
achieved ones and was fixed the same hour.

Lane A ran the hunt, killed its own twelve accounts, caught the judge's
survivorship error, and built and confirmed the champion. Lane E found that the
gating was not only the claw and measured the inhibition structure. Lane B traced
the claw-to-motor path and found it present on every leg, including the silent
ones. Mistral caught that the cause had been sitting in the calibration ledger
since 06:37 that morning, framed as a note about the code's shape rather than its
absence.

The judge measured the per-leg zeros, ran the title match, and was wrong twice in
public: once about a speed gain that was survivorship, once about a cadence cost
that was not there.
