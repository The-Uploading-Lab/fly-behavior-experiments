---
id: 67
date: 2026-09-07
title: Twelve seeds cannot see it
tldr: The route this seat named as the fleet's aim at 11:13, halving the nerve cord's presynaptic delay to make the left and right legs alternate, was refuted by two lanes within the hour. Neither refutation measured anything either: the hind left-right phase median moves 27 degrees between twelve-seed blocks of one unchanged champion, and both readings sit inside that range. Resolving a ten-degree difference on that row needs of order a hundred seeds per arm. A second challenger was held the same hour, and it was held on standing while leading on the quantity Robin made decisive.
era: standard
video_receipt: lanes/A/2026-09-07-cd2gi-clip-receipt.json
commit: the commit this entry lands in, on main
seeds: lane E 108472-108483 both arms; lane B its own twelve; wi1 batteries 30448-30471 paired against wg1's remade record on the same block, wi1 card 30520-30531 against wg1's card 30496-30507; the block spread costs no seed, being read from six committed twelve-seed blocks of one theta
---

11:46 EEST. **The champion is still wg1.** Two challengers matched today, both
held, both on standing.

## The aim published at 11:13 was dead by 11:19

The 11:13 report to Robin named one route: halve the nerve cord's presynaptic
delay so the left and right legs of a segment alternate, which the animal does
at 179.9 degrees and this model does at 20 to 36. Two lanes answered it inside
the hour, from different sides.

    body                                       hind pair   standing   cadence Hz
    wg1 (champion)                                 20-21      12/12       5.6818
    wg1 + cord x 0.5, no crossed inhibition          35.5      12/12       4.86
    wg1 + cord x 0.5 + commissural_gain 2.0 inh      32.0      11/12       5.36

Lane A had registered a third arm of the same family and had not spent its
seeds, so that arm was withdrawn rather than run.

## Why neither of those numbers is a measurement

The champion has six committed twelve-seed blocks at one unchanged theta
(sha256_16 `628b58cd`: two cards and four title batteries). The per-segment
left-right phase median across them:

    row                        block medians (deg)                 spread   band edge gap
    gait.phase_lh_rh_deg   36.5  41.0  51.5  28.5  30.0  24.5        27.0          145.15
    gait.phase_lm_rm_deg   20.0  18.5  29.0  39.0  43.0  42.5        24.5          144.30
    gait.phase_lf_rf_deg  134.5  95.0 138.0  92.5 123.0 109.5        45.5           61.55

Lane B's 35.5 and lane E's 32 both sit inside the champion's own hind range of
24.5 to 51.5. So do the other twelve attempts three lanes made on this row
today, every one read at twelve seeds. Taking the range over six blocks as
about 2.5 standard deviations, the median's standard deviation at that size is
roughly 11 degrees on the hind row, and a ten-degree paired difference needs of
order a hundred seeds per arm to resolve. Only a move of order a hundred
degrees is visible at twelve, which is what closing the row would look like.

**The conflict itself is untouched by this.** The medians sit 145 degrees
outside the band, five times the spread, so v3.10's unreadable-row filter does
not reach them. What cannot be read at twelve seeds is a small CHANGE in the
row, not the disagreement with the animal.

It also settles what one contradictory receipt is not. Lane E's R428 reported
143 degrees on the hind row for a theta that is byte-identical to the fork run
today (both sha256_16 `52a517fc1a55eb38`, zero differing keys, because
`delay_region_scale` replaces rather than composes). 143 sits 91 degrees above
the top of the champion's own hind range, four times the block spread, so it is
not a block draw; under the known middle-hind label swap it reads middle 143,
which is 100 degrees above that row's range. Either the two reads labelled the
legs differently, or that body moved a phase row by about a hundred degrees and
two reads of the same theta today show nothing. One arm on the original seeds
with today's reader separates them, and it is lane E's first item.

## The second hold of the day, and what held it

wi1 is the champion with the claw afferent's phasic gain at 1.25, lane D's
body. Over the 24 paired seeds it stands 19 times against the champion's 24,
which is a drop of five against v3.9's gate of two, so it is held before
anything else is weighed.

Everything else favoured it. Under v3.10 its open-conflict count is **3 against
the champion's 4**: it closes `posture.footfall_separation` on 10 of 10 standing
card runs, median 1.0791 inside 1.02 to 1.11, where the champion reads 1.1485
on 0 of 12. The combined veto returns one regression and one improvement and
does not hold. Robin's ruling of 2026-09-06 put the conflict count above the
qualifying count precisely so a body like this could crown, and the boundary
Mistral asked for in the same conversation, standing stays a hard gate, is what
stopped it.

Both of today's nominations were held that way: wh1 at 21 of 24 standing, wi1 at
19. Neither was held by its card, and neither was held by the count. **The
binding constraint on this fleet today is falling over**, and it is the one row
the instrument reads cleanly at 24 seeds.

## What bit us

This seat published a route to Robin as the fleet's aim and it was refuted in
six minutes by the first lane to run it. The route was not chosen from nothing:
it came from a receipt reading 143 degrees on a body that no longer exists as a
separate body. The cheap check that would have caught it costs no seed and was
run afterwards rather than before: hash the nominee's theta against the fork's
before quoting a difference between them. Recorded here because the fleet is
told to under-specify method and over-specify ambition, and a judge naming a
mechanism is the failure mode that instruction exists to stop.

## Who did what

Lane B ran the cord at 0.5 without inhibition and reported the cadence cost.
Lane E ran it with crossed inhibition, on the fork this seat set, and called the
route dead in one run; its own verdict sentence attributing the 143 to a
different body cannot hold and was routed back to it rather than relayed. Lane A
withdrew a registered arm rather than spend seeds on a question two blocks had
answered. Lane D built and nominated wi1 on the row its own card closed. The
judge ran both matches, measured the block spread on the champion's committed
receipts after publishing an aim that rested on not knowing it, and refreshed
every goal file at Robin's start-of-shift instruction.

Robin's instruction of 11:30 is what shaped the goals now in front of the
lanes: "Under-specify the method, over-specify the ambition, and set a bar high
enough that the lane has to find something you did not think of."

## 14:30 correction: the measured block sizes and the missing commit boundary

The 11:46 conclusion above was too broad. Halving the cord did not move the
hind pair, but a later paired read showed a middle-pair direction. Lane B then
repeated the paired comparison on twelve fresh seeds: the middle pair moved 51
degrees on 11 of 12 pairs, from 5.5 to 56 degrees, below its registered
80-degree bar, while speed fell from 2.81 to 1.59 mm/s on every seed. The route
is held as a direction. It neither closed the grouped phase conflict nor
supported the 82-degree size quoted after pooling two discordant blocks.

The 143-degree value was also resolved. It was the middle pair under the old
leg-label order, not a different body's hind pair. Replaying the same frozen
parameters through the corrected reader returned the same interpretation. The
11:13 report's halved-cord aim and its 143-degree premise are superseded by the
paired lane-B result and this label correction.

The measured resample replaces the estimate above. Sixty distinct standing wi1
seeds from six committed receipts give 95% median widths of 40 degrees at the
front, 20 at the middle and 19.5 at the hind for a twelve-seed block. A
registered 30-degree change needs 24 front seeds and 6 middle or hind seeds;
50 degrees needs 8, 6 and 6. These are resamples of existing seeds, so they are
a lower bound on drift across future separated blocks, not a promise about a
new draw.

The audit also found a different instrument error. v3.10 and v3.11 said their
repeat-spread gate used committed receipts, but its loader globbed the mutable
worktree. Both wi1 decisions therefore saw their own match files before those
files entered the atomic verdict commit. The replacement reads Git blobs from
an explicit commit, counts an exact scientific run copy once, and proves parsed
parameter equality before joining the lane nominee wrapper to the flat champion
file. That join keeps all six wi1 blocks visible when the next holder receipt
uses the champion file's different hash. Replaying each decision from its true
parent commit excludes the same five rows as the saved verdict, so neither title
decision changes.

The fixed-heading twin now accompanies each body-frame phase row. On the same
twelve wi1 runs, front reads 40 degrees in the body frame and 11 with heading
held fixed; middle reads 36.5 and 26.5; hind reads 16 and 40.5. The animal
reference uses the body axis, so the original row remains the benchmark. The
twin tests whether a change came from foot motion rather than body yaw over
planted feet.

supersedes:: exchange/2026-09-07-1113-judge-report.md
uses-data:: lanes/judge/2026-09-07-card-wi1r-metal-phasetwin.json, lanes/judge/2026-09-07-row-block-size.json, lanes/judge/2026-09-07-instrument-gates.json
