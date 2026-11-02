---
id: 88
date: 2026-09-11
title: The front knee opens, and six runs fall
tldr: The title changed hands again at 23:52. wk1 is wj1 with one change from lane C, the front femur-tibia joint's working-range cap raised from 96.6 to 125 degrees, and it crowns on the count by one row, 8 open conflicts against 9, because the front knee's range reads 103.5 degrees inside the animal's 81.2 to 112 for the first time on any champion. It pays with standing, 18 of 24 where wj1 stood 24, exactly the floor, and with card speed. Nothing in the rhythm moved.
era: standard
champion: wk1
budget_exception: Sixth entry of the day because the title changed again at 23:52 and a coronation is recorded where it happens; entries 83 to 87 were already down when the verdict of record landed at 21:43 EEST.
commit: the commit this entry lands in, on main; verdict of record 7eccad1e2, receipts 2d9581117 and c0ae83d67, block contract and static receipt 11f595793, nomination 6adddc32c
seeds: both arms 30532-30555 (judge block, challenger run against wj1's receipts of record); card 30556-30567 against wj1's card of record of the same day; deterministic Metal engine, joint_int exact, spike_reg 44, three workers
theta: results-wk1-theta.json (sha256 18dbba87a50a1b55ee6675cc3606a4d3ef781efa68b7c51c9ca548ad995d005c), the judge's byte-identical copy of lane C's vision/theta-c-fti125-nomination.json, wj1 plus rom_override FTi_pitch f 125.0 and nothing else
n: 24 paired battery runs per arm plus 12 card runs per arm
wall: 93.1 and 88.0 s per battery, 92.8 s for the card, three workers; the whole instrument under six minutes
---

23:52 EEST. **The champion is wk1**, lane C's body. It is wj1 with one change: the
front femur-tibia joint of the front legs, the knee, may now swing through 125
degrees where the body used to stop it at 96.6, the walking range Haustein measured
at about 15 mm/s and the model had been using as a hard limit. Fifty-eight keys of
fifty-nine are wj1's.

The front knee row has been open on every champion since it was written. The cord
was never the reason: lane C read the command to that joint as saturated (C-N335),
asking for the full range on every step, and the body delivering 0.66 to 0.86 of
whatever the cap allowed (C-N328, N329, N337). With the cap at 96.6 the achieved
range sat at 77.6 degrees, under the animal's 81.2 start, on ten of twelve card
runs. C tried the neural route first, the command's own frequency, and found a
cliff: standing 11, 5, 2, 0 of 12 as the delivered share rose (C-N336). The cap was
the one route that stood. At 125 the knee delivers 103.5 degrees, inside 81.2 to
112 on all ten standing card runs, and the row closes: 8 open conflicts against 9.

The verdict is the instrument's. Both batteries pass (10 of 12 and 8 of 12
standing); standing 18 of 24, exactly the floor of 18, where wj1 stood 24; card veto
one regression, speed 3 of 10 in band against 10 of 12, against one improvement, the
front knee 10 of 10 against 2 of 12, not held; floor veto no change read on the hop;
then the count, 8 against 9. Qualifying 16 against 21 was not needed. The dry run
over the committed receipts (7eccad1e2, 21:43) is the verdict of record; the live
run reads the same.

Read plainly. The row it closes is a body cap, not a change in the nervous system,
and its price is on the record. Six paired runs fell in their first six millimetres
where wj1 stood (30534, 30542, 30548, 30549, 30552, 30554); one more and the ordering
would have held the title before any count. `gap.the_champion_falls`, which wj1 had
repaid at 15:50, is open again. Card speed 1.92 mm/s against 2.14. Straightness 0.22
against 0.31, both far outside 0.50 to 0.98. And the rows Robin named at 16:35 did
not move: the left and right legs of each pair still step 30, 25.5 and 17.5 degrees
apart where the animal's step 178 to 180 apart (wj1 20, 36, 22), the body still hops
(0.64 against at most 0.10), and the left hind foot is still planted, stepping at
2.18 Hz against the right's 6.64 on ten of ten card runs.

What the coronation fixes for the next challenger is the base. Under the fixed
ordering a body on wk1 with the same eight rows open that stands more than 18 and
qualifies more than 16 crowns on the qualifying count, which is how wj1 took the
title this afternoon; the rom_override box stays open from 96.6 to 137.5 degrees for
a cap that keeps the knee row and gives the six falls back. A body that brings any of
the three left-right phase rows into band takes the count to 7 or fewer. Lane B's
21:39 read gives the first located reason a pair might not alternate: in this EM
volume the left front knee's extensor pools receive about a quarter of their
mirror's synaptic input (FETi 2,392 against 8,870), and the champion's front pair
commands carry opposite signs. That arm is running as this is written.

The clip is judge battery 1 seed 30532, the upper median qualifying run of sixteen
by speed (1.86 mm/s, 5.45 steps a second, 10.24 mm, tilt 10.5 degrees, knee 103.75
degrees); wj1 stood on this seed but failed the cadence clause at 4.73 Hz. Replay
parity against the battery row: 10.2359 mm against 10.24, 10.4983 against 10.50
degrees. Receipt
`publication-receipts/walker_wk1_body.sha256-4de2328bdbff6c11323a0e5a527abef1fc2bdb46bc12a776150f624c68cf6e1a.publication.json`.
Model card `MODELCARD-wk1.md`; holder receipts from here
`lanes/judge/2026-09-11-title-v36-c-fti125-battery{1,2}.json` and
`lanes/judge/2026-09-11-card-c-fti125-metal.json`.

## Defences, 2026-09-12 00:37, 00:54 and 01:12 EEST

Lane B put two doses of one change on wk1, the same night. The change is two
scalars moved together so that the foot's release level stays where it was
(0.004810) while the joint gain rises: `drive_scale` 0.0025 to 0.0080 with
`adh_thresh` 1.9242 to 0.6013, and then the middle rung of B's own ladder,
0.0055 with 0.8746. Same instrument, same seeds, three Metal workers, both
matched the pass they landed (matched files
`lanes/judge/2026-09-12-b-ds080-adh-wk1-theta.json`, sha256 45ce57ee, and
`lanes/judge/2026-09-12-b-ds055-adh-wk1-theta.json`, sha256 fe20d602, B's own
bytes).

At 0.0080 the fly stands on all 24 paired runs, every one of the six seeds that
fell under wk1 included, and cuts the bob to 0.57 from 0.64. It pays in the
step: the cadence reads 4.86 steps a second on the block against the qualifying
clause's 5.0, so it qualifies 10 against wk1's 16; on the card the feet land
1.134 apart against the animal's 1.02 to 1.11, the front knee narrows to 84.9
degrees (still in band) and the front coxa-trochanter range drops under its
72.2 floor. Three card regressions against one card improvement (contra
phase), the bob improvement under the floor rule beside them; held. Open
conflicts 10 against 9.

At 0.0055 the step keeps its rate (5.09 Hz; qualifying 19 against 16) and the
fly stands 23 against 18, the bob 0.59. Both of route (i)'s counts are cleared.
What holds it is two card rows within one per cent of their edges, both about
the front legs' stance: footfall separation 1.119 against the band's top 1.11
(wk1 1.091), and the front coxa-trochanter range 71.4 degrees against its 72.2
floor (wk1 75.0). Two regressions against one improvement; held. Open conflicts
9 against 8: it opens the footfall row and closes nothing. HELD BY THE CARD,
both; wk1 keeps.

Reading: the joint gain gives standing back and widens the front track and
narrows the front hip as it narrows the knee. The fork for lane B is in its own
receipts: the 0.0040 card files already hold both rows for six seeds and cost
nothing to read. Neither dose moved the rhythm: the left and right legs of each
pair step 13 to 35 degrees apart against the animal's 178 to 180. Verdicts of
record `lanes/judge/2026-09-12-title-v311-b-ds080-adh-wk1-verdict-DRYRUN.json`
and `lanes/judge/2026-09-12-title-v311-b-ds055-adh-wk1-verdict-DRYRUN.json`.
No clip: a held challenger earns none.

Then the lowest rung, at 01:12. Lane B answered the fork from its own receipts
within the hour: 0.0040 (with `adh_thresh` 1.2026) is the only dose whose feet
land inside 1.11 and whose front coxa-trochanter range stays above 72.2, and on
twelve unseen seeds it stood 12 of 12 where wk1's base stood 9. Matched the pass
it landed. On the judge block it keeps the same eight rows open as wk1, stands
22 against 18, and the card does not hold it (two regressions, left-right
asymmetry and the front hip, against two improvements, speed and the bob). It
qualifies 15 against 16. Equal conflicts are decided by the count, and the count
is one seed short: the two paired seeds where this dose falls and wk1 stands and
qualifies (30540, 30543) are the whole margin, and both stood at 0.0055. HOLDER
KEEPS; wk1 keeps by one seed. Three doses of one change now bracket the trade on
one block: 0.0040 keeps the rows and counts 15, 0.0055 counts 19 and opens the
track, 0.0080 counts 10 under the cadence floor. If a dose exists that does
both, it sits between 0.0040 and 0.0055. Verdict of record
`lanes/judge/2026-09-12-title-v311-b-ds040-adh-wk1-verdict-DRYRUN.json`.
