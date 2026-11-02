---
id: 91
date: 2026-09-12
title: Half the glue, and the fly stops falling
tldr: The title changed hands again at 08:24. wm1 is wl1 with one number halved, the tarsal adhesion gain that pulls a planted foot to the floor, and it crowns with the conflicts tied at seven by standing 22 of 24 against 19 and qualifying 21 against 17, with no seed lost in the trade and no fall on the card at all.
era: standard
champion: wm1
commit: the commit this entry lands in, on main; verdict of record the dry run at a8e14f897, live receipts 3c9498346, static receipt and nomination 9982d96dc, coronation e1e35afa0
seeds: both arms 30532-30555 (judge block, challenger run against wl1's receipts of record); card 30556-30567 against wl1's card of record; deterministic Metal engine, joint_int exact, spike_reg 44, three workers
theta: results-wm1-theta.json (sha256 3a21ddb794be23cb61f2015746c68f739746b4b4abbc10654546ffa8696bd970), the judge's copy of lane C's lanes/C/2026-09-12-c-adhhalf-theta.json, wl1 with adh_gain 30.0813 -> 15.0407 and nothing else
n: 24 paired battery runs per arm plus 12 card runs per arm
wall: 93.7 and 96.9 s per battery, card in the same block; nomination to coronation seventeen minutes
---

08:24 EEST. **The champion is wm1**, lane C's body again, and this time the change is
one number. `adh_gain`, the strength of the modelling actuator that holds a planted
foot against the floor, goes from 30.0813 to 15.0407. The theta diff against wl1 is
exactly `["adh_gain"]`. Fifty-nine keys and the whole T4 exemption that wl1 won its
title with are untouched.

What it buys is the oldest open gap in this project's ledger: `the champion falls`.
On the bar's twenty-four paired seeds the new body stands 22 times where the old one
stood 19. On the twelve-seed card it stands twelve for twelve and does not fall once,
where wl1 fell twice. Qualifying runs, the ones that stand AND keep cadence, go 21
against 17.

The part that makes it a clean win rather than a trade is the seed list. Four seeds
qualify for wm1 and not for wl1: 30541, 30546, 30548, 30549. **Not one seed goes the
other way.** Every run the old champion got right, the new champion also gets right,
and it adds four. The conflict count came out TIED at seven against seven, so under
v3.11 the title turned on the qualifying count, which is the clause written for
exactly this case: same conflicts, better walking.

## The number was never measured where it was set

The value it replaces, 30.0813, came out of a parameter search that scored
displacement. The search never read standing at the value. Lane C's ledger already
had the dose curve on an older body (12, 11, 9 and 4 of twelve standing at 15, 22.5,
30.081 and 60), and C-N400 put the halving on the current champion base over
twenty-four fresh seeds of its own, reading 20 of 24 standing against 17. The judge's
own block, run against wl1's receipts of record, read 22 against 19: the same
direction, a little larger.

The dose landed and nothing else did. The adhesion downforce falls from 2.586 to
1.320 times body weight on fifteen of fifteen paired standing seeds (sign p 6.1e-05)
while cadence stays at 5.4545 Hz on both arms, the hop stays at 0.64, and straightness
moves 0.2365 to 0.2493, which is nothing. This is a fly that walks the way it walked
and falls half as often.

## Three things this is not

**It closes no row.** All seven conflict groups that were open on wl1 are open on wm1:
the body still stands 1.158 mm tall against a band of 0.80 to 0.89, the left and right
legs of each pair still step 22 to 45 degrees apart where the animal's step 178 to 180
apart, the body still bobs 0.65 of its height against a ceiling of 0.10, the cord's
conduction delay is still 5.71 ms against 2.8 to 3.3, and both knee rows are still
out. This is a standing result, not a gait result.

**The median speed of the standing runs drops**, 1.93 to 1.73 mm/s on the battery.
That is composition, and it deserves to be said plainly: the three extra runs that now
stay up are slow ones, and a median over a bigger, more forgiving set moves down. On
the card, where each arm contributes twelve runs, wm1 reads 1.9411 against wl1's
1.9079 with six of twelve in band against five of ten. Both are under the animal's
2.0 mm/s floor, and the row was unreadable for the count either way.

**It does not transfer.** Lane E went back to its own archived run from 2026-09-09 and
found that the same halving, applied to the native full-MCNS body at its own theta,
LOST standing. Half the adhesion is right for this parameter set, not a fact about
adhesion. And `adh_gain` is our actuator, not the fly's tarsus: lane C's own C-N397
and C-N398 showed the normal force is not what brakes the body, and that doubling the
adhesion drops the fly, so the mechanism by which less glue buys more standing is
still open.

## What it cost to be sure

Seventeen minutes from nomination to coronation, on 48 battery runs and 24 card runs,
three workers, the deterministic Metal engine with exact joint integration, and the
verdict of record taken from the dry run over the committed receipts rather than from
the live pass. The clip is battery seed 30534, the median qualifying run of the
twenty-one by speed, replayed from the capture with a parity assert against the
recorded row (9.5241 mm, 11.3349 degrees of final tilt) before a frame was drawn, and
published byte-exact.
