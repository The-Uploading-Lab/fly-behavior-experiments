---
id: 62
date: 2026-09-06
title: The title that went against the count
tldr: we1, wd2 with one dial moved so each step is 44 per cent longer, is the first champion inside the real flies' speed band and the first crowned while qualifying FEWER runs than the holder (22 against 23); v3.9 orders on conflicts with measured data above a standing gate, and it closed one.
era: standard
champion: we1
video: walker_wd2_body.mp4
video_caption: wd2, the previous champion, on its median qualifying run; compare its shorter stride with we1 in the next player.
commit: the commit this entry lands in, on main, tagged we1
seeds: judge 30448-30471 both arms (item 8 fresh block), card 30472-30483; lane E 101500-101523 for the nomination and the clip
---

12:54 EEST. **Champion: we1.** Theta `results-we1-theta.json`, one key different
from wd2: `drive_scale` 0.005947 to 0.0025.

## What moved

Forward speed. Every champion since the speed row was written read 1.3 to 1.6
mm/s against an animal band starting at 2.0. On the fresh block 30448-30471 wd2
reads 1.540 mm/s over its 23 standing runs and we1 reads 2.115 on the same
seeds; the card puts 10 of 12 runs in band against 3 of 12. Cadence did not
move (5.55 against 5.73 Hz), so the gain is step length: 0.278 to 0.407 mm on
lane E's seeds.

## How the day got there

At 12:13 lane E measured that speed tracked cadence at r = 0.905 across eleven
bodies and that no dial anyone had turned moved step length, so the speed
conflict was a stride conflict (R419). It then found the one dial that does:
`drive_scale`, which sits before the joint's tanh saturation, so a smaller value
widens the joint sweep at the same command. Two twelve-seed batteries at 0.0025
read 2.271 and 2.069 mm/s, 11 of 12 standing each, with no card row's median
leaving its band, and lane E nominated at 12:36 with the theta hash frozen.

Lane A, in the same hour, found the second stride lever from the animal's side:
the middle coxa-trochanter cap at the tracked ratio takes wd2 to 1.974 mm/s
(paired +0.330 over 21 pairs, interval excluding zero), and the full closure of
that range gap, both middle joints, gives the gain back. It did not nominate on
a 24-seed screen; the composition with we1 is its next fork.

## The match, and the clause that decided it

Item 8 put both arms on a fresh 24-seed block. Standing 23 against 23. The count
went to the holder, 22 against 23. Under every bar before this one the holder
keeps the title there. v3.9, Robin's decision at 11:4x with Mistral's standing
gate as the boundary, orders on open conflicts with measured data first: 6
against 7, speed closed, nothing else different between the arms. The card and
floor vetoes tied two against two (speed and the front femur-tibia range toward
the animal; swing duration and the vertical bob away), and a tie does not hold.
So it crowns, on the rule as written before its own seeds ran.

## What it costs, in the record beside the win

One qualifying run. Swing duration at 9 of 12 in band (median 57.5 ms against a
60 ms ceiling). The vertical bob 0.034 body heights worse on 17 of 22 paired
seeds. Six conflicts still open, opposite-leg coordination and the bob first.
The lever has no measured anchor; the ledger keeps it as searched and calls it
a scaffold.

## Who did what

Robin: the bar (v3.9, live in session, 11:4x: "closing a real conflict may cost
a step"). Mistral: the standing gate that makes it a rule. Lane E: the finding,
the nomination, the clip. Lane A: the second lever and the cancellation. The
judge: the instrument (`title_match_v39.py`, 12:02), the match, the grade; it
built nothing it graded. Lane D's claw-phasic nomination was held twenty minutes
so it could be rebuilt on the new holder; lane D began those batteries at 13:04.

Run metadata: engine Metal, one worker, exclusive lock, `joint_int: "exact"`,
spike_reg 44; holder arm 12:32 to 12:44, challenger arms 12:39 to 12:48, card
12:48 to 12:54; instrument `lanes/judge/title_match_v39.py` at the commit this
entry lands in.
