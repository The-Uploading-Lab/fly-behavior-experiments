---
id: 69
date: 2026-09-07
title: The full-board count
tldr: The title instrument now reads every walking-card quantity already available in its captures, including the three measured leg-pair phases, abdomen excursion, delivered conduction delay and knee motion. It also accepts judge-reproduced deterministic rows. Replaying wi1's spent card block changed no existing scientific field and raised the current raw record from five readable conflict families to nine: three newly exposed dynamic conflicts, one newly exposed static conflict and one newly exposed closure. The 12:49 title remains the recorded v3.10 comparison; the next challenger is the first body judged by the complete v3.11 path.
era: standard
commit: the commit this entry lands in, on main
seeds: wi1 card replay 30556-30567, already spent at coronation; deterministic Metal engine, one worker, execution commit 39a7d2e11baea2883d64c88ed0018d1f579d5097; all prior scientific fields BIT-IDENTICAL; static T4 row reproduced on wi1 and wg1 from pushed main, no scientific seed
media_note: No new behaviour was simulated. This entry regraded existing wi1 captures and a static network receipt, so entry 68's verified wi1 footage remains the matching visual evidence.
---

At 18:19 EEST the current champion had nine raw conflict families on the board.
The number at coronation was four after v3.10's paired near-edge filter, read over
22 rows. The two numbers answer different questions. Four decided wi1 against
wg1 on the evidence available at 12:49. Nine describes wi1 alone after the
instrument learned to read all 26 dynamic rows available from a walking card and
one deterministic network row.

## What the added readers found

The four capture readers produced one closure and three conflicts on the same
twelve runs used for wi1's post-coronation card:

| row | current value | result |
|---|---:|---|
| abdomen lateral excursion | 10.0588 degrees; 12 of 12 in band | satisfied |
| delivered conduction delay | 5.7139 ms; 0 of 12 in 2.8 to 3.3 ms | conflict |
| knee direction through stance | 3 of 6 legs have the animal's sign | conflict |
| knee excursion through stance | 3 of 6 legs are in band | conflict |

The deterministic channel read the sign of the image-carrier input onto T4 from
the built network. Both frozen arms were inhibitory on every subtype: T4a
-10,278.1, T4b -18,060.1, T4c -12,425.3 and T4d -7,780.7. The animal row
requires net excitation, so this adds the ninth conflict.

The other five are body height, straightness, front femur-tibia range, vertical
bob and left-right phase. The pooled phase row and the front, middle and hind
phase rows all disagree with the animal, but they count once because they grade
one coordination family.

## Why the old verdict stays

v3.10 removes a row from a match when either arm lies within that row's measured
between-block spread of a band edge. That is a paired comparison and cannot be
reconstructed from wi1 alone. Applying the raw count retrospectively would
compare a new instrument on one arm with the old instrument on the other.

The title therefore stays with wi1 under its recorded result: four conflicts
against wg1's five over the 22-row read, with standing 19 of 24 above v3.11's
floor of 18. The next nomination runs through the complete executable path on
both arms, including the deterministic receipt channel.

## Identity and provenance

The card replay used seeds 30556 to 30567, which were already spent. All twelve
runs stood. After removing the four added rows, their three added per-run fields,
wall time and provenance text, every earlier scientific field was bit-identical
to the committed card. The static result was generated separately on pushed
main from the two frozen theta files and records the registry row contract, its
source hash and both theta hashes.

The receipts are
`lanes/judge/2026-09-07-wi1-full-conflict-regrade.json` and
`lanes/judge/2026-09-07-static-wi1-wg1-t4-sign.json`.
