---
id: 79
date: 2026-09-09
title: Two fresh seeds do not predict twenty-four
tldr: The best arm of the joint search cleared its licensed fresh pair and the judge's new block contract, then stood 15 of 24 on the holder's paired title block against wi1's 19 and qualified 11 against 17. wi1 keeps the title. No model improvement is claimed.
era: standard
commit: the commit this entry lands in, on main
theta: lanes/B/2026-09-09-loco-09766-theta.json (file sha 64a1b5cf, canonical winner 8c5e21a8); holder results-wi1-theta.json unchanged
seeds: B fresh 81572 and 81573 under license; judge paired 30532-30555 (the holder's own title block); judge 30568-30579 licensed for the card and not spent
n: two fresh pairs (four whole-CNS runs) and twenty-four paired title runs, single worker
wall: 246.4 s and 243.3 s for the two twelve-seed batteries on the deterministic Metal engine
reproduce: lanes/judge/2026-09-09-loco-09766-title-finalize.json, lanes/judge/2026-09-09-title-v38-loco-09766-verdict.json, exchange/2026-09-09-judge-loco-09766-title-license.json
budget_exception: fourth entry of the day because the title match against the champion was decided today and the third entry (78) is lane D's flight motor record; a title verdict is recorded where it happened
video_receipt: publication-receipts/loco09766_centre_81573.sha256-fb757e26360a6ce0a85556ca382fb3f62c4272a32c7b3ee6e7f4d2785dafe6e8.publication.json, publication-receipts/loco09766_centre_81548.sha256-af3258e869374fdb9cff46d9aa53a4e5dcb18c816234807e0fbc5a7a3075bb6e.publication.json, publication-receipts/posture_ctr20_fti5_81547.sha256-64237acea8d7d0af684c8e1518f62e937e6e1bb4e22261253a737573c946aa5e.publication.json, publication-receipts/posture_ctr20_fti5_81548.sha256-c12d6494eff4ab61865db2031e65ed33c5fbc187fcc3331d9112ab0b33e339b2.publication.json
---

**Question.** Lane B's joint search over 11,578 type output gains, nine
exceptional cells and 17 muscle gains produced loco-09766, the centre of its
best family. On its two selection seeds it had fewer grouped constraint
conflicts than the champion wi1 (7 against 9 on 81548, 7 against 7 on 81547).
Does that hold on seeds nobody selected on, and does it hold on the block the
title is decided on?

**What happened, in order.** At 22:25 the centre cleared its licensed fresh
pair: on 81572 and 81573 it stood, lost no readable row, and counted 7 grouped
open conflicts against the holder's 11 and 9. That made it the BUILD lead of
the joint family. The successor gate then refused a title match until the judge
adopted a block contract for the three parameter blocks, with instrument
receipts for 25 selection rows and five held-out rows; that was done by 22:55
and the contract passed preflight. At 23:08 the judge ran the title battery on
the holder's own paired block, 30532 to 30555, so that every seed is graded on
both flies.

**Result, simulated.** loco-09766 stands on 15 of 24 seeds and qualifies on
11 (standing, endogenous timing, every foot lifting at least five times,
cadence 5 to 16 Hz). wi1 stands on 19 and qualifies on 17. The protocol's
verdict is HOLDER KEEPS (count); the v3.11 floor of 18 standing seeds in 24
is also missed, so the card on 30568 to 30579 was never run and those seeds
remain unspent. The centre falls on nine seeds where wi1 stands, final tilt
about 172 degrees, and on four standing seeds its cadence is below 5 Hz. It
qualifies alone on 30533, 30539 and 30550. Where both flies stand, only one
selection row differs: the centre's walking speed median is 1.90 mm/s, below
the 2.0 mm/s floor, against wi1's 2.26. The five held-out rows (middle and
hind joint ranges, stride length) pass on both flies.

**Reading.** The centre trades standing and speed on unseen seeds for fewer
conflicts on the seeds it was chosen on. Two fresh seeds standing is weak
evidence against that: at the centre's observed 62 per cent standing rate,
two of two stand about four times in ten. The fresh-pair clearance and the
block contract did their jobs, which was to make the title battery
admissible; the battery then answered the question. Nothing in the shared
model changes. wi1 remains the champion of record; loco-09766 remains the
lead of its family, defeated once on this block.

**The films.** B26 shows the centre on fresh 81573, the arm that cleared the
pair (displacement 12.68 mm, tilt 11.69 degrees, parity exact to the archived
row). B25 shows it on spent 81548. Both receipts verify the public bytes
against the local masters.

**The posture films, a mechanism and not a candidate.** B27 and B28 show the
wi1 holder with one added rest-offset key, CTr pitch -20 and FTi pitch -5
degrees on all six legs, on spent 81547 and 81548. Body height drops from
1.21 to 0.87 and 0.89 mm, into the animal's band, and the front pair moves
from 32 to 169 and 178 degrees apart, near antiphase; the cost is speed
(2.90 to 1.67 mm/s) and the frozen reader calls the fly not standing, so the
judge's counter opens ten rows against the holder's five. The dose is not
proposed; the relation between posture and front-pair phase is the finding.

**What follows.** The judge will ask Robin at noon whether a build lead should
show standing on at least ten of twelve spent seeds before a title block is
licensed, since a two-seed clearance costs a fresh pair and predicts little.
Lane B continues on the mechanism side: the posture set point that flips the
front-pair phase toward antiphase (packet 23:05) and the body side of the
left-right lopsidedness lane C measured on the champion (packets 22:54 and
N216). Lane D tests the centre's motor-state selection on its own spent seed
(R115), because the centre remains the best-integrated candidate the joint
search has produced, title or no title.

Source records: `exchange/2026-09-09-judge-loco-09766-fresh-license.json`,
`lanes/judge/successor_block_contracts.json`,
`lanes/judge/2026-09-09-loco-09766-successor-preflight.json`,
`exchange/2026-09-09-2305-laneB-judge-packet.md`.
