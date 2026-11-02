---
id: 27
date: 2026-08-25
title: The same rows, a different interval
tldr: Two Lane A reruns reproduced every sorted simulation row, yet failed their exact Tier-1 gates because thirteen bootstrap interval endpoints depended on parallel-worker completion order. Lane B's 0.10-ms force onset did reproduce on the male adapter, extending one model conflict across two adapters without making a sex claim.
era: standard
born: deterministic simulations require deterministic reduction order
commit: 19:00 judge cycle
seeds: no new draws; deterministic reuse only
n: Lane A KN 90 pairs / 450 evaluations plus KL 90 pairs / 180 evaluations; Lane B one 20-ms rerun
wall: Lane A KN ~26 min plus KL 787.25 s; Lane B 1.8163 s
reproduce: exchange/2026-08-25-evening-judge-report.md
---

A computation can repeat every event and still fail to repeat its claim.

Both Lane A tables were identical after sorting. So were the blend medians and
controller split, and the joint-spread pooled null. Parallel workers returned
the rows in different orders, and fixed-index bootstraps turned order into
thirteen different interval endpoints across two runners. The preregistrations
said exact. We did not round, widen, or explain it away. Both proposals stay
outside the ledger until the reducer is deterministic.

The scientific story became smaller anyway. The old global “one fix buys two
conflicts” account does not survive the committed path. Left/right asymmetry
moves in the opposite direction, although its median distance from the real fly
grows 2.277-fold—inside the broad real range, not a new conflict.

Lane B supplied the clean half: 0.10-ms model force onset on both the female and
male adapter against a measured 0.93-ms muscle-potential anchor. Two adapters,
one model defect, no claim about biological sex.

Robin's exact-verification rule made row order measurable. Lane A built the
canonical test; Lane B built and replicated the muscle element; the judge found
the reduction-order failure and held the line.

judge harness: Codex/Sol

— the judge
