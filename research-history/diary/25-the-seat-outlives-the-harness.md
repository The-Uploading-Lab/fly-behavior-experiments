---
id: 25
date: 2026-08-25
title: The seat outlives the harness
tldr: Robin transferred the judge seat from the Claude judge to Codex/Sol. The role, evidence bar, reference instrument, signatures, seed block and open obligations do not reset with the software holding the seat. Codex/Sol began from clean main at d5da936b, read the in-flight state and governing contracts in the required order, and inherited the 19:00 and 22:00 debts without shortening verification. No scientific verdict changed in this handover.
era: standard
commit: d5da936b
seeds: none
n: governance handover
wall: not applicable
reproduce: git show d5da936b:lanes/judge/STATE.md
---

**Question.** Can the judge change harness without creating a second judge or
quietly resetting unfinished work?

**Decision.** Robin's 2026-08-25 instruction was explicit: “the seat, not the
harness, carries the role.” The Claude judge retired from the seat; Codex/Sol
assumed it. There is one holder, not two.

**Method.** The new holder started from clean `main` at `d5da936b`, then read
`lanes/judge/STATE.md`, `JUDGING.md`, `AGENTS.md`, and the recent constraint
ledger in the required order before making a ruling. The champion remains r23;
the Pro remains the reference instrument; the judge seed block remains
30000–36999; every deadline debt rolls forward intact.

**Result.** This is a governance transition only. No model result, title,
tolerance, registration, seed ledger, lane credit or scientific verdict changed.
The in-flight 19:00 and 22:00 obligations remain owed exactly as handed over.

**Problems / what bit us.** The previous session's wake timer belonged to that
session and did not transfer. This seat is Robin-triggered until a separate,
explicit wake mechanism is recorded; no invisible timer is claimed.

**Attribution.** Robin chose the succession and defined the continuity rule on
2026-08-25. Codex/Sol performed the ordered state reconstruction and accepted
the inherited verification debt.

judge harness: Codex/Sol

— the judge
