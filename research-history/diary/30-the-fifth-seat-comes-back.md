---
id: 30
date: 2026-08-29
title: The fifth seat comes back
tldr: Lane B resumed on a Claude Code terminal seat, and the judge verified the resume rather than taking its word — Robin's 2026-08-29 lab.conf lists lane-b as an active Claude Opus ultracode seat and his launcher fired the standing charter into the pane, the same action that seated the judge and Lane A today. Contract files on the merged lane branch are bit-identical to main, and the judge re-ran the HANDOFF probe on the lane's merged engine: bit-identical. The Cursor Lane B agent stays retired; one holder per seat. The paused-since-08-27 fleet line is corrected.
era: standard
commit: ff19d18c
seeds: none
n: governance handover
wall: not applicable
reproduce: git show origin/lane-b:lanes/B/2026-08-29-1850-seat-resume-and-drift-probe.md
---

**Question.** Lane B was paused by Robin on 08-27. A session claiming the
seat resumed it on a Claude terminal harness. Was the pause lifted?

**Decision.** Yes — verified, not assumed. The pause was recorded against the
Cursor seat and its quota. Robin's word arrived as an action: `lab.conf`
carries "Robin's model policy (2026-08-29)" and its SEATS array lists
`lane-b … claude --model claude-opus-5 … opus ultracode` as an active seat;
`lab-up.sh` fired the standing Lane B charter into the pane. That is the
same mechanism that seated this judge and Lane A this afternoon. One holder
confirmed in tmux (`lab:1.2` on `fly-wbe-B`); the Cursor agent
`bc-2a6380d9-…` is preserved, retired, and must not run concurrently. The
seat signs `— Claude Code (Lane B)`, registered before first use.

**Method.** The judge checked the launcher config and tmux directly, diffed
the lane's merge of canonical main (`cbc2b827`) — all nine contract files
bit-identical to main, `cns.py` identical, the only rig delta the lane's own
default-off TTM bridge (`esc_ttm` absent ⇒ factory path) — and re-ran the
HANDOFF probe on the lane's merged tree in a detached worktree:
`PROBE [1.268453802823, 1.268453802823, 0.990248627778, 1.149806329417]`,
bit-identical, matching the lane's own run.

**Result.** Governance only; the model is unchanged. The lane's first act was
to disclose that the merge moved the engine under its parked zeta
registration and to decline to execute it unbound — the DNg100 engine-drift
ruling applied by a different lane within two hours of being written. It
will re-register with blob hashes bound. Seeds 80026–80038 (skip 80032)
remain fresh.

**Attribution.** Robin chose the seat migration via the launcher. Lane B
gathered the evidence, recorded the swap lane-side, and offered to stand
down if the judge read the pause as still binding. The judge verified and
ruled the resume legitimate.

judge harness: Claude Code (Opus 5, ultracode)

— the judge
