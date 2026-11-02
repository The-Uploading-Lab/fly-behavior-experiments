---
id: 31
date: 2026-08-29
title: The migration completes
tldr: The last seat of Robin's terminal migration is recorded — Lane D moved from Codex Cloud to a Claude Code terminal session in fly-wbe-D, declaring itself in its own registration before first use with the new signature "— Claude (Lane D)". The judge verified one holder (no live Codex lane-d task, no Codex push after 15:58) and accepts the swap. Lanes C and E continue as Codex seats now running locally in the lab tmux. Third and final entry of the day, and the closing instalment of one arc: judge (diary 29), Lane B (diary 30), Lane D (here) — six seats, one machine, four Claude and two Codex.
era: standard
commit: fb49774d
seeds: none
n: governance handover
wall: not applicable
reproduce: git show fb49774d:exchange/2026-08-29-laneD-q3bw-dna13-mirrored-motor-axis-preregistration.md
---

**Question.** Lane D's 22:00 packet declares a platform change and asks the
judge to record it. Is it one holder, Robin's action, recorded before acting?

**Decision.** Accepted. Evidence, judge-gathered: the declaration sits in the
lane's own Q3bw registration (`fb49774d`) with the signature registered
before first use, per the rule; the session runs in Robin's lab tmux (pane
lab:1.4, fly-wbe-D) and quotes a Robin in-session directive; `codex cloud
list` shows no live lane-d task; git shows no Codex authoring after
`3d891f25` (15:58). Prior Codex-signed artifacts retain their provenance
unre-signed, as the declaration itself states. Any future Codex-signed push
to lane-d is an anti-race violation the judge reverts.

**One recorded caution.** lab.conf's seat rows for lane-c and lane-d are
crossed relative to what actually runs (conf says c=Claude/d=Codex; reality
is c=Codex-local/d=Claude). The conf is Robin's file; the record of truth
for seat identity is each lane's registered declaration plus the observable
session — written here so nobody audits seat identity from the conf alone.

**Result.** Governance only; no scientific verdict changes. The fleet now
runs judge, A, B, D on Claude terminal seats and C, E on local Codex seats,
all in one tmux on the Pro. This entry closes the day's migration arc
(diary 29: the judge; diary 30: Lane B; here: Lane D) — recorded as three
instalments only because the swaps arrived hours apart, each needing its
record before its holder acted further.

**Attribution.** Robin chose every seat's platform. Lane D wrote the
lane-side declaration unprompted and asked for the judge-side record. The
judge verified and accepted.

judge harness: Claude Code (Opus 5, ultracode)

— the judge
