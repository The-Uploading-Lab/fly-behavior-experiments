---
id: 26
date: 2026-08-25
title: The lane outlives the detour
tldr: Lane A was moved from Claude to Codex Cloud during the judge succession without the diary-recorded holder swap the contract requires. Robin caught the mismatch from mission control. The completed Codex calibration and exact-probe void remain honest historical evidence, but the seat returns to Claude Cloud/Fable before any further Lane A work. No scientific verdict, seed, tolerance, title, or deadline changed.
era: standard
commit: 2cec1b63 introduced the detour; this entry records its correction
seeds: none
n: governance correction
wall: not applicable
reproduce: git show 2cec1b63 -- ops/seats.json lanes/judge/STATE.md
---

**Question.** Does moving a research lane into the cloud also authorize the
judge to change which provider holds its charter?

**Decision.** No. The contract says seats are charters rather than platforms,
but it also requires one holder and a diary record before a holder swap. The
August 24 migration named Codex Cloud for Lanes C, D, and E. Lane A's prior
transition work was a Claude-seat responsibility. No diary entry transferred
Lane A to Codex.

**Method.** Robin challenged the provider label visible in mission control.
The judge traced it to commit `2cec1b63`, compared that change with the seat
contract and the pre-succession record, and checked that headless Claude Cloud
is available from the canonical `lane-a` branch.

**Result.** Lane A is restored to Claude Cloud/Fable. The completed Codex Cloud
calibration and the later exact-probe void are not rewritten or discarded:
their provenance stays Codex Cloud and their measurements stay exactly as
recorded. A fresh Claude-image identity calibration must precede stochastic
Lane A science.

**Problems / what bit us.** “Move to cloud” was read as “move to Codex Cloud.”
That substituted a provider decision the user had not made, and the recurring
feeder then repeated the substitution as if it were settled governance. The
dashboard made the drift visible.

**What changed in the model.** Nothing. This corrects ownership and scheduling,
not scientific evidence.

**Attribution.** Robin caught the mismatch. The Codex/Sol judge traced,
recorded, and repaired it before dispatching the restored holder.

judge harness: Codex/Sol

— the judge
