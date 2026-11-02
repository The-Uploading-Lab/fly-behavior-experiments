---
id: 41
date: 2026-09-01
title: The phase that carries the turn
tldr: The selected fly's right turn failed on half of four fresh dice rolls. A 90-degree timing offset on four hip joints, found by sweeping past the sign its inventor tried, makes the turn survive all four while walking slightly faster. Verified three independent ways and promoted to the shared target the same afternoon.
era: standard
commit: 03f48cf62
seeds: 60217, 60219-60223
n: 1 screen seed + 4 replication seeds, paired same-seed arms
wall: ~6 min per paired seed
reproduce: python3 lanes/D/run_cv41_thc_yaw_phase_replication.py
---

**Question.** C-V41, the body where live visual neurons pick the walking
phase, was selected on one seed. Does it survive dice it never practiced
on, and if not, what carries it?

**Method.** Lane D re-ran the selected body unchanged on four fresh seeds,
paired each run with the same seed plus one addition: lane A's four-joint
thorax-coxa yaw timing offset, swept across angles first on a fifth seed.
Lane A had built that offset at -90 degrees and closed it; the sweep found
the OPPOSITE sign is what works. The replication criterion was written
before any body ran: 3 of 4 seeds must improve.

**Result.** Baseline C-V41 kept its own right-turn floor on 2 of 4 fresh
seeds. With the +90 degree phase: 4 of 4, path speed up on every seed
(median +3.45%), left yaw down 7.75% but above its floor, stop drift still
inside its band. The judge verified three ways: the four-seed run replays
bit-identically at its commit; the ported mechanism joins main with the
identity probe untouched; the decisive seed's numbers reproduce on main's
engine to nine decimals. The composition is the shared target now.

**Interpretation.** The selected body's turn was partly a property of the
one seed it was born on. A fixed timing relation between front and hind
hips makes that turn a property of the body instead. Sign matters more
than magnitude: -90 worsens the stop three-fold; +90 is the only swept
angle that keeps everything.

**Problems.** Two, both recorded beside the win. Fresh seeds barely
perturb this body: randomness reaches it only through a handful of
discrete visual-neuron classes, so four seeds carry less independence than
four sounds like. And the fragility claim itself missed its own
preregistered bar (2 of 4 against a demanded 3), so "C-V41 was fragile"
stays suggestive. Separately, a first promotion attempt silently ran the
WRONG body: the engine ignores unknown theta keys, so the un-merged phase
key no-opped into a baseline run. Caught because promotion requires exact
scalar agreement, recorded as a standing rig hazard.

**What changed in the model.** `vision_teacher_hs_thc_yaw_phase_deg` is on
main, default 0.0 = bit-identical. The shared target is C-V41 + 90
degrees. Earlier the same day, Robin's measured-structure directive put
per-neuron delays from measured cable length and the graded-transmission
dial search into the engine; lane B measured that the delay dial cannot
act on C-V41 at all (its motor-neuron authority is zero) — the sharpest
statement yet of what the selected body still is not.

**Attribution.** Robin set the day's direction twice: the consultant
seat's progress-first ruling at 13:03 and the measured-structure directive
mid-session. Lane A authored the phase mechanism and closed its own -90
route. Lane B tested -90 on C-V41 (floors held, gates missed) and shipped
the delay/graded findings. Lane D swept the angle, found +90, preregistered
and replicated it, and filmed seed 60220. The judge verified and merged.

cites:: exchange/2026-09-01-1345-laneD-judge-packet.md, exchange/feedback/2026-09-01-1349-lane-d-judge.md, AGENTS.md
uses-data:: lanes/D/2026-09-01-cv41-thc-yaw-phase-replication-result.json, lanes/judge/2026-09-01-d-thc-yaw-replication-replay-note.md, lanes/judge/2026-09-01-delay-battery-result.json
