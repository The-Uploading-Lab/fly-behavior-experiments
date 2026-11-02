---
id: 40
date: 2026-09-01
title: Live visual neurons set the gait phase; a movie-following halt still kills the late turn
tldr: C-V41 is the shared BUILD body. A locomotor halt can follow a shifted movie, but the opposite turn collapses, so the halt stays out of the shared engine.
era: standard
video: c-v41-live-hs-phase-stop-transfer-seed59030.mp4
video_url: https://media.theuploadinglab.ai/c-v41-live-hs-phase-stop-transfer-seed59030.mp4
video_master: vision/c-v41-neural-phase-stop-transfer-exact-replay.mp4
commit: 66a776ea
theta: C-V41 vision_phase_neural_motion_control=true, vision_phase_from_detector absent; r23 winner otherwise
seeds: 59030 reused C-V41; 80235-80237 Lane B halt family
n: one C-V41 assembled-main replay; one quantum-halt replay; audited C-V42 and R36 no-gains
wall: 53.1 s C-V41 replay; 49.7 s quantum-halt replay
reproduce: .venv/bin/python lanes/judge/replay_c_v41_assembled_main.py; isolated 5c0c4017 replay of lanes/judge/replay_quantum_halt_shift1s.py
---
**Question.** Can live horizontal-system spike counts replace the direct
pixel-detector overwrite that set moving gait phase, and can a locomotor halt
follow a movie that has been shifted by one second without losing both turns?

**Predicted.** C-V41 claimed a better left-stop-right body after removing the
detector overwrite. Tick 28 required any promoted halt to move with the image
and keep both turns.

**Method.** The C-V41 theta was replayed on the assembled main engine, seed
59030, 6000 ms, full female BANC. The Lane B quantum halt (cancelled world
motion below half of one commanded panorama step, three quiet ticks) was
replayed on isolated commit `5c0c4017`, seed 80236.

**Result.** C-V41 matched the committed body exactly:
+36.385/+3.512/-33.558 deg/s, tilt 3.152 deg, lifts 41/25/26/48/20/41. The
quantum halt also matched: pause 3257.5-4750.0 ms, new-window path 0.000 mm/s,
right yaw -3.143 deg/s. C-V42 halted at 0.098 mm/s and missed both turn floors.

**Interpretation.** Live HS activity is now a sufficient moving-phase source
for the selected S-turn. A halt can be triggered from sensed world motion, but
that law is not yet compatible with the late turn.

**Problems.** The C-V41 runner still labels itself `INSTRUMENT_FAILURE`
because it demanded a detector decoder after correctly emitting null. The raw
artifact was kept. The selected static brake still uses an absolute-time
window.

**What changed in the model.** Default-off
`vision_phase_neural_motion_control` is on main. The halt engine is not.

**Attribution.** Robin launched the Cursor judge seat at 11:16. Codex left an
uncommitted C-V41 assembly. Cursor verified both bodies and recorded the halt
hold. Lane C authored C-V41. Lane B authored the quantum halt.

cites:: exchange/2026-09-01-1125-judge-report.md, vision/results-c-v41-neural-phase-stop-transfer-build.json, lanes/B/2026-09-01-proto-quantum-halt-shift1s-result.json
uses-data:: lanes/judge/2026-09-01-c-v41-assembled-main-replay.json, lanes/judge/2026-09-01-b-quantum-halt-shift1s-replay.json

**Postscript, 2026-09-01 12:49 EEST.** Robin relaunched the lab and the judge
seat moved from the 11:16 Cursor session to a Claude Code terminal session
(Opus 5) on the Pro. One holder was verified from lab state before this
holder ruled: the judge pane's live process is this session, the only running
cursor-agent belongs to Lane B's worktree, and the Cursor judge's 45-minute
watcher died with its pane, which is why the announced 12:10 cutoff never
fired. Signature from here on: `— Claude Code (Judge)`. Every open obligation
in `lanes/judge/STATE.md` transfers intact.
