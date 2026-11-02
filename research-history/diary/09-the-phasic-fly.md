---
id: 9
date: 2026-08-17
title: The whisper becomes a fly — the phasic champion
tldr: We traced the coordination failure to sensors that report levels instead of changes, fixed the transduction, and a new champion doubled the speed while standing better.
era: standard
video: walker_p150.mp4
commit: a4452e2
theta: results-p150-theta.json
seeds: battery 820-825 (fresh; chamber never touched them)
n: battery n=6; chamber 180 candidates x 3 seeds
wall: chamber 2.1 h; battery ~2 min
reproduce: .venv/bin/python3 regression_walk.py --theta results-p150-theta.json --reuse 820,821,822,823,824,825
---
**Question.** Why does a cord that demonstrably walks keep its legs
unphased — and what would let it hear its own legs?

**Predicted (pre-registered, quoted).** From the TR1 registration:
"T1 = closed-loop leg PLV rises above per-arm rolled floors with
dose -> transduction depth was the bottleneck." From round 18: "a
challenger takes working-fly status only if its battery beats t025
on standing (> 2/6) AND speed (> 0.52 mm/s) with all-feet lifts >= 5."

**Method (jargon defined inline).** A four-probe chain recorded named
neuron populations while the fly behaved. It found the leg cycle
reaches the cord only as a whisper: our sensors are tonic-dominated —
they fire at their absolute level (~76 Hz) with the movement barely
modulating them, where real fly proprioceptors are phasic (they fire
on CHANGE). We added one default-off parameter: subtract each
sensor's own running baseline and amplify the residual. Then a
180-candidate search (chamber) with that dial searchable, judged on
stand-and-step displacement, coordination scored only by the
validated instrument.

**Result (numbers, with variance).** The dose test moved all three
outcomes together — the first intervention ever to do so
(coordination excess -0.034/-0.013/+0.005; worst-leg lifts 6/10/13;
standing 0/4, 2/4, 3/4 across doses 0/0.5/1.0). The chamber pinned
the dial at the ceiling we allowed (1.5). Battery, per seed
(seed: mm, upright): 820: 3.68, 0.76 | 821: 5.66, 0.99 | 822: 3.38,
0.96 | 823: 5.17, 0.08 | 824: 7.08, 0.97 | 825: 0.97, 0.86.
Standing 3/6, speed 1.03 mm/s (2x the incumbent), min lifts 9, all
six joint ranges in-band. Title transferred.

**Interpretation.** The stand/walk trade that defined the whole
project broke on physiology, not on search cleverness: a cord that
hears changes can afford to move. Honest speed is now 1.03 against a
real-fly floor of 2.0 — inside 2x for the first time.

**Problems / what bit us.** (1) An intermediate battery (t025 +
phasic, before the search) went 0/6 — the fourth time a small-n
result collapsed on fresh seeds; only the paired comparison
survived, which is exactly what paired designs promise and no more.
(2) The video renderer predates the new parameter and silently ran
the champion WITHOUT its phasic sensing — a different animal that
fell. The parity rule (video numbers must match the record to the
digit) caught it; the fix is one line; the first render was
discarded.

**What changed in the model.** `phasic` joins the searchable
parameter space; round 19 is registered with the ceiling raised to
3.0, because the chamber voted for more.

**Attribution.** Human: the premotor-inhibition hypothesis that
sharpened the probe series; "run it" / "go" at two decision points;
the standing directives (graded transmission, best-practice
reporting) landed the same day. Agent: the probe chain and its four
self-caught design flaws, the transduction parameter, the chamber,
the batteries, the renderer bug and its catch.

**Video.** walker_p150.mp4 — seed 824, 7.08 mm at upright 0.97, the
longest standing closed-loop walk in the project. Parity-exact.
