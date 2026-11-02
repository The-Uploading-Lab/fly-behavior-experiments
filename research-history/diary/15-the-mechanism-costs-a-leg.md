---
id: 15
date: 2026-08-19
title: Correct physics costs a leg
tldr: We gave the model the synapse it should have had all along — one where excitation and inhibition divide each other instead of cancelling — and the fly stopped walking. Fixing our own bug in it revealed that the parameter to carry across was never the gain but the excitation/inhibition balance, and the analytic prediction for it overshot by half. With the balance mapped, true shunting is affordable and the coordination result survives it. What does not survive is all six legs: under shunting at least one leg stops lifting, every run.
era: standard
born: map a model change on the invariant it preserves, not on the parameter it looks like
video: walker_mc1_first.mp4
commit: arc 8768020..1f13188
theta: results-f23-theta.json (champion, unchanged); results-round-f26.json and f28 as controls
seeds: CB6/CB7 8600-8611; CB4 8600-8611 both animals; CB5 local search 8601/8604/8607; STD1 9300-9319 (all fresh)
n: 42 + 72 + 96 + 96 + 141 + 100 runs, 6 s each
wall: 2026-08-19, morning to night
reproduce: .venv/bin/python3 cb7_balmap.py   # then cb4_panel.py female
---
**Question.** Real synapses are conductances: excitation and inhibition
arriving together do not cancel before they reach the membrane, they
*divide* — the cell's own input resistance collapses and every signal
shrinks. Our model cancelled them. We implemented the correct version
(**true shunting**). Does the fly survive it, and do the results we have
built on the shortcut survive with it?

**Predicted (pre-registered, quoted).** Written before the balance runs:
"if the silence is a SATURATION (the network pinned high) rather than a
hush, motor firing under shunting at the inherited balance will be ABOVE
the cancellation baseline, not below — and the analytic bal* will bring
both firing and behaviour back toward it. If firing is BELOW baseline,
this hypothesis is wrong." And before the 12-seed sweep: "if firing rate
is the right invariant to map on, the BEHAVIOURAL optimum sits within
±0.10 of the rate-matched value."

**Method (jargon defined inline).** Three instruments. A **gain sweep**,
scaling every synapse together. A **balance sweep**: our network scales
inhibitory weights until each neuron's excitation/inhibition weight ratio
hits a target (1.10 at the operating point), and this moves that target.
And a **local re-search** over the six network-side parameters, which is
what our own cross-model mapping document says should be enough — "tens
of evaluations, not hundreds" — if the mapping is right.

**Result (numbers, with variance).**

The first answer was wrong, and it was our own bug. Turning shunting on
silenced the champion completely, a gain sweep found an optimum near 4x
with saturation beyond, and that 4x went into `map_theta.py` as the
project's first empirical constant. It lasted about six hours. **The
inhibitory conductance pool was never decayed** — the excitatory one was,
one line above it — so it accumulated without bound and pinned every
neuron at the inhibitory reversal potential. The tell was a dial pointing
the wrong way: **more inhibition produced MORE firing, 0.00 → 324 motor
spikes per tick**, which is impossible. Everything measured under
shunting before the fix is withdrawn, the 4x included.

With the mechanism repaired, the registered prediction came in
**confirmed**: shunting at the inherited balance does not hush the
network, it **saturates** it — **148 motor spikes per tick against the
cancellation model's 9.3**. Weight balance is not current balance. At the
measured operating point of −53 mV, excitation drives on 53 mV of
driving force while inhibition drives on only 17, so matched weights
leave roughly a threefold excess of excitatory current.

That gives an analytic correction with no free parameter — multiply the
balance target by 17/53 = 0.32 — and it **overshoots**, landing at half
the reference firing rate. Rate-matching the measured curve gives 0.48:

    balance target   1.10    0.90    0.75    0.60    0.53    0.45
    motor spikes/tick 148    41.4    24.1    13.0    9.34    7.00
                                          (cancellation model: 9.32)

Twelve seeds put the rate-matched point at 9.34 against a target of 9.32.
The pre-registered behavioural optimum landed at **0.60 — inside the
±0.10 window** — with standing 9/12, min lifts 6, displacement 2.83 mm,
against the cancellation model's 11/12, 5, 2.19. **True shunting is
affordable.**

Then the two results that had to be re-run against a mechanism that
actually divides. **Coordination survives**: graded release under true
shunting gives a tripod index of **0.495 median, p = 0.026** against the
wave benchmark of 0.302, on 12 of 12 runs — marginally stronger than the
0.485 it scored under the shortcut. Without graded release, shunting
gives −0.013: uncoordinated. It is also **MDN-specific** — the same
mechanism driven by DNg100 sits *at* the wave value (0.300, p = 0.79),
not above it. **MDN direction does not survive, and the re-run says why
it was the wrong question**: the sign flips with the synapse model
(−0.079 without graded release, +0.053 with) and every arm translates
under 0.25 mm in six seconds. That is not a direction measurement.

And the cost, which is the sharpest thing here. A local re-search from
the mapped point — 47 evaluations, exactly what the mapping document
budgets — **never once produced a run in which all six legs stepped**:
min lifts was 0 in 112 of 141 runs. Running three known-good recipes with
and without the mechanism, on the same seeds, min lifts is 4 to 21
without it and **0 in every single run with it**, while cadence stays in
band at 4–6 Hz. The fly still steps rhythmically. It has stopped lifting
at least one leg. A factorial says conductance synapses alone are free
(min lifts 13, 4, 7) and shunting is where legs start dying — and the
failure is **lateralised**: left legs 13/7/2 while right legs hold at
23/21/21.

F30, the first search ever run on the physics we believe — conductance
synapses, true shunting and graded release on every one of 220
candidates, balance free around the mapped value — **scored zero**.

**Interpretation.** The champion is tuned to a simplified synapse and
does not survive being made more realistic. That is what you would expect
if its parameters had been quietly compensating for the missing physics
all along, and it is the honest reading: we did not lose a good fly, we
found out what the good fly was standing on. The mapping idea itself came
out better than it went in — the analytic term has the right sign and the
wrong size, and the invariant worth carrying across a model change is the
network's firing rate, not any parameter that looks like the one you
changed.

**Problems / what bit us.** The 4x constant is the second claim this
project has published and withdrawn within a day, and both times the
catch was the same heuristic rather than a code review. Worth stating
plainly: **a dial that moves the wrong way is a bug, and it is the only
alarm that fires before the result looks plausible.** Separately, a
mechanism that looked like a champion-maker — short-term synaptic
depression, implemented since the beginning and never wired into the
drive path at all — was reported at 7/20 → 18/20 standing and **did not
replicate**: on 20 fresh paired seeds, 16/20 against 15/20, McNemar
p = 0.75, and monotonically harmful above that (4/20 at U 0.05, 0/20 at
0.08). Zero qualifying runs in all 100. Refuted as registered.

**What changed in the model.** `cns.py` decays both conductance pools.
`map_theta.py` carries a measured balance factor of 0.48 with the
analytic 0.32 recorded as a lower bound. `walk_search.py` gains
`bal_target`, `w_syn_scale` and `std_U`, each in the network cache key.
The champion is unchanged and the probe is bit-identical throughout.

**Attribution.** Robin asked for the most nature-accurate outcome first
and approved the graded-synapse set that makes the coordination arm
possible. The agent implemented shunting, shipped the decay bug in it,
caught it by its own wrong-way rule, withdrew the constant it had
published that morning, and re-ran both load-bearing results against the
repaired mechanism rather than leaving them resting on the shortcut.

**Video.** walker_mc1_first.mp4 — not this arc's fly, but the day's only
new footage: male #2's first behaviour, and the first render in this
project's history whose parity was checked by the machine rather than by
a human reading a printed number.
