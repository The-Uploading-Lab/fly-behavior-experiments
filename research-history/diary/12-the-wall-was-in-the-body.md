---
id: 12
date: 2026-08-18
title: The wall was in the body
tldr: A female recipe stood on 9 of 12 seeds — the highest rate this project has ever measured — and still took no title, because she barely stepped. Chasing why led out of the nervous system entirely: play a certified tripod gait through our own muscle interface and 70.9% of what it asks for is out of range. Then a literature sweep said a fly's leg joint rings near 30 Hz. Ours rings at 8.6-10.5. That is a conflict, and it deletes the corner of the search space every champion we have ever crowned was living in.
era: standard
born: the expressible envelope — a body can be too slow to be asked
commit: df5333d (arc 56353fb..df5333d)
theta: results-p150-theta.json (female champion p150); results-walk-search-m2.json winner (male)
seeds: FTP1 title 866-879; f0 dose 7 doses x 6 seeds; p150@f0-15 title 880-892; paired decider 2012-2027 (identical for both arms)
n: 12 + 42 + 12 + 24 runs, 6 s each, plus two analytic sweeps with no simulation in them
wall: dawn to midday, 2026-08-18
reproduce: .venv/bin/python3 envelope1.py   # analytic; then envelope2.py for the dose
---
**Question.** Every fly we have built either stands or steps, never
both. We had been treating that as a nervous-system problem — the
wrong wiring, the wrong gains, the wrong search. Is it?

**Predicted (pre-registered, quoted).** The envelope entry registered
a two-way split before the dose was run: "**TRADE** outcome = speed
rises with f0 AND standing falls -> the stand/walk trade IS the f0
axis... **BLIND-SPOT** outcome = speed rises and standing HOLDS ->
the searches never exploited high f0 (displacement scoring crowns the
first stable creeper) and a better champion sits unclaimed INSIDE the
existing space."

**Method (jargon defined inline).** Three instruments, in order.

*The certified-tripod probe.* Take flygym's own CPG walking program —
a tripod gait by construction, nothing of ours in it — and play it
through our full execution stack (muscle activation, joint, actuator)
at champion parameters. If the stack is faithful, a tripod goes in and
a tripod comes out.

*The envelope decomposition* (`envelope1.py`, analytic, no simulation).
Our joint is a second-order plant with a natural frequency **f0** —
how fast it can ring — and a damping ratio **zeta** (below 1 it
overshoots and rings; at or above 1 it does not). To make a joint reach
a commanded angle the controller has to invert the plant, which costs
`theta_eq = theta + (acc + 2*zeta*w0*vel)/w0^2`. Both correction terms
scale as 1/f0². A slow joint needs an enormous fictitious equilibrium
angle to produce an ordinary fast movement, and the actuator simply
cannot be sent there — the demand is **clipped**.

*The f0 dose* (`envelope2.py`): move that one number across seven
values, hold everything else, both animals, 6 seeds each, graded by
the same battery gates as a title match.

**Result (numbers, with variance).**

The record-holder first. FTP1's winner, 12 fresh seeds (866-879):
**standing 9/12 — the highest rate ever measured here** (the reigning
champion p150 defends at 6/12), nonfoot 0.000 on every run, R5 9/9.
And **all-feet qualifying runs 1/12 against a bar of 2: FAIL.** Duty
0.93, cadence 1.73 Hz, speed 0.33 mm/s, front-leg ROMs shrunken,
adhesion threshold 2.95 — 2.4x the champion's. She stands superbly by
barely stepping, leaning on her feet sticking to the floor. The
night's triptych: the male steps beautifully and cannot stay up (~33%
standing, invariant across three different search objectives); this
female stays up beautifully and barely steps; p150 is the only recipe
holding both clauses at once.

Then the probe, and it was the same on all 4 seeds because playback is
deterministic (n=1 would have done): **disp 11.80 mm, upright 0.99,
per-leg lifts [0, 2, 0, 0, 0, 1]** — a fast, stable **skater**, with
essentially no swing events at all. Tripod in, skater out. The cause
prints at the top of the log: the inversion reports **70.9% of the
program's flex/extension demand OUT OF RANGE.**

The decomposition says why. At the champion's own f0 of 10.5 Hz, the
certified tripod program demands a median |acceleration term| of
**25.2 deg against a median joint half-range of 14.0 deg** — the
correction *alone* is 1.8x the entire range before the movement itself
is added; 99th percentile ratio 12.8. Out-of-range against f0, for
that program:

    f0 (Hz)      6    10.5     20     30     50    120
    clipped     88%    71%    44%    28%    17%     8%

It needs **f0 ~ 235 Hz** for under 5% clipping. Our own gentler
designed program needs **~42 Hz**. And the search space we had been
using is f0 ∈ [6, 36] — **while every champion ever crowned landed at
the bottom of it: 6.0, 8.6, 10.5.**

Damping is the second lever (the velocity term is 2·zeta·vel/w0), and
a nearly clip-free corner turns out to exist *inside* the space we
were already searching — (f0 15-20, zeta 0.4-0.6) at 2-9% clipped,
against 36% in the champion corner.

The dose, female champion (native f0 10.5, zeta held at 1.09 so the
joint stays overdamped throughout):

    f0        6    10.5     15     20     25     30     36
    stand   3/6     2/6    3/6    0/6    1/6    0/6    1/6
    minlift 2.0     9.5   14.0     --   22.0     --   32.0
    speed  0.30    0.54   0.40     --   0.42     --   1.37
    nonfoot 0.000  0.003  0.005  0.008  0.311  0.656  0.557

The worst leg's step count rises monotonically **2 → 9.5 → 14 → 22 →
32**, a 16x effect far outside the n=6 noise that makes the standing
column unreadable. The male's dose is a trade immediately: standing
3/6 at f0 6 and 0-1/6 everywhere above, nonfoot 0.46 → 0.89. Same
parameter, opposite bill, on the two animals.

Because f0 6/10.5/15 gave 3/6, 2/6, 3/6 standing, the entry read f0 15
as a **free improvement — the BLIND-SPOT outcome** — and launched a
title match on it. It met the bar: 12 fresh seeds (880-892),
**standing 6/12, qualifying 6/6**, duty 0.81 PASS where the incumbent
conflicts at 0.86, and the **first female standing run in project
history to pass R1i** (1/6, a reflex rule she had gone 0/10 on). A
dead tie with the champion on every title criterion.

So a paired decider was registered before it ran — both thetas on the
**same** 12 fresh seeds (2012-2027), challenger must *strictly*
exceed, ties keep the crown with the incumbent so the rule cannot be
gamed by re-rolling. **p150 defends 6-3.**

**Interpretation.** The blind-spot reading was wrong, and it was wrong
for exactly the reason its own paragraph had warned about: it rested
on 3/6-vs-2/6-vs-3/6, which this project had already measured to be
coin flips. With seed luck cancelled at n=12, f0 15 **halves**
standing. The registered dichotomy resolves as **TRADE** for the
female too. What survives is the sharpest speed result she has ever
produced: on identical dice, raising f0 alone multiplies standing
speed **2.5x (0.36 → 0.91 mm/s)** and raises stepping (min lifts 10 →
12). Nothing else in this project's history has moved her speed like
that.

And then the literature closed the argument. A 30-agent sweep, every
number adversarially checked against its source (20 measured-direct, 3
model-derived, 1 refuted): FTi swing excursion ~42.6 deg (Haustein
2024); swing duration 20-45 ms, mean ~31 ms, and **speed-invariant**
(Wosnitza 2013, DeAngelis 2019 — only stance shortens). The structural
point we had missed is that **a swing is a half cycle**, so a 31 ms
swing is kinematically a ~16 Hz sinusoid, not a 5-16 Hz step rate —
and because swing duration does not relax with walking speed, *there
is no slow-walking regime where a slow joint suffices.* A fly creeping
at 3 mm/s still throws its tibia 40 deg in 30 ms. Azevedo 2020's
measured 8.5 ms spike-to-half-max-force, read as a critically damped
second-order step, puts the plant at **f0 = 31.4 Hz**; Zumstein 2004
agrees in order.

**VERDICT: our body model conflicts with measured Drosophila muscle
kinetics, and has since the loop closed.** Per the governing aim that
is a gift — it deletes the entire f0 < 15 region that every search we
have ever run was living in.

**Problems / what bit us.**

1. **A record with no title is still a failure of the fly, not the
   bar.** 9/12 standing is a genuine existence proof that balance is
   solvable in this rig. It is also a fly that does not walk.
2. **We spent weeks attributing to the network a limit that belongs to
   the body.** The female's speed conflict (0.3-1.0 mm/s against a 2.0
   floor) is the same wall seen from the inside: at f0 8.6-10.5 with
   zeta 1.09 the body is *physically incapable* of the ~2700-3800
   deg/s peak joint velocities real stepping demands.
3. **The escape route is the dangerous corner.** The clip-free region
   is underdamped, and our own tier-0 note says a resonant limb can
   manufacture a rhythm with no connectome in it. Recorded before any
   of it was run: **any move to zeta < 1 must carry the
   shuffled-connectome control.**
4. **A provenance correction, tier-0.** AGENTS.md listed "tau_act =
   12.3 ms from Azevedo 2020" among the loop's unfitted *measured*
   anchors. The string "12.3" appears nowhere in that paper: 12.3 =
   8.5/ln 2, their measured half-rise passed through a first-order
   assumption. Unfitted survives; *independently measured* does not,
   and it is one datum with the 31.4 Hz figure rather than two. Worse,
   their 8.5 ms includes conduction and NMJ delay, so real activation
   is faster — our number biases the body **slow**, in the same
   direction as the conflict.

**What changed in the model.** Nothing in the body itself — MUSCLE.md
governs that, and every number above is a measurement rather than a
change. What changed is where the search is allowed to look: design-
side work stopped (the D7 branch fired as registered), the campaign
target became the expressible envelope, and the next chamber was
ordered into the forbidden corner with f0 confined to [15, 36] and
zeta held at or above 1.0 — toward the measurement, and conservatively
so, since 31.4 Hz sits near the top of that window.

**Attribution.** Robin set the standing rule that titles are decided
on the CEO's bar and that bars never move mid-contest, which is what
forced the paired-decider protocol to be written down before it ran
rather than after 6-vs-3 was known. The agent designed and ran the
probe, the two analytic sweeps, the dose and the decider, and wrote
the amendment against its own earlier reading rather than editing it
away.
