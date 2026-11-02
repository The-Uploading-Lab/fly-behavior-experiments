---
id: 100
date: 2026-09-15
title: The body's derivative survives a tick and not a footfall
tldr: The lab started fitting the network by gradient descent, which needs a derivative through the physical fly, and lane A spent the afternoon finding out there is not one. The champion's feet make and break contact 5,362 times a second, the linearised body agrees with it to about four parts in ten thousand across one 2.5 ms tick and is wrong by a median factor of 25 over 250 of them, and only 45 of 2,400 ticks hold their contact set at all. A cheap gate picks the ticks where the linearisation is sound, 163 of 240 on the champion's own recorded drive, and the body term was licensed on that number as a search direction only, never as a measurement. Six of the runs on the way measured a body the champion does not run, which lane A caught itself and the judge recorded as its own miss. The champion is unchanged. Robin stopped every seat at 20:05.
era: standard
commit: the commit this entry lands in, on main; the judge's verdicts at 5ee5711bc (the seal), 5cb339bc6 (the body branch), 1d1eafdd7 (the linearised body) and 0b7926500 (the champion's body and the gate); lane A's packets at 651e214ce, 9e631a53f, dc27b187a and c54700fbb on lane-a
theta: the champion wm1 unchanged, results-wm1-theta.json sha256_16 3a21ddb794be23cb. No parameter changed. The champion's effective body is abdomen.make_fly, selected by its own abd_gain 0.2
seeds: no engine seed and no Metal slot all afternoon. The one trajectory used is the cord spike tape already recorded on seed 203892, replayed through the body with the network switched off
n: 240 of the recording's 2,400 body ticks for the decision number, every tenth; 120 ticks for the agreement read on the CPG walker; 6 s of walk for each contact census; 42 and 58 body inputs differenced per tick
wall: lane A's own session, about 17:52 to 19:55 EEST, on the CPU with no GPU and no network run; the judge's rulings at 18:01, 19:14, 19:28 and 19:59
reproduce: lanes/A/gp2f_mjx_body_probe.py, gp2i_transitionfd_probe.py, gp2j_contact_eps_probe.py, gp2k_floor_entry_probe.py, gp2l_contact_census.py, gp2m_walking_tick_agreement.py, gp2n_cheap_gate_probe.py, gp2o_tape_replay.py, gp2p_champion_body.py and gp2q_champion_drive_gate.py, over the receipts of the same names under lanes/A/2026-09-15-gp2*.json; the seal at lanes/judge/2026-09-15-held-out-seal.md, the rulings at lanes/judge/2026-09-15-gradient-programme.md
media_note: No clip. Nothing here ran the fly. The whole afternoon is derivatives of a body taken on the CPU with the network replaced by a tape, and the one trajectory involved is a walk already published under entry 96's lineage. Under the footage rule a cost and a closed route carry their reader and their commands instead of a picture.
---

20:1x EEST. **The programme.** At 14:32 lane A's goal was rewritten from the haltere work to
something the lab has not tried before: fit the network's free numbers by gradient descent.
Not a sweep over a handful of shared scalars, which is every search this project has run, but
a twin of the walking network that carries a derivative, a loss built from the card rows, and
a ladder of parameter sharing from super_class down through hemilineage, cell type, motor
neuron and the special cells, with ten neuron-model numbers per unit and the measured ones
held fixed. Every minimum it finds is a nominee, graded by the bar like any other body. The
free count of record at the end of the day is **49,435**, on a ladder of **147, 738, 43,860,
4,690** at the four levels.

That design has a hole in the middle of it. The loss is read off the fly's behaviour, and the
fly's behaviour comes out of a physics simulator. To push a gradient from a card row back to a
synaptic parameter, something has to carry it through the body.

**First the seal, because it was drawn before any of this.** At 14:55, before lane A's loss
proposal existed, the judge fixed how rows would be withheld from the fit: the complement of
whatever the lane proposed, plus one withdrawn row per registry group holding three or more,
chosen by the smallest sha256 of the row ids against the noon map's hash, never by a hand. At
15:19 the draw ran and was committed, seal sha256 `646d5edc1d662b5a`. **Held out:
flyscore.lr_asymmetry, gait.cadence, gait.swing_ms, posture.body_height, rom.front_ctr_deg,
rom.front_fti_deg.** The loss keeps 20 rows carrying 17 distinct facts about the animal, drawn
from a readable universe of 26 rows and 23 facts; the other 31 walking-side rows are readable
by no instrument today and are neither fitted nor sealed.

One ruling inside the seal matters more than the draw. Lane A's loss divided each row's hinge
distance by its band width, and on the champion's card that puts **81 per cent of the loss on
one fact**, left-right leg phase, whose bands are 1.6 to 2.2 degrees wide. A descent under that
loss is a phase optimiser wearing a card. The normaliser is now the row's own measured
between-block median spread, the resolution of the instrument that reads it, and every receipt
prints each term's share so a lopsided loss is visible rather than suspected.

**Then the body, and the first answer was no.** MJX is MuJoCo's differentiable path, and at
17:52 lane A probed whether it would take the title body at all. It refuses it, on contact
sensor semantics, on plane-mesh margin and gap, and on body-transmission actuators. It accepts
only a body stripped of its adhesion actuators, which is a different fly: it drifts **2.514
length units from the title body in 250 ms** and settles on five of seven contacts. On that
reduced body reverse-mode differentiation raises inside the solver's loop, and the one setting
that admits it, a single solver iteration, throws the fly off the ground entirely (zero
contacts, NaN). Forward mode does return a finite gradient, **42 of 42 inputs within 0.9 to 2.6
per cent of central differences**, at 1.00 s per window against 0.046 s for the plain rollout,
one pass per input.

The judge closed MJX as the twin's body on this version and resolved the design's body branch to
the black box: the gradient reaches the motor pools and stops there, the loss stays the selection
criterion rather than the fitted quantity, and **no FITTED value may come from a motor-pool-only
target**. One route stayed open, lane A's own: linearise the body locally, step by step, and
chain the small matrices.

**The linearised body is exact, right up until a foot moves.** At 19:05 lane A measured it
against central differences of the whole window. Between contact changes the chain reproduces
the body's own derivative to a **median relative difference of 7.19e-6 on one tick and 7.16e-6
over 250 of them** with the contact set held, and the adjoint reproduces the chain's row to
7.4e-14, which is arithmetic noise. Across a contact change it carries nothing: 5.5e-3 on one
tick, **38.2 over 25 ms** with five to seven changes inside it. No differencing width from 1e-6
to 1e-2 brings the moving window under 1.00; the best is exactly 1.00 at 1e-5, where the held
window reads 2.05e-7.

So everything turns on how often the feet hold still. Lane A counted, on flygym's own CPG
walker: **39,110 changes of the active contact set in 6 s, 6,518 a second, median constant
stretch 0.1 ms**, which is one engine step, p90 0.3 ms, longest 22.4 ms. Ten of 2,400 body
ticks hold their contacts and **none of the 48 loss windows do**. The judge had written the
decision rule before the number arrived, at 19:14: a median constant run of at least 25 ms
builds a contact-keyed piecewise scheme, under 10 ms closes it. At 0.1 ms it closed.

**What replaced it is a gate, and gates are cheap.** Rather than ask where the contacts are
constant, ask directly whether the linearisation agrees with the body at this tick: one extra
central difference along three random directions, **0.0088 s per tick, 2.9 per cent of the cost
of the chain itself**. On 120 ticks of the CPG walk it rank-correlates 0.943 with the full
reference and at threshold keeps 57 usable ticks and **zero broken ones**, dropping 63 broken
and no usable. No threshold pair tried keeps a broken tick. The directions have to avoid the six
adhesion commands, which sit pinned at their upper bound where the derivative exists on one side
only: a centred difference there returns exactly half, ratio 2.0 on all 32 affected entries.

**Then lane A found it had been measuring the wrong fly.** The champion's theta carries
`abd_gain` 0.2, and the runner selects `abdomen.make_fly` on that key: 83 positions, 82
velocities, 58 actuators, adhesion gain 15.0407, four contacts at the warmed state. Every run
from GP2i to GP2n had built `make_locomotion_fly` at gain 40.0, which is 73, 72, 48 and seven
contacts. **Six runs measured a body the champion does not run.** Lane A caught it itself, by
reading `abd_gain` against the branch that consumes it, and the judge recorded its own miss
beside it: it had accepted the body's identity as stated in three packets without once asking
which builder the champion's theta selects.

Re-read on the right body, the qualitative findings survive and the numbers move. The chain is
exact where contact holds, **4.36e-4 over a tick window in which the contact set moved twice**,
the adjoint reproduces it to 5.29e-15 over 58 entries, and nothing survives 250 steps (median
25.11, worst 13,283). The costs roughly double: 0.338 s a tick for the chain, 0.330 s for the
adjoint, **16.9 s per 50-tick loss window** against 0.054 s for a plain rollout.

**The decision number, on the champion's own walk.** Getting it looked like it needed an engine
slot, which the fleet did not have. It did not: lane A replaced `cns.CNS.run` with a reader of
the cord spike tape already on disk from seed 203892 and ran the body verbatim inside the drive
callback. **All nine fields of the resulting row equal the row the engine itself produced for
that seed** (speed 1.858867313611122 to sixteen digits; lifts per foot 21, 9, 37, 45, 39, 25;
final tilt 14.56 degrees), from 2,400 callbacks and 24,884 spikes, in 18.5 s of CPU. The lane
withdrew its own claim that a slot was needed.

On that trajectory, every tenth tick: **163 of 240 ticks kept and usable, 0.679**, and 0.682
once the first 500 ms of transient is dropped. **Zero kept-and-broken at every threshold pair
read**, seven usable ticks dropped, rank correlation 0.901 with the full 58-column reference.
The champion's own drive is markedly easier on the linearisation than the CPG walker's, 70.8
per cent of ticks inside tolerance against 47.5. Its contact census is the same picture as the
walker's: **32,171 changes in 6 s, 5,362 a second, median stretch 0.1 ms, longest 37.7 ms, 45
of 2,400 ticks holding their set**, so the piecewise route is closed on the champion's walk as
well as on the walker's.

The rule written before the read asked for one third of ticks to build the gated term and would
have closed it under one fifth. It came in at **twice the bar**, and the gated tick body is
built.

**What it is allowed to do, which is less than it sounds.** The per-tick Jacobian is the object
and the body term is truncated at tick boundaries, so no gradient crosses a tick edge. The
dropped ticks are the contact events that gait is made of, which means the instrument is
systematically blind to exactly the moments that produce stepping. Therefore nothing read
through the gated body enters as a match, as a band reading, or as a FITTED value: the loss is
read on the real engine, and every minimum the descent finds is a nominee that has to win a
title match against the folded control like any other body. It is a direction to walk in, not a
number to believe.

**Problems, in the order they bit.** The judge's own seat went 46 minutes without a wake between
18:13 and 19:01 and five packets sat unread; the five-minute watcher is now a standing measure
rather than a habit. The free-parameter count published at 19:14 was 49,433 and is 49,435: a
level's unit counts a parameter free when any of its rows runs it free, and the slow motor
neuron is the only row of its type, so its two entries add at two levels rather than one. The
same cell is the remaining honest gap in the pinning work: the fast Azevedo cell's measured
resting potential and input resistance, -68 mV and 150 MOhm, are pinned, while the slow cell's
measured -48 mV and 700 MOhm are not, because the champion runs it at -64 mV and 204 MOhm. Those
two numbers are carried as gauges at weight zero, read but not fitted and not yet reconciled.

**What changed in the model.** Nothing. No parameter of the champion moved, no card row was
re-read, and wm1 holds. What changed is the shape of the machine being built to move it, and
what the record now says cannot be done: MuJoCo's differentiable path will not take this body,
a whole-trajectory gradient through it does not exist at any differencing width, and a
contact-keyed scheme is dead on two independent censuses. What is left is a gradient that is
trustworthy for 2.5 ms at a time, on two ticks in three, and is only ever allowed to point.

At 20:05 Robin stopped the judge and every other seat until further notice. Lane B's next
experiment sits committed, registered and unrun.

**Attribution.** The black box with its target at the motor pools, which is where the body
branch landed, was Robin's own second alternative when the programme was put to him; the
premise under the whole thing is his measured-structure ruling of 2026-09-01, "why would it not
be one of the parameters that we search for, its just a number per neuron". The seal exists
because of his 11:4x instruction the same morning, "lift what you need to make this search
happen": the crowning rule's physiology requirement was lifted in part, and held-out rows sealed
before the first gradient step plus FITTED provenance were what replaced it. He ended the day
with "stop judge work for a while and stop the other agents as well until I tell otherwise"
(20:05). Lane A designed and ran every
probe in this entry, wrote the decision numbers before they were needed, withdrew its own
"49,431 passes per window" framing and its own claim that the champion's trajectory required an
engine slot, and caught the wrong-body error that invalidated six of its own runs. The judge
sealed the held-out rows before the loss existed, replaced the band-width normaliser, closed MJX
and the piecewise scheme, wrote the decision rules for the gate ahead of each read, and recorded
its own failure to ask which body the champion's theta selects.
