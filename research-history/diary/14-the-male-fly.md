---
id: 14
date: 2026-08-18
title: The second fly — a statue, a bug, real steps, and a crawl
tldr: We ran the female champion's exact parameters on a second individual's wiring, a male, and announced perfect transfer of posture. It was a bug — an empty string where a null belonged had deleted every inhibitory synapse, and the statue was rigidity, not balance. Repaired, the male stepped in twelve runs out of twelve, which is the real and humbler transfer result. Then Robin watched the footage and said he was crawling. He was: the male carries a fifth of his ground contact on shins and body where the female never touches anything but her toes.
era: standard
born: posture as a measured quantity — the standing gate could not see it
video: walker_manc_first.mp4, walker_manc_steps.mp4
commit: arc 6637096..5eb5d7e ("MANC first light" -> the amendment -> "MANCWALK3: the real male STEPS" -> 5eb5d7e)
theta: results-p150-theta.json (verbatim, unchanged through the entire saga); results-round-m1.json for the male's own search (winner cut 10, DNg100, phasic 1.26, camp 20, hz 107, f0 6.0)
seeds: 900-905 per arm (first light), 920-925 per arm (after the repair), M1 battery 930-935, nonfoot re-runs female 826-837 and male 936-942 — all fresh, nothing ever searched on the male cord
n: 2 arms x 6 seeds twice, then 6 + 12 + 7 runs, 6 s each
wall: ~14 h, first light to the posture verdict (the bug lived for most of it)
reproduce: .venv/bin/python3 mancwalk3.py   # then regression_walk.py --theta results-round-m1.json --meta data/manc-as-banc-meta.feather --edges data/manc-as-banc-edges.feather --cut 10 --seeds 6
---
**Question.** Everything this project ever simulated ran on one
individual fly's wiring. A second complete fly cord exists — MANC, a
male. If we translate it into our rig's language and run the
champion's exact parameters, does anything transfer? And if he can be
made to move, can he earn a title of his own?

**Predicted (pre-registered, quoted).** "MW1 WALKS: standing >= 2/6
AND >= 1 standing run with all-feet lifts >= 5 -> the walking recipe
TRANSFERS. MW2 STANDS: standing >= 3/6, no stepping -> the posture
program transfers; walking needs male-side search. MW3 NEITHER."
Committed before any seed ran, and left unchanged when the runs were
repeated after the bug. Later the same night, a second and separate
registration, written before the measure existed: the new posture
measure's band is [0, 0.05], and "the prediction is on record before
the batteries run: **the female champion passes, the male winner
conflicts.**"

**Method (jargon defined inline).** A one-file adapter translates MANC
into BANC's schema so the entire rig runs unchanged — every join
count-verified: 23,188 traced neurons, 692,208 connections at our
conserved-core cut, 307 motor neurons covering all 42 joints, the FeCO
sensor subtypes matched by name, MDN present natively, DNg100 found via
the published cross-individual cell matches. One honest asymmetry
carried loudly: MANC annotates only 8 leg load sensors where BANC
annotates 573, so a second arm rescaled the load gain by exactly that
ratio — a declared dataset correction, not a fit. Later, *round M1* =
the male's own search chamber (candidates scored on stand-and-step
displacement across 4 seeds), then a *battery* — the fixed panel of
published-fly constraints — on six seeds the chamber never touched.
Then a **posture audit**: a render that reports, tick by tick, which
body part is touching the floor and what share of ground force it
carries. Then a new battery measure, **nonfoot** = the fraction of
post-transient ticks in which any proximal leg segment (coxa, femur,
tibia) or the body itself touches the ground.

**Result — the part that was wrong.** First light gave **standing 12
of 12, both arms**: the male fly, on a female-fitted recipe he had
never seen, standing more reliably than the female champion stands
herself (her record: 3 of 6). Steps taken: zero, in every run; the
load-channel correction changed nothing, acquitting that asymmetry as
the blocker. We announced that the posture solution is about fly cords,
not this fly's cord.

**⚠️ WITHDRAWN, same day.** The adapter had filled the "verified
transmitter" column with empty text instead of leaving it null. The
simulator, reasonably, prefers verified over predicted whenever the
field is filled — so empty text silently beat every real prediction,
every male neuron fell back to the default excitatory sign, and **the
male cord ran without a single inhibitory synapse.** A brain with no
brakes saturates: every motor neuron pinned near its firing ceiling.
The 12/12 was co-contraction rigidity, not balance. Withdrawn with it:
the collapse at the degree-matched cut, and the density diagnosis as
mechanism. The first video stays, with this note attached — we do not
polish the past.

**How it was caught.** Three dials in a row did nothing: synapse gain
(6x range, ~flat), then the balance rule (8x range, EXACTLY flat —
identical to the decimal). A flat dial is a broken wire. We verified
the dial worked on the female (20x swing), concluded the male truly
could not respond, asked what makes balance powerless — and the answer
is "when there is nothing inhibitory to scale." The human's laugh
("the male looks paralyzed, why?") started the whole chain. With
inhibition restored the male fires at 24-29 motor spikes per tick,
about 5x the female's level, in the tameable range.

**Result — the part that stands.** Repeated on fresh seeds with the
bug fixed: formally **MW3** — standing 1/6 misses the bar, recorded
without spin. Materially, everything changed. **Every one of twelve
runs steps** (worst-leg lifts 2-14; displacements 1.4-4.9 mm) where
the broken male had zero motion of any kind. Seed 925 delivered his
first genuine standing walk: 3.07 mm at upright 0.93, every foot
lifting 9-22 times — parity-exact on video. The corrected transfer
claim is humbler and more interesting than the withdrawn one: **the
stepping machinery engages across individuals — a recipe fitted to one
fly's wiring drives rhythmic, all-legs stepping on another fly's
wiring, twelve times out of twelve.** What does not transfer is the
balance margin: he steps like her early ancestors did, falling as he
goes. The load-sensor asymmetry was tested again and again changed
nothing — acquitted twice.

**Result — his own search, and the crawl.** M1's winner measured
**standing 1/6** — a clear title FAIL. But his one standing run (seed
930, 1.35 mm, upright 0.97, min lifts 3) produced the first battery
card ever issued to a second individual: **5 of 6 joint ranges PASS**,
including hFTi 84.1 deg against Haustein's real-fly mean of 84.1 deg —
exact — duty 0.63 PASS, R5 PASS, and **R1i_tip PASS**, the same-side
swing-avoidance rule the female has never reliably passed. Conflicts
on his card: cadence 2.45, swing 70 ms, speed 0.25, min lifts 3,
contralateral at the boundary. Per-seed, the other five never stood:
931: 0.71 mm, upright 0.95 | 932: 3.57, 0.87 | 933: 0.59, 0.07 | 934:
3.02, 0.10 | 935: 2.15, 0.72. The chamber also voted internally, and
the votes are worth recording: the microscope-matched threshold
(cut >= 30) scored 19 of 27 against cut >= 10's 8, even though a cut-10
candidate won the final rerank; and the artificial command neuron
DNg100 beat the male's native MDN 22 to 27.

Then the audit. Share of ground force, and fraction of ticks touching,
female champion seed 824 versus male M1 seed 930: foot tip **96.9% /
100.0%** vs **76.1% / 88.8%**; rest of the foot 3.1% / 49.6% vs 16.5% /
87.5%; proximal leg **0.0% / 0.0%** vs 2.6% / 20.5%; body **0.0% /
0.2%** vs 4.7% / 23.0%. He stands lower too (z 1.10 vs 1.26). On fresh
seeds the registered prediction came in **confirmed on both arms**.
Female nonfoot **0.000** — every one of her twelve runs at or below
0.004, including the ones where she falls — PASS with total margin.
Male **0.436 median**, spanning 0.32 to 0.52 across all of his:
**CONFLICT**, and worse than the single audit exemplar had suggested.
Per-seed, male: 936: 0.5155 | 937: 0.4359 | 938: 0.4595 | 940: 0.4991 |
941: 0.3159 | 942: 0.3636. And his signature replicated: **R1i on 2 of
3 fresh standing runs at p < 0.05** (0.005, 0.030; the third 0.070) —
battery PASS. Across both his batteries, 3 of 4 standing runs
significant. Across both of hers, **0 of 4**, p-values 0.73 to 0.83.

**Interpretation.** Two things separate these animals now, and they are
different in kind. The R1i split is a disagreement between individuals
that keeps replicating — a first-class result under the two-fly rule,
tagged MALE-ONLY pass / FEMALE-ONLY fail. The posture split is a
conflict with the real world: real flies bear their weight on their
tarsi. He stands, but he does not stand **on his feet**, and he grazes
the floor with body or shin about 44% of the time against her never.
That is the sharpest cross-individual difference we have measured, and
it is a constraint we now have to satisfy rather than a curiosity. The
project's oldest fault line, the stand/walk divide, has also reappeared
at a new scale: across individuals.

**Problems / what bit us.**

1. **An empty string is not a null. A flat dial is a broken wire.** And
   a result that flatters the project ("perfect cross-individual
   transfer!") deserves the most suspicion precisely because it
   flatters.
2. **The body model has no self-collision.** All 55 contact pairs in
   the compiled physics are part-versus-ground; zero are fly-versus-fly.
   Legs passing through the abdomen has been legal physics in every
   video ever rendered here, for both animals. Adding self-collision
   would change the physics under every existing result, so it is a
   registrable decision, not a patch — recorded, not done.
3. **The standing gate is blind to posture.** It checks height and
   uprightness only, so a fly propped on its tibiae at altitude passes
   it. That hole existed for the whole project and was found by a human
   watching a video, not by any test.
4. **A deviation from the registration, recorded plainly**: it promised
   a same-seed re-run, and the battery's own fresh-seed discipline
   silently picked fresh seeds instead. Accepted as the better
   evidence, the overwritten title files restored from git untouched,
   and said out loud rather than tidied away.
5. The renderer had the connectome name hard-coded into the new video
   banner ("BANC female") and the data paths hard-coded too — both
   caught before the first male frame shipped a wrong label. 84 male
   motor neurons carry only generic muscle labels and stay unmapped; 62
   more lacked resolvable targets. These gaps are carried, not hidden.

**What changed in the model.** Nothing in the fly, twice over — that is
the headline: the SAME code and physiology now runs two individuals,
and the repo gains the male core as a first-class substrate. The
instrument gained `render_split --contacts` (parity-gated: 1.35/0.97
and 7.08/0.97 reproduce the records exactly) and the battery gained
**nonfoot**, band [0, 0.05], scored on every run from now on. The rig
change is capture-only and the PROBE re-verified bit-identical
[1.268453802823, 1.268453802823, 0.990248627778, 1.149806329417].

**Attribution.** Human: "another fly connectome exists. WE should
prolly run tests on both LOL"; "hahhaa the male looks paralyzed, why?"
— the question that unravelled everything; "Try again" at the moment of
the corrected render; and, watching the M1 footage, "*the male is
basically crawling... it's some of the leg and joints [on the ground]
and the backlegs are rendered inside the back of its body which should
be impossible*" — both halves of which verified against the compiled
model — plus the call to make it a battery measure ("option 1"). Agent:
the survey, the adapter with its verified joins, the bug and the chase
and the catch and the withdrawal, the corrected experiment, the M1
chamber and battery, the contact instrument, the audit, the
registration and its band, the fresh-seed verdicts.

**Video.** walker_manc_first.mp4 is the withdrawn debut — six seconds
of a male fly standing perfectly still because nothing in his cord
could inhibit anything. walker_manc_steps.mp4 is the same animal after
the repair, taking real steps. Keeping both in the same entry is the
point. The M1 footage that started the posture work is video 17,
walker_m1_r1i.mp4 (seed 930, 1.35 mm, upright 0.97, parity-exact
against the record); it is not in the diary assets, because the source
renders do not live in this repo and a reference with no file behind it
is worse than none.
