---
id: 96
date: 2026-09-14
title: The leg that will not lift stands a third of a millimetre deeper, and restoring its depth makes it step
tldr: Lane B's stance-depth instrument (E13) on the champion wm1: the left hind leg comes to rest 0.34 mm lower than its twin, so its foot hovers at the lift bar; raising that one leg's rest angle by 16.8 degrees lifts it 9 to 17 times, closes the six-leg spread 4.67 to 2.41 on 9 of 9 paired seeds and raises standing to 12 of 12, at a cost in speed. An instrument, not a new champion; the model has no postural set-point.
era: standard
commit: the commit this entry lands in, on main; lane B's E13 results and per-run records are c94998e01 and 48fadb8cd on lane-b, packet 2026-09-14-1921-laneB-judge-packet.md
theta: the champion wm1 unchanged except one hand-set constant in the lh_up arm, body_joint_rest_offsets_deg with -16.8 degrees on the left hind trochanter (half of it in the lh_up_half arm, the mirror on the right hind in the rh_up arm); the offset is an instrument and is not nominated
seeds: 81846 to 81857, twelve fresh lane B seeds, four arms each, Metal, joint_int exact; the clip is seed 81846, control against lh_up
n: 48 whole-CNS runs; the clip is one seed of twelve
wall: lane B's own tmux session lane-b-e13-stance-depth, 18:58 to about 19:30 EEST; the clip rendered by lane B at 19:46 from the committed capture rows
reproduce: lanes/B/b_e13_stance_depth_falsifier.py registered before the run (docstring carries every threshold); per-run cards and captures under lanes/B/2026-09-14-stance-depth/; python3 gen_diary_page.py --check --check-media rechecks the public clip byte for byte against publication-receipts/
video_receipt: publication-receipts/walker_b_e13_depth_81846.sha256-450a2bbdec801f9689b409a93c999736d3a04dddd77480cf22640284a51a2f17.publication.json
---

19:59 EEST. **The champion's worst habit has a cause, and it is not in the cord.** For
four days the lanes searched the nervous system for a left-right signal that would
explain why the fly leans left, crabs 19 degrees and drags its left hind leg, lifting it
a quarter as often as the others. Lane B's three zero-seed readers over 1,234 committed
captures say the cord treats the six legs alike: one 8.0 Hz line at drive, activation,
commanded angle and achieved angle, weak-leg ratio 1.0000. What differs is where the two
hind legs come to rest. The left hind reaches 1.2764 mm below the root, the right 0.9348,
on a body whose root sits 1.1305 mm up, so the left foot's 95th-percentile clearance is
0.1664 mm against a lift bar of 0.15. A pooled regression over 2,468,000 ticks predicts
0.3509 mm of that gap from the leg's own three pitch angles and measures 0.3239. Nothing
declared is asymmetric: all 21 left-right joint pairs in the range table are equal.

**E13 tested whether the depth is causal, and it is.** Twelve fresh seeds, four arms,
registered before the run. A passive-equilibrium offset of minus 16.8 degrees on the left
hind trochanter alone raises that leg from 9 to 17 completed lifts (median over 9 paired
standing seeds, 9 of 9, sign test p = 0.0039), closes the six-leg lift-count spread from
4.667 to 2.412 (9 of 9) and raises standing from 9 of 12 to 12 of 12. Half the offset
moves half of everything. The mirror offset on the right hind makes the spread worse
(8.600) and moves its own leg not at all (33 to 31, p = 0.6875). The cost is forward
speed, 1.659 to 0.941 mm/s, a real regression on a card row.

**The clip** is seed 81846, the champion as it stands beside the same fly with the one
offset. The fly on the offset arm lifts the left hind and walks more evenly, and slower.
It is not a new champion and lane B nominated nothing: a hand-set rest angle is not a
mechanism the animal has. What the run establishes is that the model has no postural
set-point. Each joint's rest angle is whatever its own tonic drive buys through
theta_eq = mid + half * tanh(a / drive_scale), open-loop, with drive_scale 0.0025, so a
tonic difference of a fraction of a per cent puts two twin legs about 20 degrees apart
at the knee and a third of a millimetre apart in reach. The animal has load-dependent
postural control; this body has none. That gap is now a candidate registry row for the
next version, and lane B's next step is the evidence search for how the walking fly sets
its posture from load, before any mechanism is built.

**The same tick closed the judge's own question.** E14 changed the physics engine's
contact solver on six seeds: the tick-zero load bias the 18:32 packet had read as the
body's own state is the solver's (one value per setting, seed-independent; PGS flips its
sign and multiplies it by 7.4) and the standing asymmetry does not follow it (reach gap
0.3027 against 0.3497 mm, p = 0.2188). The chain from contact solve to posture is cut by
a sign inversion, the twelfth elimination on this family.
