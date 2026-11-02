---
id: 70
date: 2026-09-07
title: Standing is an execution-path claim
tldr: A published Lane B replay labelled a whole-cord prototype standing while its body ended inverted. The experiment and renderer had run different graphs, and the renderer converted a missing tilt to zero. The clip and its label are withdrawn. A second reader defect changed the body-frame verdict from hind-only to both middle and hind, but the fixed-heading result remains unreadable. No self-righting benchmark exists, and no model or title changed.
era: standard
commit: the commit this entry lands in, on main
seeds: Lane B 81426-81441 paired, with seed 81441 replayed on the wrong graph; Lane D 61535-61536 paired; Lane E 108532-108533 paired; deterministic Metal engine, one worker
video_receipt: lanes/D/2026-09-07-wi1-cxhp8-exact-fall-video-evidence-r42.json, publication-receipts/walker_r548_middle_pitch_authority_108532.sha256-6e5524c4098f4f8ec6698a29c489b7de45a10914e2762934155b3a14fceec379.publication.json, publication-receipts/walker_b_wi1_cord05_core_seed81441.sha256-cc40dc39c4d361f0e172f177c7b4ea80998a9832c65e42b05ecaafb977a166d8.publication.json, lanes/D/2026-09-07-wi1-lh-claw-stability-video-evidence-r45.json, lanes/E/2026-09-07-middle-nonroll-authority-r549-clip-receipt.json, publication-receipts/2026-09-07-wi1-early-claw-stability-rescue-seed61529-r46.sha256-f65748688173c20aa86d199aee236716c8c70ecc773b03b11896a1789cd412a5.publication.json, publication-receipts/c-n148-leftward-t4t5-body-seed50614.sha256-8f5a41ee0384ae82ccd36bff804487d93333b3f3cddf57c2da1ae4ebe8944cf2.publication.json, publication-receipts/c-n148-rightward-t4t5-body-seed50614.sha256-57628a40b3adb7935219a697220133ada6e1d873ee996f1fade1d0da66c5f1d7.publication.json, publication-receipts/c-n149-leftward-t4t5-body-seed50615.sha256-9100f15455e9d15b2b7c5238d64c98b38dea12042720e76bb69b65eb4fe8a0de.publication.json, publication-receipts/c-n149-rightward-t4t5-body-seed50615.sha256-8b53d517b89822518102a174ca0568064e8cfad436d36b90b04e7999e1187e27.publication.json, publication-receipts/c-n149-leftward-t4t5-body-seed50616.sha256-d0a5cad79381f04eb44e0d774e248f44e92642e7c703d9d93d5fded73229c664.publication.json, publication-receipts/c-n149-rightward-t4t5-body-seed50616.sha256-99d926c83fa19f4a7ae0436e450da5ef004b3841135b5a7abefca21038db12d9.publication.json, publication-receipts/c-n150-integrated-leftward-t4t5-body-seed50614.sha256-1aeb9e101463db4912203fb408bb0b0173f77f69f547d49ed979db4ba384b1b5.publication.json, publication-receipts/c-n150-integrated-rightward-t4t5-body-seed50614.sha256-d48ffaaa5776baea18dcbec0be361064c16382a6be61cb9971a09dab2931e1dc.publication.json
budget_exception: Fourth entry because an exact published clip carried a false standing label after the day's three entries had landed; this withdraws a claim and cannot be merged backward except by the forward link added to entry 68.
---

19:20 EEST. **Question.** Does the current fly recover like a biological fly
after inversion, and does published footage verify its standing result?

**Predicted.** No self-righting prediction had been registered. Lane B's replay
was expected to show the standing seed selected from its paired whole-cord
result. Exact CxHP8 input was separately required to rescue at least one wi1
fall, open none and stand on both candidate runs.

**Method.** The public Lane B bytes were hashed and frames at 1, 8 and 15
seconds were inspected. The render battery and code paths were compared with
the paired runner. The paired phase values were recalculated on one
paired-standing seed intersection. Lane D tested exact CxHP8 on two paired
seeds. Lane E removed middle ThC-pitch authority on two paired seeds.

**Result.** Lane B's body ended inverted. The render battery reads 0.499 mm
endpoint height, 0.0188 upright score and total score zero. Its displayed zero-
degree tilt was an absent value replaced with zero. The paired result used the
count-filtered VNC-plus-descending graph; the renderer used the full CNS. The
clip is withdrawn.

The paired-reader correction changed the body-frame label. On eleven common
standing seeds, middle phase moved from 9 to 80 degrees and hind from 10 to 90,
with every paired change toward 180. The fixed-heading read had only one middle
and three hind pairs, so neither row closed.

Exact CxHP8 input opened one fall to 172.17 degrees and rescued none. Lane E's
pitch deletion suppressed the command by more than 99.99% and both bodies
stood, but only one of two fixed-heading middle rows was readable. Both routes
stop. Their verified clips are attached above.

**Interpretation.** MuJoCo supplies gravity and contact, but neither its physical
match nor a recovery response has been validated against a real fly. The
present model has no demonstrated self-righting behavior. Observed spontaneous
falls finish near inversion during the simulated window.

**Problems / what bit us.** Result identity did not include graph identity in
the footage gate. Missing pose data was treated as a numerical zero. Arm
medians used different standing populations despite paired deltas.

**What changed in the model.** Nothing. The exact CxHP8 selector is corrected
and remains default off. Its full-CNS scope-off probe is bit-identical. The next
instrument check replays seed 81441 through both graphs with one capture and one
standing reader; missing pose is an error.

**Attribution.** Lane B produced the result and replay. Lane D produced the
CxHP8 refutation and source audits. Lane E produced the pitch-deletion bound.
The judge found the execution-path and reader defects, recalculated the paired
result, and withdrew the clip.

20:32 EEST amendment. Lane B replayed seed 81441 through one capture schema and
one standing reader. The count-10 VNC-plus-descending core stands at 10.600
degrees with 33,004 cells and 243,894 edges. Keeping every among-core edge
stands at 10.589 degrees. Adding every one of the 31,803 outside cells that
synapses directly onto the core stands at 11.888 degrees. The full CNS ends at
172.334 degrees with 188,508 cells and 13,620,865 edges. The first body-state
difference is at 10 ms. The count threshold and direct outside inputs do not
explain the inversion; some omitted indirect brain or optic-lobe population is
required on this seed. This strengthens the prior instrument constraint: the
reduced graph is not equivalent to the whole CNS. Its new standing clip is
attached only as a diagnostic comparison and does not replace the withdrawn
full-CNS evidence.

The canonical renderer now rejects missing or non-finite tilt, a non-Boolean
standing label, wrong theta or source hashes, a source result without the exact
full-CNS row counts, direct provenance outside the judge-owned title path, and
replay disagreement on displacement, tilt or standing. A live render of the
current wi1 title row on seed 30448 passed at 14.6871 mm displacement, 13.1191
degrees final tilt and 2,400 captured ticks. The withdrawn Lane B battery fails
before simulation.

Lane D supplies a separate stability bound. Applying phasic 1.0 to the exact 31
left-hind claw receiver rows rescued all three paired wi1 falls and stood on all
four spent seeds. Every candidate footfall separation was too wide, 1.1181 to
1.1278 body lengths against the animal ceiling of 1.11. The attached clip shows
the upright candidate; it is evidence for a stability and foot-placement
tradeoff, not for improvement. Lane E's exploratory R549 clip and R550
matched-pose calculation close middle-body composition: a 158-degree roll
component contributes only 0.0621 mm to the foot, one tenth of the synchronous
non-roll contribution.

20:55 EEST amendment. Robin redirected the fleet to test whether walking is
underconstrained by the measurements themselves. The original flyscore has
seven pooled scalars, while the v3.11 title has 26 dynamic rows and one static
row. Those counts do not establish independent coverage of six legs, 42
actuated coordinates, all 15 leg-pair relations or within-leg event order.
The next pass builds that coverage map and admits a candidate metric only if
the same code reads real and simulated trajectories, disjoint real-fly halves
agree and a registered scrambling manipulation moves it. Biological parameter
and single-neuron changes follow the map and are ranked by the measured
behavior each can affect.

Two no-gain circuit closures landed before that redirect. Lane A found zero of
three active reciprocal loops that send their crossed phase to opposite
homologous motor pools; the strongest loop has zero of 12 homotypic crossed
directions. Lane C found that LPT22 passes a static projection but its identical
full-CNS LIVE condition already fails at DNa02, with 372 and 402 spikes against
the 1,000-spike floor and no required reversal. No new seed was spent.

21:14 EEST amendment. Lane D separated the stability effect from R45's late
foot-placement cost. Exact wi1 falls permanently past 30 degrees at 810.0,
2187.5 and 4852.5 ms on the three burned falling seeds. Before replay, R46
therefore froze the common interval to `[0,810 ms)`. Applying the same 31-row
left-hind claw change only in that interval stood on all four burned seeds,
rescued all three wi1 falls, opened none and placed all four footfall
separations at 1.0635 to 1.0978 body lengths, inside the 1.02 to 1.11 animal
band. Exact wi1 reproduced R37 on every row. The attached R46 clip verifies
seed 61529 at 10.935 degrees and 2,400 capture ticks. This is a BUILD chase
signal; only fresh seeds 61537-61538 are licensed, and the title is unchanged.

The last two circuit reads also close without a seed. Lane A tested 101 exact
repeated excitatory types as two-hop bridges across the reciprocal-loop
family. Only one same-cell path exists in one of 12 directions, with capacity
1 against the frozen floor of 20. Lane E read 959 structurally fixed
direct-premotor routes on spent wi1 seeds. Hind has two corrected alternating
carriers; front and middle have none, so no three-segment carrier set exists.

21:24 EEST amendment. The frozen fresh gate rejected R46. On seeds 61537 and
61538, exact wi1 stood twice and the candidate stood twice. Candidate footfall
separations remained in band at 1.0788 and 1.0798 body lengths, and no standing
reference was opened, but there was no paired fall to rescue. The registered
gate required at least one rescue, so zero is a miss and the candidate stops.
No cutoff or dose is reopened. The burned-seed clip remains intermediate
evidence for a local effect, not evidence that the fly improved.

21:40 EEST amendment. Lane A's final named coordinator also closes. The two T1
IN07B006 cells both infer right-side local input under the frozen topology, only
one passes the crossed-output screen, and their active pair is 25.568 degrees
apart against the registered 120-degree carrier floor. No fresh seed was spent.

Lane C supplies one distinct prototype conflict. On already-spent seed 50614,
the exact rendered leftward and rightward tapes drove standing wi1 bodies to
-40.1472 and +17.7674 degrees net yaw. Both signs oppose optomotor following.
The same captures supplied the reader and attached clips, but the execution
branch carried shared CNS and walk sources that differ from integrated main.
The result remains a chase signal. Fresh confirmation waits for the exact pair
to replay on integrated main with source, tape-direction and yaw-sign identity
in one receipt.

21:54 EEST amendment. Lane A completed the class-wide version of its carrier
search before the metric redirect reached it. Thirty-six of 1,042 exact
type/segment coordinates have bilateral crossed output into all three local leg
circuits. Zero has at least 120 degrees of readable left-right phase on the
unchanged spent wi1 run; the maximum is 79.517 degrees. Cell-type localization
therefore closes rather than proceeding to a new cord mechanism.

Lane E traced the two remaining hind premotor carriers through their directly
contacted motor neurons on twelve spent replays. IN21A009 falls from 136.2
degrees at the premotor population to 10.7 degrees at those motor neurons and
27.1 degrees at aggregate coxa-roll output. IN21A022 reaches active targets,
but their 98.5-degree phase is not consistent across seeds. This locates one
phase loss and leaves the other interval unresolved; neither changes the body.

22:01 EEST amendment. C-N149 repeated the C-N148 body response on fresh seeds
50615 and 50616. All four bodies stood. Leftward pixels produced -31.0199 and
-46.4153 degrees net yaw; rightward pixels produced +108.5812 and +2.9450
degrees. Every arm opposed optomotor following, though the last value supports
only a sign claim. The four same-run clips are attached.

The fresh repetition confirms the response on Lane C's frozen build, not on
integrated main's exact source. The execution branch differs from integrated
commit `2617541c1` by 381 insertions and 4 deletions in `cns.py`, plus 611
insertions and 20 deletions in `walk_search.py`. The requested registry
promotion is held. A spent-seed replay after merging main is required before a
same-tape crossed-direction control or any sign repair.

22:11 EEST amendment. Two final stale in-flight routes close. Lane A applied
phasic release to wi1's 24 graded premotor cells on seed 203759. The control
stood; the candidate ended inverted at 172.516 degrees. Only the middle
fixed-heading pair remained readable in both arms and moved 16.07 degrees
toward 180, while front and hind moved away. The all-three-pair gate fails.

Lane D compared 24 per-leg hook, club and load mean rates from three exact wi1
falls with 591 overlapping windows from three standing controls. Every fall
range overlaps standing. The closest channel, left-front hook-flexion, still
overlaps by 6.7422 Hz. These current mean rates cannot select a fall-specific
gain or interval. Lane E's R553 source split has an unadjudicated local
spent-seed artifact and must not rerun; further work waits behind the walking
instrument audit.

22:49 EEST amendment. R553 did not move the model, and its overall motor-output
interval remains unresolved. The packet is held because
`accessory_tibia_flexor_A_out_of_bundle` is labeled nonalternating at nominal
p = 0.0438 while its adjusted q = 0.0656 fails the declared q <= 0.05 gate.
Only `accessory_tibia_flexor_A` clears the adjusted grade. The 13 focused tests
pass but do not test this side of the classifier. The correction uses the
committed per-seed rows; no CNS replay or fresh seed is licensed. Walking
coverage continues.

23:25 EEST amendment. The fly did not change. Lane D reordered all six real-fly
leg trajectories while preserving every leg's samples and cadence. All 15
pairwise phases moved, but only four of 11 computable score rows responded: the
pooled tripod score and three left-right phase rows. The pooled score changed
5.64 times more than its noise and still passed in both arms, so its value
notices wrong order while its current band does not reject it.

Lane B's proposed anterior-extreme timing measure is held. Its committed
scramble reapplied the walking gate and silently removed 386 of 917 sequences.
A fixed-membership diagnostic reproduced the same 0.106790-cycle effect, but
that does not replace the committed evidence path or establish that fifteen
pair readings reduce to the proposed two independent facts. The complete
walking-coverage map remains open.

23:45 EEST amendment. The fly still did not change, but one measured conflict
became exact on integrated main. C-N150 replays the rendered T4/T5 tape on spent
seed 50614: a standing body turns -40.1472 degrees under leftward pixels and a
standing body turns +17.7674 degrees under rightward pixels. Both signs oppose
optomotor following. Crossing only `vision_dir` while holding the tape fixed is
identical in the CNS trace, quaternion yaw, body capture and four other saved
fields. `vision.optomotor_sign` is therefore CONFLICT, FEMALE-ONLY. The tape
causes the response, but this test does not locate the downstream sign inversion.

The coverage work also became executable. Lane A's complete matrix finds no
direct real-fly identity for any of the 42 simulated joint axes; six have named
2D projected-knee proxies and 36 are unobservable. Lane D's fixed real order
grades CONSISTENT and the destroyed order grades CONFLICT, while the pooled
tripod row passes both. Lane E's one-path within-leg sequence reader passes the
fixed real split on 11 of 12 rows and its manipulation on 12 of 12. Replacing
all 18 non-tip landmarks leaves the seven current flyscore values exactly
unchanged. One spent wi1 trajectory yields only 2 of 12 readable rows. These are
measurement instruments, not evidence that wi1 improved, so no title row moves.
