---
id: 77
date: 2026-09-09
title: The male fly stands on two reused seeds
tldr: Setting subthreshold adaptation to zero preserved male standing on two reused seeds, but both runs remained too slow to qualify. Four joint-search candidates failed their second-seed comparison. The shared champion is unchanged.
era: standard
commit: the commit this entry lands in, on main
theta: R596/R597 male prototype; shared results-wi1-theta.json unchanged
seeds: male spent 108523 and 108524; joint 81548 and 81547; D 61524; C 50600-50605 and 50614-50618
n: two male lead pairs; four joint candidates; eleven C pairs; two D and three E follow-up pairs
wall: per-arm timings in pinned lane receipts; judge reran readers without simulation
reproduce: lanes/judge/2026-09-09-r596-r597-review.json and lanes/judge/2026-09-09-joint-keeper-review.json
video_receipt: publication-receipts/walker_r596_male_adapt_a_zero_108523.sha256-df3fb91cc045af629967c841f5d8365680f87bbea604e7f6cfde1cf00dd356ef.publication.json, publication-receipts/walker_r597_male_adapt_a_zero_108524.sha256-0a490c1211ffe436b6b36ed010d9416ed9e6be6b06edbd48b665f6fec238f05c.publication.json, publication-receipts/walker_r639_male_hind_failure_archive_r640_108523.sha256-57e922ee9713956f690abe12597a80edfc7c891af5b5488b764c40ca4bed1c39.publication.json, publication-receipts/walker_r643_male_rf_height_failure_r644_108523.sha256-57c579e76f6ca26c25a6aed1fbc58bd5b8307e4eb4fbfc217d538c28ed54c241.publication.json, publication-receipts/walker_r633_male_height_failure_retry_r637_108523.sha256-9a836ad0b489c1f94160bf8ec82087537b763e49f4bd9718f34755c17b677fdf.publication.json, publication-receipts/walker_r645_male_contact_transplant_r647_108523.sha256-068039cb5929c44e6f72d4bb56a3c62f243ba1e2bf45c624b1a223677ec13568.publication.json, publication-receipts/walker_r648_male_motor_reset_failure_r650_108523.sha256-4b06bdb1847ab5cd95cb96acf41634a287c2885ed096b8b3b4246ae1a73380e7.publication.json, publication-receipts/walker_r658_in19b012_gain_r660_108523.sha256-0c51b2a6f5b8d7dd1257d424fe5d8d7a0b2cd6e9f53b0101f7ed8e5495ee2718.publication.json, lanes/E/2026-09-10-r671-in21a010-gain-r672-publication.json, lanes/E/2026-09-10-r679-tactile-r680-publication.json, lanes/E/2026-09-10-r682-ctr-r685-publication.json
---

**Question.** Could the full male connectome retain standing after a fixed
neural assumption changed, and could the first full-vector search candidates
retain their gains on a second development seed?

**Prediction.** Lane E's frozen comparison required upright support with the
accepted sensory and motor adapters unchanged. Lane B's frozen gate required
fewer grouped conflicts on one paired seed, no increase on the other and
retained standing, leg motion, speed, cadence and footfall separation.

**Method.** E set subthreshold adaptation `adapt_a` from 2 to 0, preserving
spike-triggered adaptation and all other recorded neural and body parameters.
Both arms used the full 211,577-neuron, 26,028,386-edge male graph. Its accepted
adapters retain 328 mapped and 53 unmapped motor rows and 663 proprioceptors;
R580-R587's source limits remain. The two saved readers, source hashes and
native vectors were checked independently. Both videos reproduce the candidate
run and have verified public-byte hashes and renderer parity receipts.

B selected four full-dimensional vectors under its frozen rule, then ran each
and wi1 on both spent identities using deterministic Metal. The judge checked
the ten native receipts against the local engine log, exact theta hashes and
paired result fields. No new simulation was run by the judge.

**Result.** The male controls fell at 170.93 and 155.64 degrees. The candidates
remained upright at 22.59 and 19.28 degrees. Candidate standing is 2/2 and control
standing is 0/2, but qualifying is 0/2 for both. Simulated candidate speeds are
0.6983 and 0.4322 mm/s; cadences are 3.2727 and 2.9091 Hz, below the 5 Hz floor.
All six feet lift in both candidate runs. These are reused seeds, so the result
selects a working prototype without establishing a fresh-seed improvement.

The next shorter adaptation-decay endpoint fell at 170.26 degrees, and critical
joint damping fell at 163.76 degrees, both on spent 108524 against the retained
19.28-degree control. Those endpoints close. E continues improving useful gait
while retaining the standing prototype.

All four joint-search candidates increased grouped conflicts on 81547, by 1,
2 or 3, with no readable rows lost. Three decreased conflicts on the screening
seed 81548. None clears. Four of sixteen keeper IDs have been used. CUDA and
Metal screening speeds for these four match within 8.1e-11 mm/s; that bounded
parity result does not replace deterministic Metal keeper replay.

**Interpretation.** The selected search gains did not transfer to the second
identity. A holder's variation between seeds does not estimate the uncertainty
of a within-seed paired difference. It therefore does not invalidate the paired
gate. Repeating one deterministic seed adds no independent identities. The
judge approved B's proposed second-seed screening before future ranking, under
a new frozen manifest and rank. The four existing failures and acceptance gates
remain unchanged. Both identities remain development data; fresh evidence is
still required before an improvement headline.

**Search execution.** Job 658's checkpoint contains 450 unique successful rows
and zero recorded errors. The judge reconstructed all parameter vectors and
theta hashes and verified 20 worker allocations, raw archives and source pins.
The complete 12,104-vector batch continues under the account-wide ten-GPU cap,
with 6 GiB per worker. Secondary eligible-row screening must share that cap.

**Other closures.** A's hook-shunt lead falls on its repeated seed, and split
synaptic decay loses speed. B's half-ms feedback, afferent-regularity and analytic
membrane endpoints fall; adhesion balancing adds conflicts on the second seed.
C's five named deletions each lose one standing arm. D's hair-plate, mean-load,
interpolated-target and softer-foot probes lose more than 20% of speed; common
claw span falls. These close specific routes. Exact artifacts and retained
scope are in `lanes/judge/2026-09-09-build-closures.json`.

**14:29 follow-up.** On eleven spent identities the tested visual variant
retains eight complete standing pairs: four turn against visual motion, two
turn with it and two show no directional reversal. Four arms fail standing,
correcting the packet's count of nine. The judge reconstructed all 22 saved
captures and checked all 22 native receipts. This establishes variation of this
variant on these identities; it does not invalidate paired yaw or establish a
general property of the body.

D's elliptic-contact option raises the height-conflict gap by 7.20%; its implicit
body integrator leaves every saved behavioural and neural readout unchanged.
Both exact options close. In the male prototype, weaker adhesion loses the
standing flag despite a small tilt; its final-second mean height was not retained.
Faster activation raises cadence from 2.9091 to 3.3636 Hz but increases heave
from 0.8139 to 0.8858. Removing the claw phasic transform overturns the body to
170.7982 degrees. These endpoints close under their frozen rules; R596 remains.
The saved readers, source hashes and native receipts were checked without new
simulation. Decision and evidence: `exchange/2026-09-09-1429-judge-visual-variation-and-retention.md`.

**14:49 follow-up.** The zero-visual-input comparison leaves only five
leftward and six rightward contrasts readable; four blank-input arms fall.
Its frozen outcome is UNDERPOWERED, not evidence that visual steering cannot
be improved. The eleven blank and 22 paired captures reconstruct. Equal input
rate integrals in the next pattern comparison will not guarantee standing.

The male activation-dependent stiffness endpoint R603 also falls: tilt rises
from 19.2822 to 155.6708 degrees, final-second mean height falls from 1.15064 to
0.54864 mm and non-foot support rises from 0.0132 to 0.2713. The judge accepted
a reader correction limited to generated actuator-name prefixes, verified
both native receipts and retained the original failed receipt. R596 remains.
E first fixes the male panel's orientation as Robin requested.
Decision: `exchange/2026-09-09-1449-judge-walk-flight-motor-separation.md`.

**What changed in the model.** The shared theta, title and nine raw conflict
families remain unchanged. E retains a lane-local male prototype. Six visual
calibration rows were accepted in the adjacent provenance audit without changing
a parameter or biological grade. Robin assigned realistic flight to A; its
wing/body calibration and full-CNS flight goal are committed separately.

**Problems.** B's first replay attempt failed before candidate execution because
of a relative output path; its traceback is preserved. A's selected Fable model
hit quota before accepting the flight shift. By 14:08 it was active on Fable
and tracing wing motor output and the flight body. The model-choice question
was withdrawn. Neither event is a biological result.

**Attribution.** Lane E built and recorded the male comparisons and videos.
Lane B froze and replayed the joint candidates. Compute ran the cluster batch
and checkpoint audit. The Codex judge independently checked the saved evidence,
retained the paired gate and assigned flight to A on Robin's instruction.

Decision: `exchange/2026-09-09-1400-judge-standing-and-joint-checkpoint.md`.

## Follow-up, 15:08 EEST

**Question.** Do contact torsion, weaker leg equilibrium gain or lower command
drive improve walking while retaining body support and all-leg movement?

**Prediction.** D required a useful change in heave or straightness with
walking retained. E required at least 5% less heave, standing, all-leg lifts,
retained cadence and no rise in non-foot support.

**Method.** Each change was paired with its exact control on an owned spent
identity using deterministic Metal. The judge ran the committed pure readers,
verified 45, 38 and 35 source bindings, compared E's entire retained control
against R602 and matched all six native receipts. No simulation was repeated.

**Result.** None cleared its stopping rule.

| Change | Control | Candidate | Consequence |
|---|---:|---:|---|
| D R94 ground torsion, speed mm/s | 3.0031 | 2.6107 | Standing retained; heave and straightness also worsen |
| E R604 weaker CTr gain, heave fraction | 0.8139 | 0.9097 | Standing retained; heave rises |
| E R605 half command rate, left-hind lifts | 7 | 2 | Standing retained; all-leg retention fails |

R605's heave falls only 0.49%, below its 5% criterion, while non-foot support
rises from 0.0132 to 0.0200. R596 remains the male standing lead and still
qualifies on neither of its two reused identities.

**Visual input.** C193 swaps the active slot pattern and waveform of the saved
synthetic tape. The registered contrast counts are 6, 5, 5 and 8; none is
significant and the result is UNDERPOWERED. Left-slot standing falls from
8/11 to 5/11 after the waveform swap, despite matched aggregate input.
The judge reconstructed all 45 new and retained captures with frozen execution
modules and verified all 23 new native receipts. Equal aggregate input is
insufficient to guarantee body support. The post-hoc 4/4 crossed-pair sign
remains conditional on four surviving pairs and grants no steering claim.

**Interpretation and next step.** The three named walking endpoints close.
D has moved to Robin's shared walking/flight motor-selection priority. C may
test the crossed-input hypothesis on declared held-out spent identities with
falls and per-pair displacement explicit. E continues its useful-gait question.
The male display rotation also passed a separate frame and geometry check;
it changes the viewing orientation only.

**Other review.** B's frozen secondary screen binds 78 vectors and one holder
row on spent 81547. Its nine file bindings and twelve existing tests pass;
Compute must validate the cluster environment before execution. The original
array and paired Metal gate remain. The MaleCNS source audit verifies the five
official raw-file checksums but promotes no neurotransmitter assignments.

**Attribution.** Lanes D, E and C ran the comparisons. Lane B prepared the
secondary handoff. The Codex judge checked the retained evidence and committed
the scoped decisions. Robin assigned shared motor selection and requested the
male-panel orientation correction.

Decision and audit paths: `exchange/2026-09-09-1508-judge-pass.md`.

## Follow-up, 15:28 EEST

The existing search centre clears the paired spent gate. Its grouped conflicts
are 7/7 against wi1's 9/7 on 81548/81547. Both candidate arms retain the complete
motion predicate; speeds are 3.2754/2.4713 mm/s and cadences 5.7273/5.0000 Hz.
The second identity is slower than wi1 and sits exactly on the cadence floor,
but the frozen criterion passes. The judge keeps that criterion unchanged.

The exact centre theta is already loco-09766. No fresh retry or new candidate
identity follows. Fourteen reconstructed vectors, 28 raw-to-card mappings,
927 runtime source bindings and 28 native receipts verify. The first runner
was modified at launch and archived after execution; the second runner matches
its committed source and independently establishes the same paired result.
Repeated centre runs add no independent seeds. All twelve displaced points
fail the paired gate, which limits the tested directions without proving that
the entire search population is noise.

B's bounded nearer-centre Metal comparison was already running during the
review and is preserved. Its original bindings and result remain to be checked;
a duplicate CUDA screen is unnecessary. The original array and secondary screen
continue under the same resource and keeper limits. D's shared walking/flight motor-state work and
A's flight mechanics remain active. Nothing in this walking gate establishes
appropriate flight-power output, a fresh gain or a champion change.

Decision: `exchange/2026-09-09-1528-judge-center-gate.md`.

Further adjudication: `exchange/2026-09-09-1600-judge-pass.md` records two spent shell clearances, corrects the cap-publication chronology, closes male R606 and binds the reference-only secondary repair. The shared champion remains unchanged.

## Follow-up, 23:23 EEST

Two later male routes ended at their first standing failure and are on film.
R639 raised nine left-hind CTr flexor rows and two left-hind FTi extensor rows
to the muscle box's upper bound 3.0 on spent 108523: the male overturns (final
tilt 155.6 degrees, root height 0.55 mm, non-foot force share 0.195 against the
control's 0.013). R643 changed only the right-front CTr and FTi pitch readout
denominators from 0.0025 to 0.00375 through the shared body_joint_drive_scale
key: the male stays upright but drops to 0.914 mm against the 1.0 mm standing
floor with more non-foot support. Both controls reproduce R596 exactly. The
R640 and R644 films replay the archived records with every body position exact.
Neither route is revived; E's next test transplants the standing female's
per-cell load-input operating points into the male through the existing six
leg gains. Records: `exchange/2026-09-09-2224-laneE-judge-packet.md`,
`exchange/2026-09-09-2313-laneE-judge-packet.md`.

Two more films complete the evening's male record. R637 replays R633, the
standing R596 male with the female's mapped per-type delays (67 type
coordinates, VNC delay scale 0.7): the fly changes heading and settles below
the frozen standing-height floor without overturning, so that route closed on
its first seed. R647 replays R645, the same male with the standing female's
per-cell load-input operating points through the six existing leg gains: it
stays upright with irregular steps, non-foot support falls from 0.0126 to
0.0036 and every foot lifts, but net speed drops from 0.698 to 0.624 mm/s,
below the 1 mm/s target, so that route closed too. The comparison that
frames what remains is already on record (R642): the unchanged female fly at
the male's constant DNg100 command walks 2.56 mm/s at 5.0 Hz on the same seed,
so the male's slowness is a property of the male nervous system and body, not
of the command it receives.

## R648 and R649, 00:0x EEST

The next assumed parameter, a motor reset at rest on the 369 assigned male
motor rows (364 values moved inside [-68, -48] mV), overturned the male on
spent 108523: root height 0.549 mm, final tilt 155.6 deg, non-foot force share
0.38, against the standing control at 1.142 mm and 0.0126. Dropped at its
first gate. The film of the overturn is R650, reproduced exactly from the
archived run.

R649 answered the judge's bar from the captured records without an engine
second: mapped motor firing is higher in the male overall (19.25 against
11.81 Hz per cell), but the right-front CTr flexors fire at 4.96 Hz where the
female's fire at 20.74, with less negative input (0.010 against 0.043). The
slowness is localised to one pool, and the next step is to watch the 403
cells that feed it, not to move another parameter.

Source packet: `exchange/2026-09-10-0003-laneE-judge-packet.md`.

## R658 and R660, 01:0x EEST

The one type whose excitation onto the male's slow right-front CTr flexors
falls shortest of the female's is IN19B012, a single male cell. Doubling that
cell's output gain (R658, bounded [0.25, 2], labelled ASSUMED, spent 108523)
did what the arithmetic said it would: the flexors fired harder, the male kept
standing and put no more force through anything but its feet. It also heaved
more and did not reach 1 mm/s, so the route was dropped at that gate by the
rule written before the run. The control reproduced the R596 record exactly
(439 leaves, zero delta). R660 is the film of the changed rows, reproduced from
the archived run and verified exact against its public copy (4,276,234 bytes).
A per-leg muscle gain that would finish the job by hand (R662) was refused for
the night; whether the male's right-front pool is thin by reconstruction is the
next read, without an engine second.

## R671 to R675, 02:4x to 03:0x EEST

The first causal test on the slow right-front pool doubled the output gain of
IN21A010, cell 801894 (`type_out_gain_id`, shared bounds [0.25, 2], labelled
ASSUMED, spent 108523). Its 186 outgoing weights doubled and nothing else moved;
204 of its 815 raw synapses land on mapped right-front motor neurons and 611
elsewhere, so this was not a CTr-only intervention. The control reproduced the
R596 standing record exactly (final tilt 22.59). The doubled arm fell, final
tilt 170.89, non-foot support 0.0126 to 0.3284, while the cell's own rate
dropped from 27.82 to 17.64 Hz and the right-front flexors gained five spikes
over a window that includes the fall. R671 is REJECTED under its own rule. R672
is the film of the changed rows, reproduced from the archived run and verified
exact against its public copy by the judge's own download (3,842,362 bytes,
sha256 b0a4bf13). The fly stands at first and lies upside down later; the
labels state the failure, the intervention value and the spent seed.

R673 asked whether the male has the female's one clock (entry 80). Reading the
four archived male standing records on two seeds with C's N229 arithmetic: it
does not. Pair coherence of the CTr_pitch commands is 0.21, 40 of 60 pairs sit
below the 0.30 floor, the dominant frequency wanders between 3.7 and 5.2 Hz,
and the across-tripod median is 70 degrees with within-tripod 80, which is
scatter, not alternation. R674 found the same below-floor coherence in the
flexor, extensor and difference commands. The male does not have the female's
problem; it has no rhythm to phase.

R675, without an engine, read why the right-front pool is thin: it receives
four direct synapses from its own accepted afferents, and its premotor cells
receive 1,890, 0.102 per cent of their input, against 0.33 to 1.67 per cent for
the other five legs; the right-front leg has no accepted claw-flexion or
hair-plate rows. Own-leg afferents supply 19.4 per cent of the accepted
afferent input to that pool, against 51 to 85 per cent elsewhere. That is an
anatomical input deficit under the current sensory selection, not yet its
behavioural cause; R676 partitions every annotated leg afferent into accepted
and undriven rows before any feedback intervention is proposed.

Source packets: `exchange/2026-09-10-0244-laneE-judge-packet.md`,
`exchange/2026-09-10-0256-laneE-judge-packet.md`.

## R677 to R682, 03:1x to 04:1x EEST

Nothing moved the model. R677 found that the legacy bristle selector admitted
688 gustatory cells; the corrected source, `bristle_source_mode native_tactile`,
joins exactly the 1,878 MaleCNS mechanosensory_tactile cells and is integrated
absent-by-default, with Compute's replay of the 81548 default bit-identical.
Switched on with the assumed contact law (5 Hz when the leg's raw load exceeds
1), it fells the male: the R596 control stands at 0.698 mm/s with every saved
array exact, and the tactile arm ends upside down, final tilt 155.6 degrees,
nonfoot force 0.876 (R679, spent 108523). The
[failure film](https://media.theuploadinglab.ai/walker_r679_tactile_r680_108523.sha256-b487e9fbb344746d8630bced7be4673b2fb331573569b24602409744483eda75.mp4)
(R680) reproduces all four delivered-input arrays, all 19 capture fields and
480 poses exactly; public 3,097,102 bytes, SHA256
`b487e9fbb344746d8630bced7be4673b2fb331573569b24602409744483eda75`, verified by
judge download. Adding input rows also moves the renewal noise stream, so this
is a configuration comparison, not an exclusive test of tactile signalling.
R681, without an engine, showed that the shared muscle box cannot equalize
both hind joints at once: the tibia branch ratio moves from 0.113 to at most
0.115 across the whole box while the hip ratio can reach 0.81. R682 declares
the hip-only corner (three muscle_gain entries inside the box) for one
control/candidate pair.

Source packets: `exchange/2026-09-10-0312-laneE-judge-packet.md`,
`exchange/2026-09-10-0352-laneE-judge-packet.md`,
`exchange/2026-09-10-0409-laneE-judge-packet.md`.

## R682 and R685, 04:5x to 05:1x EEST

Nothing moved the model. R682 ran the hip-only corner of the shared muscle box
that R681 had computed (trochanter extensor 3.0, trochanter flexor 0.3,
accessory trochanter flexor 3.0, all inside the declared bounds, 68 motor rows
and the six CTr columns changed) as one control and candidate pair on spent
108523. The R596 control reproduced its archive on every compared leaf; the
candidate ended upside down, final tilt 171 degrees, nonfoot support 0.92. The
[failure film](https://media.theuploadinglab.ai/walker_r682_ctr_r685_108523.sha256-a168d61da7b89271dd8319d88f0777d75011cd1be1b6dd03c077f92354007bfa.mp4)
(R685) reproduces all five motor arrays, four delivered-input arrays and 480
full poses exactly; public 3,257,293 bytes, SHA256
`a168d61da7b89271dd8319d88f0777d75011cd1be1b6dd03c077f92354007bfa`, verified by
judge download. A frozen-count branch ratio that looks better on paper does not
stand the male up. R683's reading of R679 stands beside it: adding input rows
moves the shared renewal draws and the command timing before any tactile
event, so the tactile fall is not yet a tactile result; the 0 and 5 Hz pair
with the same input list is the test, behind the default gate.

Source packet: `exchange/2026-09-10-0512-laneE-judge-packet.md`.
