---
id: 78
date: 2026-09-09
title: Flight motor activity in the walking model
tldr: The walking command also activates flight-power motor neurons. Robin made motor-state separation shared work for D and A. An old reader mislabeled the DNg02 body as fallen; the accepted standing record is true. No model improvement or physiological band is claimed.
era: standard
commit: the commit this entry lands in, on main
theta: results-wi1-theta.json; cmd alone changes in two flight-command arms
seeds: spent A 203760
n: three whole-CNS arms
wall: 45.48 s including load and three arms
reproduce: lanes/judge/2026-09-09-fl1-flight-review.json
video_receipt: publication-receipts/fl2_flybody_lift_ellipsoid.sha256-e7527a9a5ba4cde10dfcc89bbc7d39e1a18b9f21cf170eddfa84525e192e6f98.publication.json, publication-receipts/fl2b_hover_closed_loop.sha256-6f18b6096b9b391c9cd279acbc6aecd793aacf378c90eb61f92619b948eabf92.publication.json, publication-receipts/loco09766_centre_81548.sha256-af3258e869374fdb9cff46d9aa53a4e5dcb18c816234807e0fbc5a7a3075bb6e.publication.json, lanes/D/2026-09-09-r112-publication.json, lanes/D/2026-09-09-r113-publication.json, lanes/D/2026-09-09-r114-publication.json, lanes/D/2026-09-09-r115-publication.json, lanes/D/2026-09-10-r116-publication.json, lanes/D/2026-09-10-r117-publication.json, lanes/D/2026-09-10-r118-publication.json, lanes/D/2026-09-10-r120-resumed-publication.json, lanes/D/2026-09-10-r124-publication.json, lanes/D/2026-09-10-r127-publication.json, lanes/D/2026-09-10-r129-publication.json
---

**Question.** Does the walking model keep flight-power output inactive, and
can a different descending command recruit it selectively?

**Prediction.** A's pilot required control power-pool mean below 2 Hz and
sustained 2–40 Hz activity in at least half the power cells under a flight
command. These are prototype selection assumptions, not measured walking bands.

**Method.** A ran the whole 188,508-neuron BANC network on one reused seed.
The body and theta were unchanged except for DNg100 versus DNg02 or DNp31
command populations. It recorded motor activity in 5 ms bins, then kept
population and wing-cell summaries over 1–6 seconds. The judge checked the
17 execution source hashes, the BANC cell census, retained wing summaries
and all three native Metal receipts. The original bins and body captures
were not retained, so an independent raw-trace reconstruction is not available.

**Result.** With the walking command, DLM1–4 averages 21.5 Hz and DLM5 19.3 Hz;
the complete 24-cell power pool averages 10.175 Hz. The 40-cell steering pool
averages 16.295 Hz and the 49 neck cells average 42.3673 Hz. DNg02 and DNp31
raise the power-pool mean to 34.825 and 16.600 Hz. The frozen pilot outcome
is CONTROL_POWER_POOL_NOT_QUIET.

**Correction.** The DNg02 arm was reported as fallen because the table used
the old heading-sensitive quaternion-w test. Its accepted standing_v32 flag
is true and mean final tilt is 11.8 degrees. All three stored accepted
standing flags are true. This is a correction of the retained record;
no raw body capture was reconstructed.

**Interpretation.** Robin made this a shared natural-walking and flight
priority. D owns reducing inappropriate power recruitment while preserving
walking and a usable flight response. A continues the wing/body model; B
will add the instrument prospectively to future candidate diagnostics.
The blanket assertion that every neck neuron should be quiet is not established:
walking Drosophila move their heads, and identified neck neurons control
head posture. Neither observation validates 42.3673 Hz; it requires a
cell- and state-specific reference.
[Walking head movements](https://pubmed.ncbi.nlm.nih.gov/25386124/);
[neck motor control](https://www.nature.com/articles/s41586-024-07222-5).

**Problems and next step.** A's lane-only fetch mapping left its origin/main
reference stale. A repaired the mapping and read assignment 82122d54b;
the judge verified both the main tracking entry and updated goal reference.
D adopted its new goal and started the instrument, which must retain
raw motor activity and the canonical body capture. Shared theta, titles and
physiological grades remain unchanged.

**Attribution.** Lane A ran FL1. Robin broadened the work to natural walking.
The Codex judge checked the saved evidence, corrected the standing claim and
assigned the shared motor-selection question to D.

Decision: `exchange/2026-09-09-1449-judge-walk-flight-motor-separation.md`.

## Follow-up, 2026-09-09 16:02 EEST

**Motor selection.** D now retains all 805 motor rows, raw 5 ms spike bins and
the canonical body capture. Its recording control preserves every body field
and original spike bin. The judge reconstructed both comparisons and integrated
the exact observer; the default-off compatibility probe is BIT-IDENTICAL.
The scientific comparisons use D's spent seed 61524 and deterministic Metal.

R95 changes thresholds only in the 24 power cells. Walking power activity
falls from 9.5083 to 3.8583 Hz, while displacement retains 75.6056% and the
flight-command increment retains 60.1236%. Both are below the required 80%.
R96 changes the command to membrane input. Power activity rises to 20.6083 Hz,
displacement retains 68.0094% and walking qualification fails. Both endpoints
close. B will record the motor diagnostic prospectively; no blanket neck or
steering silence band is established.

**Flight-body control.** A's FL2 prototype uses prescribed wing motion, with no
CNS or neural seed. The judge replayed both configurations and the three
time-step conditions; every scientific field matches. Ellipsoid aerodynamics
supplies 0.9304818 of body weight while tethered, versus 0.2337338 for
inertia-box aerodynamics. Its 3.678909 nNm nose-down mean torque accompanies a
tumble within about five wingbeats after release. The displayed clip records
that tethered control and release. It is not neural flight.

The prescribed pattern is 218 Hz. Lift/weight is 0.9512, 0.9305 and 0.9200 as
the integration step falls from 100 to 50 to 25 microseconds. The partial-lift
conclusion persists. Servo power includes dominant numerical damping and
cannot be read as muscle power. A continues trim, stability and perturbation
controls before the whole-CNS loop. This untrimmed tumble supplies no biological
feedback-latency band.

**Attribution and record.** A produced FL2; D produced the recorder and
R95/R96. The judge reconstructed their saved evidence, replayed FL2, checked
the exact public clip bytes and integrated the observer. New receipts:
`lanes/judge/2026-09-09-motor-selection-review.json` and
`lanes/judge/2026-09-09-fl2-review.json`. The original FL1 metadata above remains
that pilot's metadata; this follow-up has separate run identities. Shared wi1,
fresh-seed licenses and physiological grades are unchanged.
Decision: `exchange/2026-09-09-1600-judge-pass.md`.

## Flight-body control, 2026-09-09 16:38 EEST

This follow-up has separate FL2b metadata: no CNS seed; two trim configurations,
release controls, perturbation and a delay grid; independent wall time 38.8 s.
Reproduce with `.venv/bin/python lanes/A/fl2b_control_derivatives.py --out lanes/judge/fl2b-replay.json`.

**Question.** Can a calibrated wing pattern support the body, and which motor
adjustments must a neural controller deliver to maintain its pose?

**Method.** A reused the Flybody flight configuration in FlyGym 2.1.0 with
MuJoCo 3.9.0. Prescribed wings beat at 218 Hz. An initial amplitude/stroke-shift
trim left forward force equal to 20.6% of body weight. A then disclosed and
added stroke-plane angle as a third calibration parameter, keeping both arms.
The external controller updates from the previous wingbeat's mean body state.
No nervous system, learned policy or neural seed runs in this control.

**Result.** The three-dial trim supplies 0.999775 of body weight with mean
pitch torque 0.002376 nNm. Over one simulated second, maximum height error
is 0.2523 mm and maximum pitch error is 2.7046 degrees. Horizontal drift is
+7.4941 mm in x and -2.4176 mm in y. The frozen HOVERS label checks height
and pitch only; stationary three-dimensional hover has not been established.
After a 6 nNm pulse lasting 5 ms, pitch excursion peaks at 20.1728 degrees
and returns within 2 degrees after 97.1284 ms. The clip shows the external
body control, including the perturbation and recovery.

**Limits.** Twenty milliseconds is the largest passing delay on the tested
grid, with a 4 Hz controller bandwidth. The 30 and 40 ms rows fail; 21–29 ms
were not tested. This finite controller comparison supplies no universal neural
deadline. Its cited sub-millisecond haltere result comes from Calliphora.

**Power correction.** The lane packet compared 44.6838 W/kg of body mass
with 50–130 W/kg of muscle mass. Fry et al. assume flight muscle is 30% of
body mass, so their corresponding body-mass range is 15–39 W/kg. The model's
trimmed aerodynamic power is 14.57% above that reported maximum. The claimed
within-range match is withdrawn; six reported hovering flies do not define
a new hard physiological band. Numerical servo power is a separate quantity.
[Fry et al. 2005, Eq. 6 and Table 1](https://authors.library.caltech.edu/records/1t77q-80135/files/FRYjeb05.pdf).

**Verification and next step.** The judge replayed the complete numerical
control and matched every scientific field, then imported the exact sources,
tests, pattern and results. All 23 tests pass on main. Its frozen full-CNS
compatibility probe remains BIT-IDENTICAL. The public clip bytes match the
publication receipt and its frames were inspected. A now has an executable
body response for its explicit motor, muscle, wing and sensory-feedback loop.
The selected neural fly, fresh-seed licenses and physiological grades stay fixed.

**Other decisions in this pass.** B's 16 sparse and 16 distant vectors add
no paired improvement; the complete 12,104-vector search continues. C194's
held-out crossed comparison gives 9 correct signs and 7 opposite signs among
16 readable pairs, with walking-retention losses. C195's static direction paths
and archived DNa02 firing bias reproduce, but do not establish causal steering.
Its missing first attempt of 64 partial arms must be recovered or documented
from the remaining logs; changed authority must be revalidated before another
invocation. D R97 retains only 63.64% displacement and closes. E R607 improves
some gait quantities but raises heave from 0.8139 to 0.9608 and closes. E's
inspected calcium sources do not calibrate command spike intervals.

**Attribution.** Lane A produced FL2 and FL2b. The Codex judge reproduced the
body control, corrected the power units, checked the other retained results and
integrated the calibration component. This is not connectome-driven flight.

Decision: exchange/2026-09-09-1638-judge-pass.md. Reproduction evidence:
lanes/judge/2026-09-09-fl2b-review.json and the adjacent replay receipt.

### One full model must retain the walking and flight boundaries

Robin's Janelia reference makes coordinated whole-body walking and flight leg
posture explicit requirements. His follow-up applies every subsystem's
constraints to the eventual full fly. The judge registered four requirements:
flight leg posture, sensory action selection by state, whole-body walking
trajectories and speed/turn-dependent leg coordination. Their numerical readers
remain open; they do not add four demonstrated conflicts to the current card.

The primary sources change how those requirements can be tested. The published
Flybody walking reference uses tracked 2D landmarks; body height, pitch and
full joint poses were partly estimated. Its released flight and walking tasks
can disable the legs and wings respectively. Those controls therefore do not
establish a single controller that switches the intact body between behaviors.
Ache et al.'s tethered female flies give a direct state comparison: identical
frontal looming evokes escape preparation while perched and leg extension for
landing during flight. Keeping legs tucked through that landing stimulus would
miss the observed behavior.

The shared mission, judge contract, registry and all six goals now require
applicable constraints on the same complete neural/body/controller identity.
The current neural title remains wi1. New comparisons will reuse retained
full-pose traces and existing walking readers, with source scope and missing
coverage recorded separately. The running joint search retains its frozen rank.

Directive and sources: exchange/2026-09-09-robin-unified-boundaries.md.
Audit: lanes/judge/2026-09-09-unified-boundary-source-review.json.

The secondary checker handoff was already published by B at 16:12. The judge
confirmed the unchanged 79 inputs, source/task binding and preserved failed
version; Compute can use its existing one-execution authorization after the
cluster checks. The earlier live card's handoff wait is corrected.

### Pooled walking scores can conceal per-leg differences

The judge reproduced a limitation of the current contralateral-phase scalar:
15 of 18 deliberately unequal-rate signal pairs still fall inside its passing
interval. In B's archived duty-factor sample, 209 actual graded passes retain
large differences among unfiltered per-leg stance fractions. B's larger pass
count included records without a graded value, and its new tripod grouping
swapped middle and hind legs. Those interpretations are corrected.

These results make Robin's request for coherent whole-body walking testable:
per-leg timing must be compared with body pose, joint movement and contact,
using the same event filter and applicable animal conditions. A finite archive
of threshold crossings cannot establish that no parameter combination walks.
The current title and running search stay fixed while these readouts are
validated. Decision: `exchange/2026-09-09-1722-judge-walking-reader-review.md`.


### Flight calibration and the identity of a shared constraint

A's FL3 component converts proposed motor activity into wing commands and
provides an external pitch/roll/altitude control for body calibration. On main,
its 22 tests pass, its complete body replay matches and the canonical neural
compatibility probe is BIT-IDENTICAL. The final steering and altitude additions
can exceed the power-amplitude clamp; the mapping and controller remain model
assumptions. A's subsequent neural loop uses separate walking and flight bodies
and reports tumbles. Flight leg posture and shared-body transitions remain open.

The judge found a motor identity error in C's walking analysis. Its 493 alleged
leg motors include 124 non-leg motors and omit 22 leg motors. The retained
contrast therefore mixes legs, wings, halteres and other effectors. C's DNa02
response remains a separate result; the completed eye-to-leg-chain inference
is withdrawn. The next leg readout uses effector labels and retained per-cell
traces while C's active body comparison continues.

E's replay shows why model identity also includes numerical implementation.
Identical inputs give the same total spike count over 52 ms, yet a cell fires
at 51.0 ms on Metal and 51.1 ms on NumPy. Two captured exponential-decay
calculations differ. This does not establish which complete engine is correct
or explain a later gait change. A constraint transfers between implementations
only after parity for the dynamics and observation being claimed.

No neural-model improvement, title change, new physiological band or fresh-seed
license follows. A produced the body component; C and E produced the source
runs. The Codex judge replayed the calibration, reconstructed the motor pool
and numerical traces, and corrected the shared record. Decision:
`exchange/2026-09-09-1742-judge-integration-review.md`.


D's next two source-specific interventions also fail their retained walking
requirements. Halving the DNp31-to-power projection retains 62.56% displacement;
raising the two DNp31 thresholds retains 65.03%. Both increase power activity.
The judge reconstructed the motor records and delayed source inputs. Those
settings close while independent motor research continues.

The later walking audit also exposed a model-identity mismatch: B's per-leg
instrument used a 33,004-neuron subset. Those observations cannot establish
full-CNS wi1 behavior. The proposed per-leg contact floor does not follow from
absence of an aerial phase, and a coupled limb response does not by itself
invalidate a reader. The completed secondary GPU execution now awaits a
reader-only correction for explicitly missing gait fields in failed-standing
records, preserving all 79 outputs and the frozen ranking. Decision:
`exchange/2026-09-09-1754-judge-reader-and-motor-review.md`.


Robin asked what produced progress during the past week and how the fleet
should change its behavior. The judge's archive review finds the clearest
walking gains after inspecting delivered sensory input and motor-to-body
conversion, followed by small paired interventions. The stored wd2/we1 speed
comparison rises from 1.532 to 2.115 mm/s on 22 mutually standing pairs; both
arms stand on 23 of 24. The large search has not yet produced a retained gain.
Changes to engines, readers and standing policy prevent one comparable weekly
percentage. The six goals now prioritize executable effects, failed-record
preflight and early endpoint decisions, preserving the current search and
whole-model constraints. Evidence, costs and the historical comparison limits:
`lanes/judge/2026-09-09-weekly-progress-review.md`.


The repaired secondary reader now reconstructs all 79 original records,
including 16 with failed standing and missing gait, without changing their data.
The judge accepts that exact-corpus repair. The next executable obstacle is in
the frozen ranker's interface and its conversion of missing values; its rule
already excludes these records. Compute can finish the source audit and existing
parity comparisons while B repairs the interface. The later sensitivity data
remain reduced-graph parameter effects: divergence does not establish random
noise or invalidate a paired total-effect comparison. Decision:
`exchange/2026-09-09-1820-judge-secondary-reader-decision.md`.


Robin's robotics question adds a practical test: establish what a signal changes
in the body and how that response depends on timing and behavioural state.
Current code already sends simulated contact loads to sensory neurons. A
separate gap is that the active Metal backend rejects the optional sensory
terminal-inhibition feature available in NumPy. Animal experiments support
phase-sensitive steering and different mechanisms for halting during feeding
and grooming. These define comparisons for the common fly; no model gain or
new numerical pass band follows from this source review. The note and sources
are in `exchange/2026-09-09-robotics-signalling.md`.


### Timing a motor signal and testing the resulting movement

The secondary search's 79 records now pass the full-source audit. Its repaired
rank keeps the original order and routes four more candidates to the existing
paired Metal comparison. This completes an execution interface; it adds no
model gain. All missing gait records remain in the inventory.

The flight review finds that a reported28.75 ms half-rise times an unweighted
motor aggregate. The gain-weighted pitch projection responds halfway by11.25 ms
before the actuator window. Neither identifies a pure neural transport delay.
Left/right steering responses also cancel in one reported bilateral mean.
The next useful comparison follows the final wing command into body motion.

The visual lane corrected its leg assignments and prototype comparator. Plain
wi1 stands on20/24 seeds against11/24 with the lane's optic-lobe exemption.
The tested gain ladder does not retain enough walking pairs. Its unsigned motor
counts do not establish where turning fails or whether every leg moves.
These questions need joint commands, poses and contacts in the same model.

Two power-cell interventions retain46.64% and66.62% of walking displacement;
both close. Male friction2 and load0 settings overturn the standing prototype
and also close. No neural gain or new physiological bound follows. A supplied
the flight traces, B/Compute the secondary execution and rank, C the visual
records, D the power-cell arms and E the male contact captures. The Codex judge
reconstructed their numeric results and revised the six goals. Decision:
`exchange/2026-09-09-judge-signals-and-secondary-rank.md`.

### Retracted flight legs and feedback from actual joint motion

The shared flight calibration now supports a retracted leg pose from the
Flybody source. The legs are set programmatically and remain rigid. Simulated
body mass is unchanged; the roll-axis inertia falls by 9.01%. The model's
joint kinematics and trim replay, and old default playback stays bit-identical.
The final unified fly still needs neural leg control, landing extension and
contacts. The source still is `lanes/A/fl2c_leg_pose_spawn_vs_flight.png`;
its label identifies the fixed pose and the prescribed wings.

The contact-gain experiment produces no candidate that passes both spent
seeds among its six tested doses. Those points close; the continuous parameter
and other contact pathways remain available. E's actual-joint feedback raises
raw displacement from 2.38 to 6.82 mm but lowers final-second root height from
1.151 to 0.914 mm and increases non-foot support. The failure is height,
without sampled overturning. All saved sensory values reconstruct against
the native input tape, so this experiment identifies a delivered intervention
whose present setting loses retention. It does not establish a walking gain.

Lane A supplied the pose and recalibration, B the contact doses and E the
physical-state/input captures. The Codex judge integrated A's exact components,
replayed the body and feedback evidence, and updated the six goals. The body
checks and 45 component tests pass. Six avoidable repeat runs after B's reader
failure are retained in the execution account; future aggregation reuses
completed arms. Decision: `exchange/2026-09-09-judge-flight-pose-and-feedback.md`.


### A joint-search candidate clears its screen

Candidate 00616 clears the fixed paired spent-seed screen. Simulated speed on
81548 rises from 1.8791 to 2.8872 mm/s and front-knee range enters its band;
abdomen excursion also becomes a new conflict at 12.2733 degrees. Grouped counts
change9 to 8 on that seed and stay7 on 81547. All candidate motion gates pass.
The candidate therefore advances under the original net-count rule while the
new conflict stays recorded. wi1 remains the shared model. Exact candidate
footage and a committed fresh-seed runner are the next steps; 81572/81573 are
reserved but not licensed. The full primary search continues.

The signalling results constrain the next experiments. A's positive flight
half-rise records reproduce, but a zero signal incorrectly receives t50=0;
that reader needs its negative case fixed. C's motor-pool lateralization under
larger outgoing DN weights does not define the amplitude required from vision.
D's hook-input change increases flight-power activity while the body falls.
E's withdrawal of external command input leaves command-cell spikes and leg
motion, then fails height/support. A verified intervention must be followed
through its actual neural output, actuator command and body response.

Lanes A–E supplied the raw experiments; the Codex judge independently replayed
the saved records, retained the failed attempts and updated all six goals.
No new neural simulation or physiological interval was needed for these
reviews. Details and immutable source identities: `exchange/2026-09-09-judge-00616-and-signalling.md`.


### Robin's word: progress to the model, and the centre's fresh pair

Robin asked for ambitious goals that drive the model forward and for every
seat to work until he returns at noon. The judge rewrote all six goals around
one question, whether the fly got closer to a real fly, and named per-lane
targets: neural flight that holds attitude, walking coordination by mechanism,
a correct-direction visual turn, state-appropriate motor programs, a walking
male.

Lane B corrected itself: the loco-09766 centre, not 00616, is the best arm on
the frozen paired gate (7/7 against wi1's 9/7, nothing newly opened), and it
now has footage rendered by the unmodified canonical renderer with parity. The
judge regraded all 273 committed records and reproduced both the clearance and
the census: among 120 arms that walk, the gain search never brought body heave,
hind-leg antiphase, body height or straightness into band. The film shows the
fly stepping on six legs and then pitching and rolling. The centre is licensed
on two fresh seeds; the search array finishes and no new gain arms start.

Lane A repaired its timing reader's zero case and showed that measured response
kernels cannot stabilise the replayed flight power while an angle-integrating
reflex can. Lane C found that the pooled leg total was blind to a steering
command the per-joint actuator shows on 23 of 24 seeds, and that joint and body
disagree in sign. Lanes D and E each closed one endpoint. No neural gain; wi1
remains. Decision: exchange/2026-09-09-judge-centre-license-and-goals.md.


**R112 control admission, 2026-09-09 21:36 EEST.** Nothing moved the model.
The planned command-switch comparison added 38 initially silent DNg02 input
entries. Its walking control stays upright and qualifying, but displacement
falls from 16.517039 to 12.215829 mm, retaining 73.958951% against the existing
80% floor. The switch arm was not run. All native weights and neural parameters
remain unchanged. The extra input entries alter the initial renewal draws;
this fixed-entry comparison is closed on the tested seed.

The clip shows that control on spent D 61524. One scientific arm and one
presentation replay completed on deterministic Metal, each running all 188,508
neurons for six simulated seconds. All 20 body fields and all 851 common
recorded spike rows match exactly. The film uses achieved joint angles and
plays six simulated seconds in 16 seconds; its displayed tip cadence is
6.090909 Hz, while the separate h10 gate reads 5.181818 Hz. The public
7,005,924-byte MP4 matches the local SHA256 exactly. No fresh replication,
physical-flight result or model gain is claimed.

D ran the experiment and certified the presentation. Primary execution
`f6a4a4f2655b39cdec0e108b248e9815d20c3fad`, total wall 102.180 s;
presentation `b16a51e5814f4839bd3e7af262e25ca29b255a5b`, 174.233 s.
Thirteen tests pass. Raw results, retained theta and reproduction commands:
`lanes/D/2026-09-09-wi1-command-switch-r112.json`,
`lanes/D/2026-09-09-wi1-command-switch-video-r112.json` and
`exchange/2026-09-09-2136-laneD-judge-packet.md`. The next comparison keeps
the original sensory input list and adds flight-command candidate events
through the existing Metal interface, retaining the refractory check.


**R113 command switch, 2026-09-09 22:24 EEST.** Nothing moved the model.
The control exactly reproduces wi1. At three seconds the native input switches
from DNg100 to DNg02 while the original sensory sampler and all parameters stay
fixed. In the final two seconds, power-cell output rises from 11.916667 to
35.083333 Hz; leg output retains 90.1% and actual joint motion 97.4%.
DNg100 still fires at 34.75 Hz after its external candidates are removed.
All six legs retain positive mean ground load. These are readings from the dynamic
42-joint walking body; the flight body with programmatically retracted, fixed
legs cannot validate this leg response.

The clip shows the switch on owned spent D 61524, with six simulated seconds
played in 16 seconds. Every canonical body field, all 851 common recorded spike
rows, callback input, native weights and command-event tape match the scientific
arm exactly. Public bytes match the local SHA256. The switch remains standing
but loses qualification and covers 10.138268 mm versus control's 16.517039 mm.
Its displayed tip cadence is 6.181818 Hz; the h10 gate is 4.636364 Hz. No fresh
replication, physiological retraction interval or physical-flight gain is claimed.

Two scientific native arms and one presentation arm ran. The initial invocation
stopped on changed authority after its completed control; the continuation
preserved that record and ran only the unrun switch. D's independent reader v3
certifies both results and the film. Source identities, exact commands and the
per-leg rate/angle comparison are in `exchange/2026-09-09-2224-laneD-judge-packet.md`. The next causal test changes
adaptation only in the two walking-command cells, fixed across command states,
and retains walking and flight-response checks.

## R114: reduced DNg100 output, increased walking power

Nothing moved the model. Setting assumed spike-triggered adaptation to 8 mV
in exactly the two DNg100 cells reduced their walking output from 108.6 to
93.8 Hz, but raised power-cell output from 9.51 to 17.28 Hz. The fly stood
and retained 80.25% of displacement. Body height rose from 1.205 to 1.272 mm,
which failed the existing height-retention gate, so the flight-switch arm
was not run. The endpoint is closed. These are simulated results on spent
seed 61524; no physiological band or gain is claimed.

The unadapted control reproduces archived wi1 exactly. The changed walking
arm has a live canonical clip: 7,307,895 public bytes, identical to the local
master. All body fields, recorded neural rows, command events and adaptation
arrays match the scientific run. The independent reader certifies both.
D's next question is motor-state selection in the accepted loco-09766 centre.

Source packet: `exchange/2026-09-09-2304-laneD-judge-packet.md`.

## R115: the centre fails motor-state retention

Nothing moved the model. The searched loco-09766 centre raised walking
power-cell firing from 9.5083 to 16.5000 Hz on spent seed 61524. Displacement
fell from 16.5170 to 11.8077 mm, retaining 71.4881%; cadence fell from 5.1818
to 4.8182 Hz. Both arms stood, but qualifying, cadence, displacement and
speed failed the existing gates. The flight-switch arm was not run.

The control reproduces R113's archived wi1 capture byte for byte. The
centre's three gain blocks were verified in the runtime arrays. Its live
canonical clip matches all body fields, common neural rows and input/gain
records, with 6,993,328 public bytes identical to the local master. No gain
or title claim follows; D returns to wi1. The next probe weakens the ten
existing IN06B042-to-IN13A013 inhibitory entries, with source and relay
activity read before attributing any effect. This is an assumed mechanism.

Source packet: `exchange/2026-09-10-0004-laneD-judge-packet.md`.

## R116: the proposed relay increase fails

Nothing moved the model. Halving ten inhibitory IN06B042-to-IN13A013 entries
after balancing lowered the four-cell relay mean from 27.55 to 26.30 Hz on
spent seed 61524. This failed the first mechanism gate, so the flight switch
was not run. Both arms stood. Walking power-cell output rose from 9.5083 to
13.6500 Hz; displacement fell from 16.5170 to 13.1490 mm, retaining 79.6090%.
The disinhibition route is closed at this setting. These are simulated values.

The intact control reproduces R113's body and common neural records exactly.
The native comparison finds only the ten declared half-strength entries.
The changed walking arm has a live canonical clip matching all 20 body fields,
855 common neural rows and input records. The public master has 7,238,257 bytes
and the same SHA256 as the local clip. No model gain or new physiological band
follows. D next tests whether reducing residual DNg100 output after the flight
command changes leg activity, with the command-cell response checked first.

Source packet: `exchange/2026-09-10-0039-laneD-judge-packet.md`.

## R117: the two-cell clamp loses standing

Nothing moved the model. The diagnostic clamp applied to two DNg100 cells
from 3 s ended at 171.54 degrees tilt on spent seed 61524. Both unchanged
controls reproduced R113 exactly and remained standing. The changed arm
failed the standing gate, so its later command, power and leg-state readings
do not locate a cause. The clamp route closes. These are simulated values.

The [changed switch arm](https://media.theuploadinglab.ai/2026-09-10-r117-wi1_dng100_clamp_flight_switch-61524.sha256-446dd4fd67463b8b6baa8a774e1b83fd12be06948c53900e46bc0af5a4fa773e.mp4)
replays all 20 body fields and 851 common neural rows exactly. The public
master has 6,073,750 bytes, identical to the local clip. Six simulated
seconds play in 16 seconds; the cadence label is censored. No gain or
physical-flight result follows. R118 next tests direct input to the
source-identified DNp10 landing pair with the same model in both command states.

Source packet: `exchange/2026-09-10-0111-laneD-judge-packet.md`.

## R118: the landing pulse does not extend all six legs

Nothing moved the model. On spent seed 61524, an assumed 100 Hz native-input
pulse to the two DNp10 cells from 4.0 to 4.3 s raised their simulated mean
firing from 58.33 to 111.67 Hz. The fly remained standing at 9.73 degrees
and retained 87.59 percent of displacement, but only three legs cleared
the registered reach increase. The right middle leg shortened by 0.1243
body lengths. The failed walking pulse cancelled the flight pulse, and
this direct-input route closes. Both unchanged controls reproduce R113 exactly.

The [changed walking arm](https://media.theuploadinglab.ai/2026-09-10-r118-wi1_dnp10_walk_pulse-61524.sha256-d8394d28c55f6ec97ac0c95083b4ce936c83f44ba093c87ccd13ebf0bf958e7f.mp4)
replays all 20 body fields and 853 common neural rows exactly. The public
master has 7,000,591 bytes, identical to the local clip. Six simulated
seconds play in 16 seconds. The new observer also records zero DNp10
spikes in the unchanged switched control's matching 300 ms window. This
is a ground-body command comparison, with no physical-flight or gain claim.
R119 next tests the source-identified ascending histamine projection to
leg motor neurons, after checking its command-state response.

Source packet: `exchange/2026-09-10-0149-laneD-judge-packet.md`.


## R119: the ascending leg-inhibition cells barely respond

Nothing moved the model. In the final two seconds on spent seed 61524,
the two AN27X004 cells fire at 0/0 Hz during walking and 2/0 Hz after the
DNg02 switch. Both miss the registered 5 Hz response increase, so neither
2x output-weight arm runs. The planned 49-entry intervention is untested.
The threshold is a diagnostic assumption, not a measured physiological band.

Both unchanged controls reproduce R113's twenty body arrays and 851 common
neural rows exactly and remain standing. The added observer records four
ascending histamine cells, giving 855 rows; AN27X008 fires at 0/2.5 Hz while
walking and 0.5/12.5 Hz after the switch. All parameters and native weights
remain unchanged. Two native Metal starts took 137.527 s including preparation.
The committed reader certifies the records; no changed behaviour requires
a new clip. No model gain or physical-flight result follows.

The next proposed mechanism strengthens the 28 existing DNg02-to-AN27X004
inputs at one assumed gain fixed across both states, with walking retention
and actual source, leg and joint responses checked before any attribution.

Source packet: `exchange/2026-09-10-0216-laneD-judge-packet.md`.


## R120: recruitment rises, but leg activity does not fall

Nothing moved the model. Doubling28 DNg02-to-AN27X004 inputs on spent 61524
raised the pair's switched firing from 2/0 to 23/10.5 Hz, while leg motor
firing rose from 10.724 to 11.541 Hz. Actual joint RMS changed from 5.704 to
5.682rad/s. The registered20 percent cuts both fail, so R120 is
DROP_R120_FLIGHT_LEG_RESPONSE and mechanism_signal is false. Walking
retained105.64 percent of displacement and all retention gates; walking
power remained9.59 Hz. All four arms stood. These are simulated values.

Both controls reproduce twenty body arrays and 855 common R119 neural rows
exactly. The expanded observer records1441 cells, including all 613 anatomical
inputs of AN27X004. Six native Metal starts comprise four science arms
and two presentations. The first preparation failure, with zero native
starts, remains separately preserved.

The [walking clip](https://media.theuploadinglab.ai/2026-09-10-r120-resumed-wi1_dng02_ahn_gain_walk-61524.sha256-6eb9de8068db85b66275a4ebccaf4a71b047e28637d99e8400a9cf387e618d0d.mp4)
and [switched clip](https://media.theuploadinglab.ai/2026-09-10-r120-resumed-wi1_dng02_ahn_gain_flight_switch-61524.sha256-8ab24fb52b72ef628b8b70b92d55996e2567f38ad016b8e0f9791d512fa10644.mp4) match all twenty body arrays,
all 1441 science neural rows, inputs, events and native weights exactly.
Public masters match locally verified bytes:6,843,997 and 6,702,559.
Each shows six simulated seconds over sixteen seconds. No gain or
physical-flight result follows from the ground-body comparison.

R122 finds that49 directly inhibited motor neurons carry only 9.4 percent
of switched leg spikes. That is a ceiling only if the other neurons stay
fixed; it does not refute recurrent effects. Their mean firing also rises.
R123 identifies173existing inhibitory entries onto 170excitatory leg-premotor
cells, with no anatomical output to wing-power neurons; their outputs reach
all six legs. R124 will test this broader output path at one fixed assumed
gain alongside R120's input recruitment. Source reach does not establish
inhibition of the closed-loop motor output.

Source packet: `exchange/2026-09-10-0314-laneD-judge-packet.md`.


## R124: the broader inhibitory route fails walking height

Nothing moved the model. Fixed 2x DNg02 recruitment and assumed 2x
AN27X004 inhibition of 170 leg-premotor cells left the fly standing,
but height reached 1.221144453 mm against the retained 1.220602326 ceiling.
The changed walking arm retained 99.586 percent of paired displacement.
Power fell from 9.591667 to 8.591667 Hz, short of the 50 percent target.
The combined flight switch was not run. R124 is closed.

The [walking clip](https://media.theuploadinglab.ai/2026-09-10-r124-wi1_ahn_premotor_gain_walk-61524.sha256-e16a3b45bc554668182b585b17bdaa9e2d569d182138bd38da6896e7707f2722.mp4)
reproduces all 20 body arrays and 1599 science neural rows exactly.
The public master matches 6,897,901 local bytes, SHA256
`e16a3b45bc554668182b585b17bdaa9e2d569d182138bd38da6896e7707f2722`.
There were three native science starts and one presentation on spent 61524.
No model gain or physical-flight result follows.

Source packet: `exchange/2026-09-10-0409-laneD-judge-packet.md`.


## R127: DNg02 threshold reduces power, below the selection gate

Nothing moved the model. Raising the assumed threshold from -45 to -35 mV
in exactly 38 DNg02 cells reduced simulated walking power from 9.508333
to 6.808333 Hz, a 28.396 percent cut against the 50 percent gate. All
walking retention gates passed; displacement retained 90.035 percent.
Late DNg02 output fell from 3.605263 to 0.197368 Hz. Neither flight arm
ran. R127 is closed.

The [walking clip](https://media.theuploadinglab.ai/2026-09-10-r127-wi1_dng02_threshold_m35_walk-61524.sha256-2cbd097e33f004de626fdd784191768b776e3f067241ae79f1e180e0260cbd6d.mp4)
reproduces 20 body arrays and 855 science neural rows exactly. Public
bytes match the 6,566,339-byte local master, SHA256
`2cbd097e33f004de626fdd784191768b776e3f067241ae79f1e180e0260cbd6d`.
Two science starts and one presentation used spent 61524. The original
walking control exactly matches archived wi1. No model gain follows.

Source packet: `exchange/2026-09-10-0445-laneD-judge-packet.md`.


## R129: recruiting the inhibitory relay raises walking power

Nothing moved the model. Assumed16x input from right DNg100 to left
IN03B065 recruited the relay from0 to7Hz late in walking, but power
rose from9.508333 to17.316667Hz. Displacement retained61.269percent,
below the80percent gate. Both walking arms stood; neither flight arm
ran. The single-connection diagnostic is closed.

The [walking clip](https://media.theuploadinglab.ai/2026-09-10-r129-wi1_in03b065_input16_walk-61524.sha256-2faa19cd182ebf452124f91229256004e7e2f765ae484c17fdebafe359ef2947.mp4)
reproduces20 body arrays and856 science neural rows exactly. The public
master matches7,028,582 bytes, SHA256
`2faa19cd182ebf452124f91229256004e7e2f765ae484c17fdebafe359ef2947`.
Two science starts and one presentation used spent61524. The original
control remained bit-identical to archived wi1. No model gain follows.

Source packet: `exchange/2026-09-10-0511-laneD-judge-packet.md`.
