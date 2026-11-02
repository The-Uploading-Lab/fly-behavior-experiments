---
id: 80
date: 2026-09-10
title: The champion skates, leans on one tripod, and its one map asymmetry is six transferred cells
tldr: Zero-engine reads of the champion's own records show that two thirds or more of its forward travel is its stance feet sliding, that it stands on one tripod sixteen times more often than the other, and that the only left-right asymmetry in the motor map it walks with is a six-cell male-to-female transfer routed four right and two left. Removing that transfer drops the fly on one of two spent seeds, so it holds the fly up. Underneath all of it, the champion runs one body-wide neural oscillation at 7.8 Hz and delivers one phase to all six legs; twenty-three knobs, including a 180-arm bound-edge screen, a half-period delay on one inhibitory hemilineage and a hand-written reciprocal inhibition, leave that one phase in place or dissolve the rhythm, the clock runs on with the body cut out of the loop, a certified six-phase input through the fly's own joints does not move it, the direct afferent line at parity is heard and answered with coherence rather than phase, the shared adaptation constant moves the clock's period 15 ms over its whole legal range and its phase not at all, and the wiring itself is neutral between synchrony and alternation. A judge ruling that a joint was unpowered was wrong and is withdrawn. The first title battery of the night, on the adaptation constant's 100 ms rung, stood 23 of 24 and was held by the card: cadence and front knee range regress.
era: standard
commit: the commit this entry lands in, on main
theta: results-wi1-theta.json (mn_gains size, signs measured, mn_transfer t2_remotor_manc6), unchanged; no parameter moved
seeds: B spent 81547 and 81548 (eight diagnostic runs); C's recorded plain seeds 50000-50023 (20 standing arms, read from disk) and spent 50000-50011 re-run for the 60-arm clock ladder and three 48-arm ladders; no fresh seed spent; 50619 and 108524 sealed
n: 20 standing plain arms read under two contact instruments; 61 standing arms for the yaw fit; 391 mapped motor rows and 34 femur reductor cells for the census; 8 engine runs for B's probes; 60 arms for C's clock ladder; 360 leg-pair phase measurements; 48 arms each for the inhibition, delay and seam ladders; 180 arms for B's bound-edge screen; 48 arms for C's hemilineage timing ladder; 36 and 60 arms for C's six-input scaffolds; 48 arms for B's reciprocal inhibition; 72 arms for C's hand-built tapes; 108 arms for C's coupling dose; 84 arms for B's labelled line at parity; 48 arms for C's adaptation ladder
wall: zero engine seconds for every lane-C read except the clock ladder (686 s) and the inhibition and delay ladders (542 s, 560 s and 607 s) and B's screen (about 2,450 s); about a minute per run for B's eight probe runs on the deterministic Metal engine
media_note: no film; every read here is zero-engine over records already on disk or a diagnostic on spent seeds with no gain claim, so no clip was made
reproduce: vision/read_c_n221_foot_ground_truth.py, vision/read_c_n221b_tripod_and_slip.py, vision/read_c_n220_wiring_side_census.py, vision/read_c_champion_motor_map.py, vision/read_c_n228_assignment_audit.py, vision/run_c_n227_mechanical_or_neural.py, vision/read_c_n229_cross_leg_phase.py, vision/run_c_n231_inhibition_phase.py, vision/run_c_n232_delay_phase.py, vision/run_c_n242_hemilineage_timing.py, vision/run_c_n244_six_inputs.py, vision/run_c_n245_open_loop_inputs.py, vision/run_c_n246_built_inputs.py, vision/run_c_n247_coupling_dose.py, lanes/B/b_afferent_pathway_census.py, lanes/B/b_reciprocal_inhibition_probe.py, lanes/B/b_premotor_sharing_census.py, vision/read_c_n233_alternation_ceiling.py, vision/read_c_n234_half_centre.py, vision/read_c_n235_contrast_reachability.py, vision/read_c_n236_two_hop.py, vision/run_c_n237_seam_contrast.py, vision/read_c_n238_afferent_census.py, vision/read_c_n240_drivability.py, vision/read_c_n241_frequency_patterns.py, lanes/B/b_network_phase_screen.py, lanes/E/read_male_clock_r673.py, lanes/B/b_motor_norm_probe.py, lanes/B/b_mn_transfer_absent_probe.py, lanes/B/b_mn_transfer_symmetric_probe.py, lanes/B/2026-09-10-motor-norm-81547/result.json, lanes/B/2026-09-10-motor-norm-81548/result.json, exchange/2026-09-10-0046-laneC-judge-packet.md, exchange/2026-09-10-0145-laneC-judge-packet.md, exchange/2026-09-10-0110-laneB-judge-packet.md, exchange/2026-09-10-0135-laneC-judge-packet.md, exchange/2026-09-10-0137-laneC-judge-packet.md, exchange/2026-09-10-0207-laneC-judge-packet.md, lanes/B/b_labelled_line_probe.py, lanes/B/b_labelled_line_read.py, vision/run_c_n248_clock_timescale.py, vision/read_c_n248_paired.py, exchange/2026-09-10-0521-laneB-labelled-line-result.md, exchange/2026-09-10-0519-laneC-judge-packet.md, exchange/2026-09-10-0522-laneC-champion-candidate-tau-w-100.md
---

**Question.** The shared fly walks at 1.9 to 3.0 mm/s and walks crooked. Two
lanes had measured a lopsidedness: the left legs carry more stance load on 19
of 20 seeds, and the front footprint sits 0.2 mm off centre on 24 of 24. Where
is it, and is it in the body, the map, the wiring or the dynamics?

**What was wrong first.** At 23:5x the judge ruled that the right middle leg's
thorax-coxa pitch joint had no motor neuron while the left had one, and told
Robin so at 00:25. That was a muscle-table fact read as a joint fact. On the
matrix the fly actually walks with, built with size gains and the T2 transfer
and then normalised, that joint carries 17 motor neurons at full weight on
both sides. Lane C found the error, lane B confirmed it against the delivered
matrix, and the ruling was withdrawn on main at 00:3x and corrected to Robin
at 01:1x.

**What the map asymmetry actually is.** BANC has 17 femur reductor cells on
each side, all mapped, with left-right input within 3.7 per cent: the right
side is neither thin in cells nor starved of drive. The one unequal mirrored
axis is the middle leg's coxa-trochanter roll, delivered at 0.258 on the right
against 1.0 on the left. Lane C's routing census found why: the front and hind
legs put every femur reductor on that roll axis on both sides, but the middle
legs route some to the thorax-coxa axes, four of six on the right and two of
six on the left. The judge traced the split to its source. A transfer set,
`t2_remotor_manc6`, re-maps six middle-leg femur reductors whose BANC
`manc_cell_type` matches the MANC remotor type onto the pleural remotor and
abductor muscle; the list is four right cells and two left, labelled
TRANSFERRED under the male-transfer rule of 2026-09-04. The normaliser's
`max(colsum, 1.0)` floor then leaves the right roll column under-normalised
and divides the left one down, a 3.87 times delivered-torque difference. The
floor transmits the asymmetry; it does not create it.

**Is it load-bearing?** Lane B lifted the right column to the left's delivered
weight in memory, without editing any table, on the two spent seeds. The fly
stands on both. On 81547 its tilt improves from 13.5 to 11.7 degrees; on 81548
it worsens from 9.2 to 39.9 and travels less. Lane B's earlier run had already
shown that every arm removing the middle-leg handedness falls on one of the
two seeds. The champion's walking is tuned around the asymmetry. Whether to
keep the transfer as annotated, revert it or symmetrise it is a mapping
decision and goes to Robin at noon.

**The transfer, absent (B, 01:43).** Lane B then ran the champion's theta with
the `mn_transfer` key removed, the harness's own bit-identical default path,
against the plain arm on the same two spent seeds, in memory, no table edited.
On 81547 the fly stands either way and is slightly straighter without the
transfer: tilt 13.5 to 11.3 degrees, tripod duty gap 0.362 to 0.306, the
middle-leg footprint offset 0.015 to 0.006. On 81548 it falls without it: tilt
9.2 to 172.9 degrees, slip ratio 0.41 to 1.23. Lane C's assignment audit of
the same night (N228, zero engine, both map extractions) says why. BANC
natively gives the two middle legs six femur reductor cells each at equal
weight; the transfer takes two off the left and four off the right, and it
exists because the right middle leg natively has no pleural remotor and
abductor motor neuron at all (the left has one). Reverting it empties that
muscle again, and the fly that walks on 81548 needs it. So reverting is
refuted as a shared change; keeping it leaves a 3.9 times torque deficit on
one axis; lane B measured the third option, two cells from each side, in memory at
01:57: the fly stands on both seeds (tilt 13.5 to 14.7 and 9.2 to 11.5
degrees, tripod duty gap 0.36 to 0.30 and 0.35 to 0.36, travel 16.0 to 12.9
and 10.3 to 11.8 mm), viable and not better, so the noon choice is between
the annotated 4-2 and a symmetric 2-2 that buys no measured gain. Eleven other side
asymmetries in the map survive the transfer's removal and look like
reconstruction noise (tarsus depressor 3 against 5 on the front legs and 5
against 3 on the hind).

**One oscillator (C, 01:3x).** Asking whether any cell of the premotor layer
is rhythmic, lane C found that the layer's output is: the champion runs a
single body-wide oscillation at 7.8 Hz. Sixty arms on spent seeds 50000 to
50011 (five per seed, paired on seeds standing under both conditions) gave the
clock's law. Halving the synaptic delay raises the frequency by 12.8 per cent
and a 12 ms delay lowers it by 15.0 per cent, while halving or doubling the
body's joint stiffness moves it by 1.25 and 0.00 per cent where a mechanical
resonance would move it by 29 and 41; period = 103.3 ms + 3.70 x delay, r
0.995 over a 3.85-fold range. It is a neural clock, and the four arms that
fall keep the rhythm in their leg drives with the body's bounce gone, so lane
C withdrew its own earlier story that the body bounces and the network
follows. The clock reaches the coxa-trochanter pitch commands of all six legs
within a median 18.7 degrees of one phase (360 leg-pair measurements, median
coherence 0.829); across the two tripods the difference is 25.9 degrees, one
seventh of the 180 a walk needs. Step frequency, 5.4 to 5.5 Hz, does not move
under any knob. One phase delivered to six levators predicts, without further
assumption, legs that cannot alternate, a body that heaves at the clock's
frequency, feet that slip because all six push together, a walk that will not
go straight, a descending command that can only scale the rhythm, and vision
that moves no gait row. Lane C claims none of those as results; each is a
cheap prediction. Its survivorship re-read of every visual steering arm on
disk (ten tests, seven bodies, Holm-corrected) found no stable yaw sign
anywhere. The fleet's question is now what would break one oscillation into
six.

**Fifteen knobs (C, 02:0x).** Lane C then asked the two things the ledger
still allowed. Premotor inhibition from half to four times its value leaves
the across-tripod phase at 26.6, 25.7, 21.4 and 18.6 degrees: more inhibition
tightens the lock. Coupling delay across its whole bound, 0.5 to 12 ms, leaves
the coherent rungs at 29.3, 25.7 and 22.5 degrees, falling where a
delay-driven coupling would rise, and confirms the clock's law out of sample
within 1.4 ms at every rung. At 0.5 ms the population stops cohering at all
(pair coherence 0.20, within-tripod scatter 116 degrees): the fly stays up on
11 of 12 seeds and travels 2.8 mm instead of 11.5. Delay is what holds the
population together, not what could split it; by the clock's own law a
delay-driven phase could never pass 97 degrees. With lane A's thirteen
strength knobs of 2026-09-04 that is fifteen knobs and no per-leg phase.

**The wiring's own bound (C, 02:1x to 02:3x).** Lane C then read the graph
the fly is built from, zero engine, each read registered before its first
number. Across-tripod inhibition onto the coxa-trochanter pitch motor
neurons is 1.57 times the within-tripod value at one synapse, but a leg's
own inhibition is 2.5 to 3.1 times the sum from all five other legs, and no
class gain in the ledger moves the one-synapse contrast by more than 0.024.
At two synapses the contrast is movable, 1.05 to 2.61 against 1.50 as the
champion stands, short of a half-centre, and the between-leg two-synapse
coupling is net excitatory, a synchronising sign. Running that best case on
the same twelve spent seeds (intersegmental inhibition at 4 and 16 times,
commissural inhibition halved) leaves the across-tripod phase at 23 to 30
degrees against 25.7 plain; the strongest arm loosens the lock only by
decohering it, as lane A had seen on another body. What would break one
oscillation into six is not in the parameter space or in a class gain on the
wiring; it is a structural decision, and it goes to Robin at noon.

**The last screen, and a graph that does not care (B and C, 02:5x).** Lane B
then ran the last inside-the-ledger check it had registered: every remaining
network coordinate at both of its bound edges, fifteen arms on the same twelve
spent seeds, 180 runs, read only as reached, shifted or null. Null: nothing
reaches the 120 to 240 degrees a tripod needs; four edges drop the fly on all
twelve seeds; one, synaptic scale halved, stands on all twelve at 93 degrees
across tripods and waits for its coherence to be read before anyone calls it
a phase. Lane C, meanwhile, read the graph a third way, zero engine: a leg's
premotor pool hears its own leg's afferents 2.2 to 4.0 times more than all
five others, but they are 4 to 6 per cent of its input; and the linear gain
from the premotor layer to the six leg pools is neutral between synchrony and
tripod alternation at every frequency (ratio 1.007, six singular modes, one
per leg). The loaded graph couples the six pools as near-independent channels
fed one common drive; synchrony is what common drive buys, and the lock is a
choice of the dynamics, not a missing wire. Lane E's read of the male build
with the same instrument found no coherent clock there at all (four records,
two seeds). What would give six legs six phases is a structural or engine
decision, and it goes to Robin at noon with lane C's brief.

**Timing, and a correction (B and C, 02:5x to 03:0x).** B's shifted arm was read
the way C's 0.5 ms arm was: at half synaptic weight the within-tripod scatter
(109 degrees) exceeds the across (93), pair coherence falls from 0.865 to 0.524,
and both sit near the 90 degrees that random phases give. That is a fly that
has stopped agreeing on a rhythm, not one that has split into two tripods, and
the columns that say so were written per run before the question was asked.
B's connectome census then found the six CTr pools sharing premotor input
equally within and across tripods, with the strongest sharing between same-side
neighbours (4.56 times the rest, partner sign not read): the pairs a tripod
holds in antiphase carry the most common drive. C found that the champion
already runs a per-type delay map (`length_by_type`, 63 cell types at 1.0 ms
over 413 cells), so the night's delay law is a law in the mean of a distributed
delay, and the per-hemilineage timing C's brief had asked Robin for costs a
search, not an engine change. N242 ran that search on the spent seeds: the
437-cell 13A inhibitory hemilineage delayed by a quarter and by half of the
128 ms period. The half-period arm stood 6 of 12 with across 41.3, within 31.6
and coherence 0.424, the twentieth knob and the same dissolution; the
completion control alone (all 73 13A types at 1.0 ms) kept the across median
at 26.3 but tripled the within scatter, 9.3 to 29.3, so the offset arms are
read against it and not against the champion. PARTIAL by its registered rule;
no arm alternated.

**Six inputs, and a clock that does not need the body (C and B, 03:2x to
03:5x).** The open question was whether the six pools could hold six phases if
something gave them six. C tried to deliver them through the fly's own
afferents by moving the legs in tripod with a body scaffold at the pools' own
7.8 Hz (N244), and then with the network's output to the muscles cut so that
only the afferent stream reached the pools (N245). Neither delivered: at gains
0.5 to 8 the scaffold never produced a tripod pattern at the feet (tip
amplitude 0.02 to 0.05 mm against the champion's 0.26), both runs read
VOID_NO_INPUT by their registered gate, and no neural claim was made. The
control that mattered came free: with the motor output cut and no scaffold the
fly lies still and the clock runs on unchanged, across-tripod 23.7 degrees at
coherence 0.865. The lock is intrinsic to the network; the body is not in the
loop that holds it. B then wrote the reciprocal inhibition by hand, scaling the
synapses of each leg's private inhibitory cells that the same-side neighbour
drives (eight seams of about two thousand edges) by two, six and eighteen. The
across number rose to 44, 33 and 59 degrees; the within number rose with it to
40, 81 and 62 and coherence fell to 0.54, 0.44 and 0.53. The twenty-first knob,
and the same dissolution. What remains is not a knob: a six-phase input channel
the fly does not have tonight, which is an engine and mapping decision.

**The body is the channel, and the pools are deaf to it (C-N246, 04:06).** C
then built the six inputs by hand. The ledgered playback path drives the joints
from a flexor/extensor tape while the network runs, receives every afferent and
cannot reach the muscles. The tape was the champion's own drive statistics with
a 7.8 Hz oscillation added, in tripod phase or in synchrony, at one and four
standard deviations. Delivery was certified on the channels the afferents carry:
joint angle across the tripods 180 degrees, within 0, load across 179 and 176,
on twelve seeds of twelve, and the synchronous tapes read across 0. The pools
did not move. Tripod tapes gave neural across 24.5 and 18.0 degrees at
coherence 0.85 and 0.79; their matched synchronous controls gave 23.0 and 23.6.
POOLS_DEAF by the registered rule, with the instrument exact on plain. The
defect is now located rather than described: the afferent-to-pool coupling in
this fly is too weak to re-phase the pools, or their mutual lock too strong,
which is the same shape A read in flight the same hour, a pitch loop twenty
times under-gained and a roll loop open. The next question is the dose.

**The dose, and the census that predicted it (B 04:17, C-N247 04:57).** B
sized the channel first, without an engine: each leg's own afferents are 1.7
to 3.6 per cent of its CTr pool's input by synapse mass and outweigh the other
five legs' afferents onto that pool ten to twenty-nine times, a labelled line
rather than a wash; the direct afferent-to-motor route is small and 92 to 100
per cent excitatory, while the two-hop route reaches three quarters of each
pool's input through premotor cells that are half inhibitory. A channel like
that can set how hard a pool fires, not when. C then multiplied every synapse
leaving the 5,200 leg afferent rows by four, sixteen and sixty-four, with the
tripod tape and its synchronous twin at each gain. Four changed nothing that
followed. Sixteen and sixty-four dissolved the clock in tripod and synchronous
arms alike: within-tripod scatter 79 to 139 degrees and coherence 0.26 to 0.59
against the champion's 10 degrees and 0.87, and at sixty-four the tripod input
left the across number 19 degrees below its synchronous twin. Registered
verdict PARTIAL, since no gain followed; the mechanism is gain-induced
decoherence that does not care about the pattern, the twenty-second knob. More
afferent drive breaks the lock instead of steering it. What is left in the
engine is B's labelled line, the excitatory direct route alone at parity; what
is left for noon is the mechanism question, since gains, delays, wiring and
inputs have all been tried.

**The line alone, at parity (B, 05:2x).** B then gained only the direct
afferent-to-motor edges of the three engine-driven classes, per leg, until each
leg's own afferents were half of its pool's input (gains of 27 to 58 by leg) and
then twice that, with C's certified tripod tape and its synchronous twin through
the same path; 84 runs on twelve spent seeds. The ungained control reproduced
C's deaf pools (24.5 against 23.0 degrees across). At parity the pools heard the
pattern and answered backwards: the tripod input tightened the whole-body lock,
across 16 degrees, within 18, coherence 0.63, while the synchronous input
loosened it, 65 and 40 degrees at coherence 0.82; at twice parity the tripod arm
decohered (47, 53, 0.36). B's registered rule stamps it DECOHERENCE and B adopted
the judge's description beside it: heard, not steered, the twenty-third knob.
The pools change how tightly they lock, never which phase each leg gets.

**The one knob that moves the clock's rate (C-N248, 05:19).** The axis left
untried was the cells' own adaptation constant, tau_w, one number shared by
every neuron. C ran it at 20, 100, 271 (the champion) and 400 ms on twelve
spent seeds, 48 runs. The period followed it on 12 of 12 seeds, 116.3 to 131.6
ms over the whole legal range (slope 0.037 ms per ms, sign p 0.0005), and the
six-leg phase structure did not move anywhere. Registered PARTIAL: a 15 ms
spread against a 40 ms clock bar, a weak frequency knob and no phase knob. The
100 ms rung stood 12 of 12 where the champion stood 10 of 12, rescuing seeds
50006 and 50008, with within-tripod scatter 7.4 against 10.2 degrees and
coherence 0.892 against 0.865; paired, only the period change survives Holm.
C filed it as a champion candidate with one key changed and no claim, and the
judge took it to the v3.11 title battery on the holder's own block.

**The battery (judge, 05:33 to 05:44).** The judge ran the candidate through
the v3.11 title match on the holder's own paired block, 30532 to 30555, with a
fresh card on 30568 to 30579. It stood 23 of 24 where wi1 stands 19, the best
standing of any arm ever put to the block, and it was held by the card: its
step frequency fell to 4.91 Hz against the animal band's 5.0 floor (4 of 11
card runs in band against wi1's 10 of 12) and its front knee swept 68.5
degrees against a floor of 81.2 (0 of 11 against 5 of 12), with no row
improved; open measured-data conflicts 9 against 5, qualifying runs 8 against
17. A longer adaptation buys standing with slower, shorter stepping. wi1 keeps
the title, and the card seeds are spent.

**What the walking actually is.** With the joint-to-body link checked (a
no-slip fit over the planted feet predicts the body's angular velocity at
slope 0.928, so the legs do drive the yaw), lane C decomposed every stance
into how far the foot slid through the world and how far the leg stepped. On
20 of 20 standing plain arms, under two contact instruments that agree on 86.7
per cent of ticks, 65.7 to 84.8 per cent of the champion's forward travel is
its loaded feet sliding. Tarsi move at 3.4 to 6.5 mm/s in stance against 25
mm/s in swing, so the detector is finding stance. The same records show the
fly stands on the tripod left-front, right-middle, left-hind alone 57.3 per
cent of the time and on the other tripod alone 3.6 per cent, a duty gap of
0.358 on 20 of 20 arms. The left-heavy stance load was the left-side
projection of that: the sign alternates front, middle, hind. The lean is in
13 of 15 bodies on disk and reverses only under a bilaterally symmetric knee
sign flip, so it is a broken-symmetry state of the dynamics, not a lopsided
map. Vision moves posture and never stepping; the descending command into the
leg pools changes amplitude and never timing; an 8,220-cell premotor layer
carries 87.6 per cent of motor input and is reached by 95 per cent of
descending output. Nothing in this model retimes stance.

**What it means for the score.** Speed alone no longer says the fly walks.
The judge's v3.12 proposal adds travel earned by stepping and the tripod duty
gap as title rows, and puts two questions to Robin at noon: the six-cell
transfer, and whether a whole-brain fly whose premotor layer carries most of
the motor input should be expected to make its own gait rhythm.

**Not claimed.** No behavioural cause of the crooked walk is shown by a wiring
census. The predictions of the one-oscillator diagnosis are predictions. No
parameter moved, no fresh seed was spent, no gain is claimed, and wi1 keeps
the title.

**Correction, 2026-09-10 11:14 EEST.** B found that C-N249 changed the wrong neurons.
The judge reconstructed the production motor map and ID joins: none of its
391 ordinal motor positions was a leg motor neuron. The corrected 10,760-row
membership matches B's IDs exactly. Its original 26,915-row run cannot reject
per-leg detuning, determine coupling strength or show that the connectome lacks
a phase mechanism. Those conclusions are withdrawn; the corrected experiment
stays open. This corrects an instrument and does not improve the fly.
The battery paragraph also used "longer" incorrectly: 100 ms is shorter than
wi1's 270.98 ms. Every title number and the verdict stand. Evidence:
`lanes/judge/2026-09-10-1114-n249-scope.json`; reader `lanes/judge/read_n249_scope.py`.

The N247 audit also reproduces: 1,035 of its 5,200 source neurons enter the
default external-rate callback, contributing 28.84% of its synapses. That
fraction does not describe spike traffic; neurons outside the callback still
receive recurrent input. No behavioural claim or acceptance band changes.

**Further correction, 2026-09-10 11:32 EEST.** C traced the same wrong motor indices
into N250, N252 and N253. The judge reproduces 92 left and 103 right neurons
in the supposed left group, and 98 of each in the supposed right group.
Neither contains a leg motor neuron. Their steering-law and heading-authority
interpretations are withdrawn. Correctly addressed tests remain open; the
90-degree bound of a two-oscillator approximation does not settle them.

B's direct shared-input lesion leaves median motor coherence at 0.824 against
0.865 in the control, with amplitude at 91% and peak share at 107%. That result
applies to the declared shared edges into 110 motor neurons. The next-layer
lesion loses rhythm before its phase change can be interpreted. A zero-input
regression exposes a reader bug, but all completed records have finite positive
amplitude and peak share, and their verdicts reproduce.
