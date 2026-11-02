---
id: 82
date: 2026-09-10
title: Correct neurons change the phase
tldr: A corrected neuron selection produces a phase shift that the earlier experiment missed. The largest tested split stands on eleven of twelve identities but loses speed; it does not replace the champion.
era: standard
commit: 867cc7c721ec and 609f8b8d3084, frozen lane C records
model: full 188508-neuron female CNS; deterministic Metal binary c40c5e1a
theta: wi1 with an assumed per-neuron adaptation-time vector; no shared default change
seeds: 50000-50011, previously spent
n: C-N255 four arms and C-N257 three arms, twelve identities per arm
wall: 476 seconds for C-N255; C-N257 completed before its summary error; judge readers execute no simulation
reproduce: lanes/judge/read_c255.py and lanes/judge/read_c257.py
media_note: Neural phase diagnostic with a speed loss; no integrated gain or new clip. Stored summaries were audited; raw trajectories were not saved.
---

Nothing moved the integrated model. Correcting the neuron selection changes
the scientific conclusion: different adaptation times across the two tripods
move their neural phase. The earlier experiment had changed the wrong cells.

The 100/400 ms split raises the paired tripod gap by 28.35 degrees but stands
7/12 against 10/12. A 50/400 ms split raises it by 41.43 degrees and stands
11/12; speed falls from 1.91 to 1.23 mm/s. Standing therefore varies
nonmonotonically across the fixed settings. This does not identify seed noise
as the cause. The parameter changes also alter the mean time constant.

Among the nine identities standing under both the 50/400 ms arm and control,
the phase gap rises 37.91 degrees, within-tripod scatter rises 11.25 degrees,
and speed falls 0.595 mm/s. These comparisons preserve the effect and expose
its costs. A phase shift is not yet alternating, coordinated walking.

C-N257 completed its engine runs before its summary code failed. The committed
reader reconstructs the verdict from the saved records; the judge verifies the
record identity and recomputes the comparisons. The stored scalar and spectral
summaries do not replace raw trajectories. Future gain claims need those
signals and footage. No title or animal acceptance band changes.

The other closed candidates retain their failures. D's combined intervention
fails cadence and height on its second spent identity; the later command-switch
candidate falls. E's six-cell redistribution also falls. Their committed
readers reproduce those outcomes. E's male spectrum varies between the saved
halves while the female's stays near 8 Hz, so the female clock is not assumed
for the male. Evidence: lanes/judge/2026-09-10-de-closeout-review.json.

The complete noon decision is exchange/2026-09-10-judge-noon-review.md. wi1 retains v3.11.

B's completed private-drive comparison also has a cost: the 4x arm that
changes phase sign stands 3/12 and fails its within-tripod/coherence checks.
Both repaired readers reproduce the complete archives. Robin asked for the
goals to pursue the strongest model outcomes; the rewrite and historical
evidence are recorded in exchange/2026-09-10-judge-goal-review.md.

H100 admission is now shared on main: all lanes together are limited to
8 CPUs and 41,250 MiB; GPU submission stays disabled pending Robin's
quarter-share decision. The default probe remains bit-identical. This adds
execution support, with no model gain or H100 runtime qualification:
exchange/2026-09-10-judge-h100-policy-review.md.

The fast-extensor comparison adds a second scoped signal. Its all-leg overlay
reduces the median imbalance between the tripods' foot-lift counts from
0.28990 to 0.10849 on twelve spent identities, improving each pair and
standing 12/12 against 10/12. Among the ten identities where both stand,
imbalance improves by 0.18095 and speed changes -0.1715 mm/s. These are
crossings of a foot-clearance threshold; balanced counts do not yet establish
balanced loads or alternating steps. The original hind-only rectification
hypothesis still fails. Three reader failure cases are returned to B for
repair, while the complete archived values reproduce independently.

The steering endpoints C-N264–N272 provide no qualifying heading control.
D's withdrawal-only body falls; its supported command contrast retains 96.70%
of walking speed and therefore does not stop. E's shorter motor synaptic
endpoint falls, and brain adaptation's small speed increase costs support,
lifting and heave. These outcomes close the named endpoints and leave the
lanes' independent research active. Complete decision and readers:
exchange/2026-09-10-1505-judge-build-review.md. The full-graph default probe
remains bit-identical, and wi1 retains v3.11.

C-N273 supplies a steering lead after those endpoints. DNa02 input passes the
registered scalar screen. Excluding the identity that falls under both inputs
leaves five paired standing identities, four positive yaw differences and a
median difference of 75.06 degrees. This is a BUILD lead; intact visual steering
and reliable direction control remain unproved.

B's FTi combination cancels: common-standing speed changes -0.414 mm/s,
below both single arms. A ThC partner preserves the lift-count effect but
also costs speed, so it establishes no general rule for combining gains.
The weighting profiles lose standing. A's recorded roll-pulse recovery fails
at both doses, while D's R159 stops at its height gate. D's separate component
read shows that co-contraction alone cannot change wi1's active stiffness at
fixed load. The complete decisions, scopes and reader receipts are in
exchange/2026-09-10-1534-judge-build-review.md. No integrated-model or title value changes.

Robin approved 15 reserved H100 GPU minutes per rolling hour for the fleet.
The shared admission tool now permits that allowance while keeping the
8-CPU and 41,250-MiB ceilings. Compute can complete the prepared 81548
runtime comparison; compilation alone still establishes no fly-runtime
qualification. The approval and evidence are recorded in
exchange/2026-09-10-judge-h100-approved.md.

The next walking endpoints retain their costs. B's joint-timing and damping
panels add no retained locomotion; C's alternative descending-cell inputs
lose standing or miss the directional bar. In the male fly, R767 reduces
non-foot force fraction from 0.0126 to 0.0056 but reduces the minimum
body-frame lift count from four to three and increases heave. A's stronger
roll feedback recovers two of five small-pulse releases against its required
three. The judge's frozen readers and source checks are in exchange/2026-09-10-1806-judge-build-review.md; these
records change no shared model or title.

Robin next asked for a direct foreleg-bristle test of stepping or turning
away, and for one agent to develop a working compass before integration.
D owns the sensory-to-body comparison. Robin then asked for the compass and
navigation circuit connected to the complete VNC and body before restoring
the remaining brain. A separate specialist owns that staged prototype. Its
heading, steering and body response are read separately; reintegration keeps
the learned parameters and tests retained behavior. Reduced-system success
remains a partial result. Robin also suspended
all H100 use; the local compatibility probe remains bit-identical.
