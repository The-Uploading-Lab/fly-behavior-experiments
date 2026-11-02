---
id: 81
date: 2026-09-10
title: Two seconds of flight with silenced motor inputs
tldr: A partial neural flight control survives on the shared runtime and repeats its fixed survival result on a second seed. Four free-flight windows complete on each identity; the tighter whole-window attitude hold passes only one release on the second. Silenced inputs and prescribed leg pose keep this a prototype; the integrated fly and its title are unchanged.
era: standard
commit: cc324e6fd127bef9e1be18ddccaf174a67cbced7 and 9970be442; independent main replay followed by a complete licensed-archive read
model: full 188508-neuron female CNS; deterministic Metal source 48ebcded, binary c40c5e1a; FlyBody flight rig with prescribed leg pose
theta: results-wi1-theta.json plus the declared FL5n phase interface and silenced b1 background; partial control
seeds: 203760 previously spent; fixed replication on licensed 203762, now spent; 203763 remains sealed
n: five release times per identity; four decisive arms replayed on main for 203760 and all twenty licensed arms independently read for 203762
wall: judge replay 109.83 s; second-seed lane batch 966.7 s; archive readers execute no simulation steps
video_receipt: publication-receipts/fl5v_top111_connectome_final_203760.sha256-5352a7ea7516435a0a690adc163f99c2dcbfcfce7a6b0d3fb4829f8facaa467e.publication.json, publication-receipts/fl5n_flight_connectome.sha256-b3c520d136bcd2e0949f0c83e1e90a8a53c159f6a909246631bea148a4e0e38a.publication.json, publication-receipts/fl5q_flight_connectome.sha256-6fe220816faa0ec0a9a59117ce6071e6148bea4846d851ab3267e0bf9d8f48c7.publication.json, publication-receipts/fl5e_phase_S2_M1_tau5_Kf2.6_Ka0.65_bilat_kp50_rel4.sha256-09ed0bd0950ad5b5114b09f3f90f4bbd36f8735eb56dd61040c3e06c5bc2f17b.publication.json, publication-receipts/fl5q_203762_phase_S2_M1_tau5_Kf2.6_Ka0.65_bilat_kp50_rel3.25.sha256-4d310cec307bc33f5b2d40c508b26eda1b98a8e38f28f33a15ff06e6556e2fbf.publication.json
reproduce: lanes/judge/replay_fl5n_flight.py, lanes/judge/2026-09-10-fl5n-main-replay.json, lanes/judge/read_fl5n_flight.py, lanes/judge/read_fl5q_replication.py, lanes/judge/2026-09-10-fl5q-replication-review.json, lanes/judge/read_fl5v_control.py, lanes/judge/2026-09-10-fl5v-review/review.json
---

Nothing moved the integrated model. A partial flight control now reproduces
on main: it completes the two-second free-flight window while the count-code
control tumbles after 0.882 s. A second control has the same silenced inputs
but no phase readout and tumbles after 0.935 s. A control whose volleys carry
no body information tumbles after 0.327 s. The four replayed trajectories and
motor traces agree exactly with the lane's recorded flights.

The controller converts body rotation and a hypothesized attitude signal into
haltere spike timing, then converts the two b1 motor neurons' spike timing
into wing steering. It suppresses 591 non-haltere inputs to those motor
neurons, sets 439 afferent delays to 1 ms, and fixes the flight leg pose by
program. These are declared parts of the prototype. Several interface
settings change between the phase and count controls, so the comparison
establishes the complete prototype's effect rather than each part's necessity.

Survival and attitude are different outcomes. Across five release times on
this one seed, the bilateral phase arm passes the registered survival margin
at all five and completes four free-flight windows. It holds the diagnostic
10-degree attitude band in none. The unilateral arm passes the survival
comparison four times and holds the band once. Release-time variation does
not supply an independent seed.

The displayed flight is the bilateral arm released at 4 s, shown at one sixth
speed. Its movie hash matches the lane receipt and the public object. It gains
224 mm of altitude during the free window and ends outside the tighter pitch
band. The legs are prescribed, and a separate walking body supplies the neural
harness; this does not yet establish one integrated walking and flying model.

The next replication keeps the candidate fixed and uses complete matched
controls on one licensed identity. Restoring its non-haltere input and driving
its leg pose through neurons remain the route toward a shared-model change.
wi1 retains v3.11.

**Input-dose follow-up, 2026-09-10 12:02 EEST.** FL5p's complete tick archive reproduces
the survival comparison at five of five releases with one quarter of b1's
background restored, and three of five with half restored. Whole free windows
fall from four at c=0 to two and zero. This remains the same spent identity
and the same prescribed leg pose. The next independent comparison is fixed;
its seed stays sealed until the execution guard certifies complete traces.

At 2026-09-10 12:18 EEST, the fixed twenty-arm replication is running on licensed
seed 203762 after the full-trace guard and fourteen tests passed. This is a
license and an active run, not a second-seed result. 203763 stays sealed.

**Replication complete, 2026-09-10.** The fixed candidate passes the survival
margin against all three matched controls at every release on licensed seed
203762. Its smallest margin is 1.351 seconds. Four whole windows complete;
the earliest release tumbles after 2.003 seconds of a three-second window.
The fixed candidate transfers, but the second comparison adds the matched S0
control and requires a margin against all three controls, so the two protocols
are not identical. No tuning on 203762 is licensed.

The second film shows the 3.25-second release, the selected one of five that
holds the whole-window 10-degree attitude band. It climbs about 101 mm and
rotates in yaw. This is survival with a limited attitude hold, not hover or
landing. Playback is one sixth speed; the public object matches the original
same-run movie. The silenced input, assumed sensor and prescribed-pose limits
above remain.

The replication packet overstates walking retention. Its recorded
`standing_v32` passes 19/20 arms, including all candidate arms. The quiet
control released at 3.75 seconds falls. The distinct `standing` field passes
8/20 arms, including only one candidate arm. The judge corrects the recorded
fields without granting a walking retention claim. Complete decision:
exchange/2026-09-10-judge-flight-replication-review.md. wi1 retains v3.11.

**Video correction, 2026-09-10.** The first two movies now include the
synchronized connectome, colored by neural layer, with brightness showing
firing in the preceding 20 ms. Full-network visualization replays reproduce
every saved flight and motor value and the original movie clocks. The
replication caption now carries its actual seed, 203762. The original movies
remain below as source footage. These changes alter the presentation; the
partial-control result and its limits above are unchanged.
**FL5v: a smaller lesion, 2026-09-10 15:05 EEST.** Disabling 160 of the
591 background entries onto b1 preserves the survival advantage at all five
release times on the previously used seed 203760. All twenty flight traces
are complete, and the five count-control arrays reproduce FL5n exactly.
Four whole free windows complete, as with the original lesion on that same
seed; none holds the tighter attitude band for two seconds. The smallest
advantage over the three controls is 0.5798 seconds.

The 111 selected source cells supplied 95.05% of the reference background
drive. The other 431 entries are unmasked, including one that already had
zero weight. This is a smaller intervention set, with most of the delivered
background still suppressed. It does not establish an intact flight network
or a population that must always be silenced. The separate walking body
stands on four of five candidate arms under its v3.2 flag and three under
its other standing flag; no walking retention claim follows.

The new film shows the 3.25-second release with the original recorded neural
activity beside it, colored by layer. The full retained point cloud fits in
the panel, and the caption states the recorded seed and the input lesion.
The public movie matches the publication hash. No simulation was repeated
to compose it. The top111 control is accepted for further BUILD work;
203763 is reserved for a fixed replication and remains sealed pending its
frozen consumer. Decision: exchange/2026-09-10-1505-judge-build-review.md.

**Replication, 2026-09-11 10:28 EEST.** Lane A ran the frozen partial control
once on the reserved seed 203763 under the judge's license (consumer 25e24e9f,
runner afa7841b, manifest 2f1eceaf; all three hashes match at the execution
commit 15aab4540). Four of five release times pass the pre-registered three-of-
five bar; release 3.0 s fails (0.552 s against the strongest control's 1.044 s);
the passing margins over the strongest control are 0.651 to 1.555 s. One flight
lasts the whole 2.25 s window (release 3.75 s) against four of five on the
selection seed, and none stays within the tighter 10-degree band. The judge
reproduced the decision from the committed reader in the lane's checkout,
without an engine: REPLICATED, 4 of 5, the reader matches the recorded
decision, no model gain claimed. The seam of silenced background inputs to b1
now carries the two-second flight on two seeds. Scope unchanged: unperturbed
flight, prescribed legs, a hypothesised attitude input; not disturbance
recovery and not the walking body. Seed 203763 is spent.
