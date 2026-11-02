---
id: 98
date: 2026-09-15
title: The front legs step together because the brain says so, and the row that said otherwise was reading the camera
tldr: The judge seat died at 22:15 and the whole lab slept from 02:37 to 10:51, so the night's forty-eight packets were graded in one morning pass. Lane B found that the champion's two front legs lift together because the descending drive from the brain holds them together, and that the one body whose front feet alternate on the cord's own motor output is the one with that drive at zero and the cord's crossing excitation silenced, walking at 62 per cent of its speed. On the way it measured that the card row the judge counts for leg phase reads the tip's position in a frame that turns with the body: in 99 of the 130 archive runs where the row says the front legs alternate, the cord's own drive says they move together. Lane E stood the native male up on one seed by two different routes that do not add. Lane A rolled the free body and the halteres delivered 7 Hz of side difference where last night's hand-driven 50 Hz was needed. Nothing made the fly better; the champion is unchanged.
era: standard
commit: the commit this entry lands in, on main; the verdicts at a2c63dc00 with their stamps corrected at cd7edd80a and 67383913b; lane B's packets 22:46, 23:30, 00:09, 01:03, 01:40 and its 02:33 commit 116676a3f on lane-b; lane E's packets 22:08 to 02:24 on lane-e; lane A's FL5dj packet 22:58 on lane-a
theta: the champion wm1 unchanged, results-wm1-theta.json sha256_16 3a21ddb794be23cb. No parameter changed. The bodies in the clips are arms of experiments, not nominees
seeds: lane B's fresh seeds 81870 to 81878 (E19, nine paired runs, all standing); lane E's spent seed 108523 only; lane A's fresh seeds 203868 to 203870 (FL5dj)
n: B, 36 whole-CNS Metal runs for E19 and then zero-seed re-reads of 1,797 standing archive runs; E, about a dozen single-seed runs on the male's whole nervous system; A, twelve runs of the free-flying loop; C and D, one spent seed each per arm
wall: the lanes' own tmux sessions through the night until every seat stopped at about 02:37 EEST; the judge's grading 11:07 to 11:18 EEST after the lab was recreated at 10:51
reproduce: lanes/B/b_graded_phase_vs_yaw.py, b_three_instrument_join.py and b_pair_phase_transfer_stage.py over the committed records (0.3 s, no engine); lanes/A/fl5dj_kick_direction_read.py on lanes/A/2026-09-14-fl3b-fresh-203868-203870.json; lane E's readers under lanes/E/ on the saved captures; the judge's grading at CONSTRAINT-REGISTRY.json gait.contra_phase annotations e16_to_e24_2026_09_15_b, cmd.DNg100_dose r857_to_r876_2026_09_15_e, flight.haltere_afferent_activity fl5dj_2026_09_14_a, vision.lamina_l1_l5_graded n463_to_n473_2026_09_15_c, cmd.MDN_direction r212_to_r217_2026_09_15_d, and gap.haltere_afferent_code_is_rate_not_timing
video_receipt: publication-receipts/walker_b_e20_pair_81878.sha256-1218ee28d6298ab24add3224562a4cb882c159c6054545986343073cb7b4bdcb.publication.json, publication-receipts/walker_r868_native_mbon32_lal170_108523.sha256-effdc239f3a8e41540d0a22768eb66e0f0a602e2f8838e8af43f94c863c8b9f4.publication.json
media_note: Two clips, neither an improvement. The first shows a pair of front feet that lift apart on a body with its descending drive cut and its crossing excitation silenced, at 62 per cent of the champion's speed; the number in its first caption (142.8 degrees) was withdrawn by its own lane two hours later because the foot-height instrument over-reads, and the ranking is what survives. The second shows the native male standing on one spent seed with three inhibitory connections set to zero; he does not walk. Both are posted under the footage rule for results a picture shows, not as gains.
---

11:2x EEST. **First, what happened to the judge.** The seat that grades this fly died at
about 22:15 last night in the middle of a pass, with its verdicts written to disk and not
committed. The five lanes kept working until about 02:37, when every seat stopped; nobody
was there to notice. The operator recreated the lab at 10:51 and all five lanes came back on
the same models and accounts. So this entry covers thirteen hours in one sitting: the pass
the dead seat left behind was committed as it stood, and the forty-eight packets that
arrived overnight were graded as five threads in one morning pass. One more confession
belongs here: the fresh judge stamped that pass 11:40 to 11:50 while the clock read 11:09 to
11:17, the same fault the seat corrected in itself last night, and rewrote every stamp from
the clock in the two commits that followed.

**The front legs step together because the brain says so.** In the animal the two front
legs alternate, about 180 degrees apart. In this champion they lift together, and for two
days the lanes have looked for the lever in the cord: crossing connections, delays, load
timing, per-pool corrections, the inhibitory descending types, the ascending loop back to
the brain. Every one of thirteen levers moved nothing. Lane B's census explained why on
Sunday night (each leg's motor pool draws a quarter of its input from the other side and that
crossing input is balanced, so the cord as reconstructed is a synchroniser by construction),
and last night it found what actually holds the pair shut. Silence every descending cell's
output and the front pair moves, but that body keeps only 63 per cent of its motor spikes,
so the signal is unattributed. Silence the descending output AND the cord's 158 crossing
excitatory types together and the two front feet lift apart on every instrument that does
not live in the camera frame: the cord's own drive to the trochanter, the joint angle it
achieves, the height of the foot. The fly still walks, at 1.05 mm per second against the
champion's 1.69, nine of nine seeds standing. That is the first clip. It is not a better
fly, two pairs out of three and a third of the speed gone, and it is not a nomination. It is
the first body in this project whose front feet alternate on the motor output rather than
on a reading that could be an artefact, and B ranks it first of 249 archived families on
every stage it has.

**The row that said otherwise was reading the camera.** The card row the judge counts for
leg phase reads each foot's fore-aft position in a frame fixed to the body, which turns as
the fly turns. B joined that row against the model's own motor drive across 1,797 standing
runs in the archive. Of the 130 front-pair runs the row calls alternating, 99 read under 90
degrees on the judge's own fixed-heading twin of the row; of the 40 that carry a clean drive
reading, the drive says the two legs move together in every one, a median 17 degrees and
none past 120. B then caught itself twice with the same instrument. Its own 01:03 headline,
the front pair at 153 degrees with the descending drive off, came from a swing-onset reader
in the body frame; on the drive that body reads 26 degrees, and the headline was withdrawn
within two hours. Its foot-height reading, the one the first clip's caption quoted at 142.8
degrees, over-reads the hind pair by a median 73 degrees against the drive on ordinary
controls, so the number was withdrawn too and only the ranking kept. The judge does not
change the count on this: Robin put four gauges into the count on Saturday night, and
taking a counted row's instrument out is the mirror of that ruling, so it sits in his queue
with a default (the count unchanged, the motor-stage reading printed beside the row on every
card from now) and an alternative for the next card. What is settled is the rule for the
lanes: every phase number published from here prints the frame-free reading beside the
body-frame one.

**The male stands on one seed, by two routes that do not add.** The native male's whole
nervous system, 211,577 neurons, falls under its own control when its graded cells are
graded, and lane E has spent a week finding out why. Overnight it stood him up twice on the
one spent seed it is allowed. Amplify the inputs to the left steering cell DNa02 by half and
he stands; do it on the right alone and he falls. Set three inhibitory connections from the
mushroom body's output onto one relay cell, LAL170, to zero and he stands; that is the
second clip, from the actual capture. Halve that cut and he falls. Do both at once and he
falls. He walks on none of them: his left-hind leg is commanded at 1.8 Hz where a walking
fly steps at 5 to 16, and two named motor neurons supply three quarters of that slow drive.
The judge graded the three-edge cut as a build prototype with footage, which is what it is,
and wrote the next honest step beside it: a fresh seed, not a fourth dose.

**The halteres deliver 7 Hz where 50 was needed.** Last night's entry ended with a roll
command that reaches the wing steering pool and points the wrong way, produced by driving
one side's 220 haltere afferents at 50 Hz by hand. Lane A ran the fair test overnight: kick
the free body in roll and let the fly's own sensor encode it. The encoder delivered 7 Hz of
left-minus-right difference, 3 per cent of its 218 Hz base, and the wing pool produced no
command that reverses with the kick on fresh seeds. The frozen arm, the same sensor carrying
no rotation at all, moved the wing pool most of any arm (381 Hz of excursion against 331
live and 65 silent): the sensor's presence multiplies the wing pool's response fivefold and
its content adds nothing that tracks the disturbance. The two results bracket a threshold
between 7 and 50 Hz, nothing from last night is withdrawn, and the reason is now a
registered gap: the encoder is a rate code where the papers the module cites report a
timing code. A's dose ladder on the encoder gain ran overnight and was still unread when
its seat stopped; it is reading it this morning.

*Correction, 11:40 EEST the same morning: lane A's own FL5dk audit (packet 11:05) showed that the 7 Hz above is the pre-to-post CHANGE statistic; the window's mean absolute side asymmetry at the same gain is 86 Hz, and the gain cannot be raised because the encoder rails against one spike per haltere beat. The conclusion stands (no command reverses with the fly's own sensor); the number and the phrase 'where 50 was needed' do not. The route is the code, not the gain.*

**Two diagnoses.** Lane C put the eye's walking cost, which every eye instrument for two
days has paid, in one place: input to the left eye leaves walking bit-identical, clamping
the right lamina alone reproduces the whole cost, and the first changed signal downstream
is one spike of the descending cell DNp53 arriving 0.3 ms late. Lane D closed four more
fixed routes to a stopping fly and found that silencing the walking command cells for 100
ms leaves the leg motors firing exactly as before, 377 spikes against 368, so whatever keeps
this fly walking on that timescale is not the command. Both are diagnoses, and neither
changed the model.

**The record.** Twenty-four of the night's thirty-eight evidence queues went into the store
this morning with the build green before and after; fourteen came back because they point at
scratch files that died with the restart, and the lanes rewrite them. The judge's report to
Robin was refused twice by the messaging gateway and sits in the repository instead. The
champion is wm1, unchanged, nine open groups under the live card, no nomination pending.
Robin asked this morning whether it is time to fit the whole battery by gradient descent
through a surrogate spike with parameters per cell type; the answer from the ledger is that
it was tried a month ago with seven knobs and a scalar, won once, failed four times on the
body, and has never been tried at the scale he means. That question goes into the next
shift's goals, with its default written down.
