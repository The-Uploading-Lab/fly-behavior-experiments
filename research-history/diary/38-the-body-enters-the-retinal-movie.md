---
id: 38
date: 2026-09-01
title: The body enters the retinal movie
tldr: Adding the simulated body's own yaw to the synthetic retinal movie improves both turns of the neural-hybrid controller but removes one turn from the stronger direct controller, selecting reafferent neural integration as the next BUILD target.
era: standard
video: cv20-body-reafferent-neural-parent-seed60214-sim-yaw.mp4
video_url: https://media.theuploadinglab.ai/cv20-body-reafferent-neural-parent-seed60214-sim-yaw.mp4
video_master: lanes/D/artifacts/2026-09-01-cv20-body-reafferent-neural-parent-seed60214.mp4
commit: d709fc2c, 55a51906, 437f8884, f530c5ef, 779668d6, 1ade8dac
theta: C-V20 plus vision_image_body_reafference=true
seeds: 60214 neural parent and exact audit; 60215 direct controller
n: one paired neural screen, one paired direct screen, one exact replay
wall: 94.1 s parent; 58.7 s audit; 102.4 s direct screen
reproduce: check out origin/lane-d at 779668d6 and run the renderer against its old artifact; 1ade8dac carries the corrected bytes, while the scientific runners were frozen by f530c5ef
---
**Question.** Does visual input change usefully when the walking body's own
turn enters the image, and does the neural pathway respond differently from a
direct pixel-to-command controller?

**Predicted (pre-registered, quoted).** The C-V20 registration required at
least 80% retention of both yaw arms, path speed, upright posture and every leg
cycle after body yaw was added to the panorama. Its formal gate also required
the detector to keep following the external motion schedule. The direct test
applied the same 80% retention rule to the stronger detector-phase controller.

**Method (jargon defined inline).** Reafference is sensory change caused by the
animal's own movement. The previous panorama depended only on an external 4 Hz
movie. The new phase subtracts six spatial cycles times wrapped yaw read from
the simulated body, so each completed physics step changes the next image. The
same complete 188,508-neuron, 13,620,865-edge CNS and external 7 Hz walking
waveform ran in both arms. C-V20 converted stimulated HS-neuron spike counts
into the body phase; the direct controller converted the pixel detector sign
straight into that phase.

**Result (one paired seed per controller).** On seed 60214, C-V20 changed from
+7.977/-9.243 to +9.695/-12.370 deg/s. The two yaw magnitudes increased 21.5%
and 33.8%, while path speed increased 0.7% from 7.025 to 7.073 mm/s. Final tilt
was 3.326 degrees and every leg made 15 to 45 lifts. The rendered body's yaw
ranged from -0.0719 to +0.6024 rad and contributed at most 0.5752 panorama
cycles.

The stronger direct controller behaved differently on seed 60215. Its exact
baseline was +19.255/-19.372 deg/s. Body coupling changed it to
+17.190/-5.958 deg/s, retaining 89.3% of the first arm and only 30.8% of the
second. Path speed increased 1.8%, every leg cycled and final tilt was 3.116
degrees. The failure is directional, not a fall or loss of walking.

**Interpretation.** The direct controller remains the strongest command when
the movie ignores the body. It cannot yet separate external motion from
self-motion. C-V20's intervening neural readout retains the response and
improves both arms under the same class of perturbation. The body-reafferent
neural controller is therefore the BUILD exploitation target. This comparison
does not establish that the neural route implements biological reafference; it
identifies the better executable architecture for the next engineering step.

**Problems / what bit us.** All three formal outcomes are failures. The C-V20 parent
failed because its old detector gate assumed external-motion direction. A new
exact audit matched detector sign on all 2,000 nonzero retinal-slip ticks with
zero mismatches, but still missed its registered opposition count by one:
125/108 observed against 126/108 required. The onset frame has zero external
displacement. The failed label was retained and no third audit was opened. The
direct controller formally failed because its second arm retained only 30.8%.
Both screens used one seed, the panorama is synthetic, HS neurons are externally
stimulated, the body waveform and adhesion timing are external, and raw motor-
neuron body drive is zero.

Selective transport exposed one more defect before merge. The new evaluator
called a quaternion-to-yaw helper inherited from unrelated Lane D history. The
helper had not been transported, so a real reafferent run would have raised a
`NameError` despite the focused unit tests passing. Only the conversion was
added. A complete six-second main execution then reproduced the direct arm's
yaw, speed, tilt, lifts and 2,000/2,000/0 slip receipt exactly.

**What changed in the model.** The default-off renderer remains bit-identical.
One optional parameter now makes the synthetic panorama depend on body yaw and
records actual retinal-slip consistency. The verified r23 champion did not
change. The direct controller remains a body-independent performance reference;
body-reafferent C-V20 becomes the neural-integration target.

**Attribution.** Robin proposed coupling eyesight to the walking fly and then
ordered the fleet to abandon flat research, explore distinct options and exploit
a real gain hard. Lane D implemented the body-dependent renderer, ran the neural
and direct comparisons, preserved both failed formal labels and rendered the
existing neural-parent trajectory. Independent judge reviews replayed all three
results. The judge integrated the smallest executable mechanism, found the
missing yaw helper through a second code review, reran the complete body and
selected the target without ordering more seeds.

The exact top-down source video is
https://media.theuploadinglab.ai/cv20-body-reafferent-neural-parent-seed60214-sim-yaw.mp4,
SHA-256
`4a0af176692a9a344c222a58824c785cc1a4a8966358b23f774852368c2de8c0`.

— the judge
