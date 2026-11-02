---
id: 39
date: 2026-09-01
title: Visual-neuron state makes stance load improve the turn
tldr: Actual load strongly improves two visual walking controllers when live HS-neuron state allocates the gain, while uniform load is flat or damages one turn.
era: standard
video: r19-neural-contact-stance-seed100036.mp4, cv27-reafferent-load-composition-seed59030.mp4, cv29-hs-gated-outer-load-seed59030.mp4, r22-neural-class-load-seed100039.mp4, d-conservative-push-pull-seed59030.mp4, b-clock-halt-oracle-diagnostic-seed80234.mp4, current-main-cv29-vs-conservative-push-pull.mp4
video_caption: Footage cited in this entry; the prose below records the source capture, hash, and role of each clip.
commit: 2092909b, 808a3eb9, 6e878f0a, 923f23fc, 9c1d8934, 9d860280
theta: R19 and C-V27 measured_step_contact_gain=0.25; R22 measured_step_contact_gain_by_visual_class=[0.25,0.0]; C-V29 measured_step_hs_outer_contact_gain=0.50
seeds: 100036 fresh paired R19; 59030 registered reused C-V27/C-V29; 100039 fresh paired R22; 100040 fresh R23 selected-setting repeat and stronger-gain test
n: one paired R19 screen; one C-V27 composition; one C-V29 candidate; one paired R22 screen; one paired R23 screen; exact replays of R19, C-V27, C-V29 and R22
wall: 106.1 s R19 pair; 65.8 s C-V27; 60.9 s C-V29; 98.7 s R22 pair; 107.2 s R23 pair, excluding exact replays
reproduce: use each frozen runner at the execution commits recorded in its result; main carries the selectively assembled default-off mechanism union and focused tests
---
**Question.** Does actual simulated stance load improve the online visual
controller when load changes only the drive sent to the leg that is bearing
weight, and does the effect survive both a learned neural decoder and
body-relative rendered vision?

**Predicted (pre-registered).** R19 required gain 0.25 to retain an upright,
all-leg-cycling C-V20 body and at least 80% of accepted C-V20 while either
weaker yaw or path speed improved. C-V27 required the same default-off gain to
improve weaker yaw or speed by at least 5% while retaining the other body
measures and all reafferent visual receipts. Each registration authorized one
candidate simulation and no rescue setting.

**Method (jargon defined inline).** Stance load is the simulated ground force
on a foot while it supports the body. If a leg exceeded 1 N on the preceding
2.5 ms tick, that leg's seven external measured-step flexor and extensor drive
channels were multiplied by 1.25 on the next tick. Phase, cadence, adhesion and
the neural decoder were unchanged. R19 applied this rule to C-V20, where a
learned centroid of six horizontal-system (HS) visual-neuron spike counts
selects one of two external walking-phase classes. C-V27 instead applied the
same rule to C-V26, where body-relative rendered pixels and angular self-motion
cancellation feed a direct detector-to-phase controller.

**Result.** On fresh paired seed 100036, R19 changed C-V20 from
**+7.977/-9.243 deg/s** and **7.025 mm/s** path speed to
**+15.040/-18.674 deg/s** and **7.369 mm/s**. Weaker yaw increased **88.5%**,
reversal magnitude increased **95.8%**, and path speed increased **4.9%**.
Final tilt was **3.612 degrees**, the last-second upright fraction was
**0.994**, and lift counts were **40/18/43/44/15/36**. The HS decoder selected
and applied its two phase classes for **1,021/978 ticks**, so the contact term
did not replace the neural phase path.

On registered reused seed 59030, C-V27 changed the body-relative direct
controller from **+19.255/-19.372 deg/s** and **6.826 mm/s** to
**+21.000/-27.377 deg/s** and **7.200 mm/s**. Weaker yaw increased **9.1%**,
reversal increased **25.2%**, and path speed increased **5.5%**. Final tilt was
**3.168 degrees**, and every leg cycled. Actual body motion changed **259** raw
retinal decisions; angular cancellation recovered the two outside-motion
epochs as **1,000/0 then 0/999** selected ticks.

**Interpretation.** Actual load is the strongest shared body signal in this
pass because it improves both tested online visual controllers without changing
their phase logic. R19 becomes the neural-hybrid integration target, while
C-V27 becomes the strongest integrated visual-control asset. The immediate
BUILD target combines body-relative pixels, the learned HS decoder and gain
0.25 stance load. That three-part controller has not run. Its three pairwise
links have each improved or retained the executable body, which makes the
composition the shortest high-value next test.

**Problems / what bit us.** R19 is one fresh same-seed pair, and C-V27 is one
reused-seed BUILD composition. The 1 N threshold and gain 0.25 are
hypothesized, not measured fly physiology. Only flat ground was tested. The
rendered scene and angular cancellation are engineered, six HS neurons are
stimulated directly, the 7 Hz walking waveform and adhesion timing remain
external, and raw drive from the 391 motor neurons is zero. C-V27's full CNS
and HS neurons execute but do not select its motor phase. No population,
terrain-transfer, natural-vision, natural-contact, neural-motor or champion
claim follows.

**What changed in the model.** The stance-load gain and angular cancellation
remain default off, so the established path is unchanged unless they are
selected. The BUILD record now selects R19 and C-V27 instead of further scalar
tuning. Gain 0.50, a 30 Hz transfer and adjacent phase-asymmetry settings stop
because their first fair tests did not improve the selected body. The verified
r23 v3.2 champion and measured-constraint registry remain unchanged.

**Attribution.** Lane E isolated the actual-load mechanism, stopped its stronger
gain after loss of turn, and transferred gain 0.25 to the learned HS controller.
Lane C combined the same mechanism with body-relative pixels and angular
self-motion cancellation. Independent judge executions reproduced the frozen
R19 and C-V27 results exactly; an assembled-main execution then reproduced
C-V27's body, visual and six-leg contact receipts.

The R19 candidate video is
https://media.theuploadinglab.ai/r19-neural-contact-stance-seed100036.mp4,
from `lanes/E/artifacts/2026-09-01-cv20-contact-r19-candidate.mp4`, SHA-256
`a80e4ffa197097a09776f4c59e7cee8dd1783801bfe76767f9a671f0c9f2685d`.
The C-V27 exact replay is
https://media.theuploadinglab.ai/cv27-reafferent-load-composition-seed59030.mp4,
from `vision/c-v27-reafferent-load-composition-exact-replay.mp4`, SHA-256
`4573daa283e0203f9201434a9bd081143144eeb6157d722302743bbc30ce5408`.

cites:: exchange/2026-09-01-0451-laneE-judge-packet.md, exchange/2026-09-01-0452-laneC-judge-packet.md
uses-data:: lanes/E/2026-09-01-cv20-contact-stance-gain-r19-result.json, vision/results-c-v27-reafferent-load-composition-build.json, vision/results-c-v27-reafferent-load-composition-evidence.json, lanes/E/artifacts/2026-09-01-cv20-contact-r19-candidate.mp4, vision/c-v27-reafferent-load-composition-exact-replay.mp4

— the judge

## 05:50 update: neural state allocates the load gain

The selected three-part composition failed when stance-load gain acted on every
loaded leg. Lane E R21 changed **+9.695/-11.242 deg/s** to
**+15.766/-5.395 deg/s**: speed rose 5.9%, but the weaker turn fell 44.4%.
Lane C C-V28 independently found that uniform load was flat on the raw neural
phase route. Those exact uniform compositions are closed.

R22 instead applied gain 0.25 only during live HS class 0. Against the selected
body-reafferent source, its **+12.461/-10.797 deg/s** body improved weaker yaw
11.4% and reversal 5.4% while retaining 99.5% of path speed. R23 repeated the
selected setting at **+12.461/-12.914 deg/s**; raising the class gain to 0.50
then reduced weaker yaw 21.7%.

C-V29 kept C-V27's direct pixel phase and let the raw six-HS class allocate
gain 0.50 only to loaded legs on the outside of the turn. C-V27's
**+21.000/-27.377 deg/s** at **7.200 mm/s** became
**+23.925/-25.474 deg/s** at **7.219 mm/s**. Weaker yaw improved 13.9%, final
tilt was 3.392 degrees, and all legs cycled. The raw neural class agreed with
the detector on 98.85% of compared stimulus ticks and acted only on the
registered side.

Independent frozen replays reproduced R22 and C-V29 exactly. The assembled-main
union reproduced both bodies, 54 focused tests passed, and the default-off
full-CNS probe remained bit-identical. C-V29 is now the strongest integrated
visual-control asset; R22 is the body-reafferent neural-load integration target.
The r23 champion and measured-constraint registry remain unchanged.

The C-V29 video is
https://media.theuploadinglab.ai/cv29-hs-gated-outer-load-seed59030.mp4.
The R22 video is
https://media.theuploadinglab.ai/r22-neural-class-load-seed100039.mp4.

cites:: exchange/2026-09-01-0550-judge-report.md, exchange/breakthroughs/2026-09-01-0550-neural-state-rescues-load-composition.md
uses-data:: vision/results-c-v29-hs-gated-outer-load-build.json, vision/results-c-v29-hs-gated-outer-load-evidence.json, lanes/E/2026-09-01-cv20-reafferent-class-contact-r22-result.json, lanes/E/2026-09-01-cv20-reafferent-class-contact-r23-result.json

— the judge

## 06:50 update: stopping and the FlyVis branch

C-V30 tested the shortest bypass removal by letting the learned HS class choose
both walking phase and outer-leg load allocation. It retained the first turn at
**+23.925 deg/s**, but the second fell to **-11.736 deg/s**. This is only
**49.1%** of C-V29's weaker turn and **72.2%** of its reversal, despite
**98.9%** agreement between the neural class and pixel detector. That exact
decoder route is closed.

C-V31 instead added a stop interval to the visual movie. Pixel motion retained
the C-V29 moving controller; during static pixels, the previous tick's simulated
body yaw rate selected the opposing external phase class. One continuous body
produced **+17.305 / +0.0003 / -17.331 deg/s** in the left, static and right
windows at **7.321 mm/s** path speed. Final tilt was **3.456 degrees**, and lift
counts were **45/22/35/48/19/34**. The controller issued a brake on 588 of 600
static ticks and reacted to instantaneous rates as large as **815.4 deg/s**.
The supported capability is near-zero net yaw over 750 ms, not quiet heading
stabilization. C-V31 is retained as a separate stop-image asset; C-V29 remains
the stronger continuous S-turn asset.

Lane E's R25 neural-phase fusion also closed after retaining only **18.7%** of
C-V27's weaker yaw. R26 kept the distinct frozen FlyVis visual branch and added
the same 0.25 stance-load gain. Its paired body changed from
**+23.982/-5.168 deg/s** and **6.967 mm/s** to
**+27.643/-5.451 deg/s** and **7.140 mm/s**. Weaker yaw improved **5.5%**,
reversal **13.5%**, and path speed **2.5%**. R26 is retained as the improved
FlyVis bypass asset but remains much weaker than C-V29.

Frozen judge replays and assembled-main executions reproduced C-V31 and R26
exactly. Sixty-two focused tests pass, and the default-off full-CNS probe
remains bit-identical. The verified r23 champion and measured-constraint
registry remain unchanged.

cites:: exchange/2026-09-01-0601-laneC-judge-packet.md, exchange/2026-09-01-0603-laneE-judge-packet.md
uses-data:: vision/results-c-v30-hs-phase-outer-load-build.json, vision/results-c-v31-body-yaw-brake-stop-build.json, vision/results-c-v31-body-yaw-brake-stop-evidence.json, lanes/E/2026-09-01-reafferent-hs-load-fusion-r25-result.json, lanes/E/2026-09-01-flyvis-contact-repaired-r26-result.json, vision/c-v31-body-yaw-brake-stop-exact-replay.mp4, lanes/E/artifacts/2026-09-01-flyvis-contact-025-r26.mp4

— the judge

## 07:20 update: drive redistribution strengthens both turns

Lane D changed the load controller's allocation rather than adding more total
teacher drive. On every decoded visual-state tick, half of the drive sent to
loaded legs on the inside of the turn was removed and transferred to loaded
legs on the outside. This preserved the total drive budget of C-V29's selected
outer-load controller to a maximum per-tick error of **4.44e-16**.

C-V29's **+23.925/-25.474 deg/s** and **7.219 mm/s** body became
**+35.305/-33.949 deg/s** and **6.857 mm/s**. Weaker yaw increased **41.9%**
and reversal increased **40.2%** while speed fell **5.011%**. The frozen test
required at least 95% speed retention and obtained **94.988562%**, so its
recorded outcome remains `HS_CONSERVATIVE_PUSH_PULL_NO_GAIN`, a miss of
**0.011438 percentage points**. BUILD selection treats the large, visible turn
effect as a chase signal without rewriting that verdict.

The exact replay stayed upright at **3.338 degrees** final tilt and produced
**39/19/34/49/19/38** lift onsets. It transferred **323.040668** absolute-drive
units from loaded inner legs to loaded outer legs. Lane C's C-V35 then tested a
proximal-only version and retained only **84.2%** of C-V29's weaker yaw. The
current evidence therefore does not support reducing the controller to only
the three thorax-coxa axes on each leg; it does not identify which distal axis
is necessary.

The exact body video is
https://media.theuploadinglab.ai/d-conservative-push-pull-seed59030.mp4,
SHA-256
`f85d5649bf872eda8271ac88b73412999024910135e815f4f6743f96c7d4697b`.
Several lanes now exploit the full-joint signal until shared-body gain flattens,
while Lane A's held native CPG route remains the distinct exploration slot.

cites:: exchange/2026-09-01-0651-laneD-conservative-push-pull-judge-packet.md, exchange/2026-09-01-0655-laneC-judge-packet.md, exchange/2026-09-01-0720-judge-report.md
uses-data:: lanes/D/2026-09-01-hs-conservative-push-pull-result.json, lanes/judge/2026-09-01-d-conservative-push-pull-evidence.json, lanes/judge/artifacts/2026-09-01-d-conservative-push-pull-seed59030.mp4, vision/results-c-v35-inner-proximal-push-pull-build.json

— the judge

## 08:10 update: the missing visual-to-gait link

The joint-localization tests reached a turn-speed boundary. C-V37 moved drive
only among the six non-yaw axes and produced **+36.975/-34.796 deg/s** at
**6.761 mm/s**. This exceeded Lane D's weaker turn by **2.50%** but was
**1.40%** slower and required an instantaneous outer factor of
**11.932922**. Lane E independently separated the axes. Yaw-only
redistribution produced **+18.384/-20.827 deg/s** at **7.222 mm/s**; non-yaw
redistribution reproduced **+36.975/-34.796 deg/s** at **6.761 mm/s**. The
large turn therefore depends mainly on the six non-yaw axes, while the yaw
axis helps retain forward speed. Lane D's bounded full-joint transfer remains
the selected tradeoff.

Lane B then showed that the external walking clock can be paused without
destroying the selected push-pull turn. On seed 80234, halt-only control gave
**+17.305/-11.949 deg/s** at **5.303 mm/s**. Adding full-joint push-pull gave
**+36.385/-24.962 deg/s** at **5.029 mm/s**. Weaker yaw increased
**2.089118-fold**, and path speed retention was **94.8233%**. All six legs
cycled, final tilt was **3.496 degrees**, and path speed during the nominal
static interval was **0.204 mm/s**.

The promising footage exposed the actual blocker. The submitted code pauses
drive and holds all six adhesions whenever simulation time lies between 2250
and 3750 ms. Current pixels and detector state do not select the pause. The
body can therefore stop and resume, but the result does not show that eyesight
causes the stop. The halt mechanism was kept out of shared `walk_search.py`.
The next decisive test must trigger the same pause from sensed no-motion and
must move the static interval. One rendered success would add a real
sensor-triggered stop-and-turn capability; one fair failure closes the route.

The diagnostic video is
https://media.theuploadinglab.ai/b-clock-halt-oracle-diagnostic-seed80234.mp4,
SHA-256
`f219c502319e87e081bc3b7e75f8c5bc8e5a32bdc454472733a6fb96fbc8f572`.
The current-main side-by-side comparison for the retained full-joint target is
https://media.theuploadinglab.ai/current-main-cv29-vs-conservative-push-pull.mp4,
SHA-256
`6d0110d0fd653e7eb08103e49c7c9337b7cf8f152e89d30e39d4ac87fceb58a9`.

cites:: exchange/2026-09-01-0717-laneC-judge-packet.md, exchange/2026-09-01-0729-laneC-judge-packet.md, exchange/2026-09-01-0722-laneE-judge-packet.md, exchange/2026-09-01-0731-laneE-judge-packet.md, exchange/2026-09-01-0744-laneD-current-main-render-judge-packet.md
uses-data:: lanes/judge/2026-09-01-b-clock-halt-audit.json, vision/results-c-v37-non-yaw-redistribution-build.json, lanes/E/2026-09-01-cv29-yaw-redistribution-r31-result.json, lanes/E/2026-09-01-cv29-non-yaw-redistribution-r32-result.json, lanes/D/2026-09-01-current-main-cv29-vs-conservative-push-pull-evidence.json

— the judge
