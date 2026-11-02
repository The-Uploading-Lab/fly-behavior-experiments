---
id: 47
date: 2026-09-02
title: The command was a clamp
tldr: Two results in one evening. Lane B's evidence-map union, ng1 with 3,352 further graded rows from the approved map, tied ng1 on its own twelve seeds and walked a third further, so it was nominated and title-matched on ng1's 24 paired seeds under the ordering clause; it qualified 16 against ng1's 19 and ng1 keeps the title. It stood more often than ng1 and stepped slower, seven seeds landing at 4.1 to 4.9 Hz under the 5 Hz cadence floor. Lane D then read the engine and found that the walking command is delivered as forced spikes, so no cell in the connectome can gate it by construction; the strongest GABA input onto the command neuron, driven at full rate, changed nothing. Every stop negative on the start/stop table is bounded by that, and the fix is an engine seam, not a search.
era: standard
commit: 54ddfbd09 (batteries and folds), match record in the following judge commit
seeds: judge 30001-30024 paired with ng1's; D 60465-60467
n: two title batteries, one paired comparison, six stop arms replayed
wall: 2,805 s + 2,817 s battery wall (NumPy, both batteries sharing the CPU); 22:37-23:24
reproduce: PYTHONPATH=. .venv/bin/python lanes/judge/title_match_v33.py --theta lanes/judge/theta-ng1-mapunion-challenger.json --seeds 30001-30012 --battery 1 --out receipt.json
video: walker_ng1_body.mp4
video_url: https://media.theuploadinglab.ai/walker_ng1_body.mp4
---

**Question.** Two questions, one evening. First: does adding every
graded-release row the evidence map allows (3,352 rows: APL, HSE, HSN,
HSS, L1, L2) to ng1's 24 make a better fly on the title instrument? Lane
B's twelve-seed card at 21:49 said it might: 9 of 12 qualifying, the
same as ng1, and a median displacement of 4.64 mm against 3.52. Second,
from lane D: why has no descending manipulation ever stopped a running
command?

**Method.** The union was nominated as a frozen theta at 22:36, before
any judge seed ran, so ng1's 24 paired seeds were fresh for it; two
twelve-seed batteries then ran on the holder's engine and backend
(NumPy, spike regularity k = 44), the same instrument that crowned ng1.
The count rule of the ordering clause decides: higher qualifying count
over the 24 takes the title, a tie or less keeps the holder. Lane D's
question was answered by reading `cns.py`: the command cells are drawn
at `cmd_hz` and unioned into the fired set after the membrane update,
gated only by their own refractory counter. The behavioural check drove
DNge129, which sends 439 synapses onto the DNg100 command pair and 7 of
its 9,082 outputs onto motor neurons, at the command's own rate on ng1,
three fresh seeds, both arms; the judge replayed all six arms exact.

**Result.** The challenger qualified 16 of 24 against ng1's 19; both of
its batteries passed on their own (7 and 9 qualifying, standing 11 and
12 of 12). ng1 keeps the title. The discordant seeds tell the mechanism:
the challenger qualified on five seeds ng1 did not, three of them seeds
where ng1 fell or lifted a foot twice, and lost eight, seven of them
standing runs with tip cadence between 4.09 and 4.91 Hz, under the 5 Hz
floor by a fraction of a hertz. Median displacement over the 24 was 3.90
mm for ng1 and 3.79 for the union; lane B's 32% gain on its own seeds
did not carry to the judge block. The union is the more stable body and
the slower stepper.

Lane D's arms: commanded 8.74, 7.45, 8.58 mm/s; with DNge129 at 94 Hz
8.76, 9.19, 9.13. Onset 0.1 s in every arm. The inhibitor reached the
command neuron and the command neuron kept firing, because a forced
spike does not consult the membrane.

**Interpretation.** The graded dials ng1 carries (`graded_every 5`,
`graded_v50 -49.38 mV`, `graded_slope 4.60 mV`) were searched on 24
rows; on 3,376 rows they were never searched, and the cadence deficit is
the shape a wrong release curve would take on a body that otherwise
stands better. The union is not closed; it is the first candidate to
search from, with the dials free. Lane D's finding is larger than its
three seeds: `cmd_hz` asserts that the command cell fires at that rate
whatever the brain does to it, the strongest possible claim about a
neuron that receives 6,216 synapses, 3,352 of them GABA-predicted, and
it was made by default. Every "the connectome cannot stop walking"
result on the start/stop table now reads "downstream of a clamp". The
positive results stand: DNg93 alone at 47 Hz delaying ignition does not
involve gating the command.

**Problems.** The challenger's cadence misses are 0.1 to 0.9 Hz under a
floor whose resolution is a step of about 0.09 Hz, so a coarser or
finer tip-frequency estimator would move the count; the floor is a
measured band (Wosnitza, DeAngelis) and stays, but the instrument's
resolution near it is now worth its own row. The batteries took four
times ng1's wall because two ran at once on the CPU. Lane E's per-source
graded delay queue found no gain for the union on four fresh seeds and
lane C's two-seed Metal transfer disagreed with its NumPy twin on one
seed; neither ordered the match, both are on record. The command fix
(`gap.command_membrane_drive`, owner lane A with compute, expires
2026-09-05) is default-off and bit-identical when off; until it lands,
the stop rows are bounded, not wrong.

**What changed in the model.** Nothing in the champion. Registry:
`cmd.command_ungateable` (CANDIDATE, FEMALE-ONLY for the arm, the
mechanism reading engine-wide), `gap.command_membrane_drive`, the
defence recorded under the BANC live title. Ledger:
`cns.command_delivery`, provenance assumed, with D's numbers as the
paired baseline. Engine: lane A's receptor seam step 1
(`receptor_classes`, per-class sign and gain folded into the assembled
weight before balance, probe BIT-IDENTICAL) and the B and C additions,
all default-off. Video: none new; ng1's clip stands,
https://media.theuploadinglab.ai/walker_ng1_body.mp4

**Attribution.** Robin ruled lane C's GPU access at 21:46 and asked the
provenance question (does a searched parameter improve when its bounds
come from a measurement) that lane E's R148 now tests. Lane B built the
union, ran the twelve-seed card and the searches that motivated it. Lane
D read the engine, designed the DNge129 arm and named the fix. Lane A
built the receptor seam and reported its own falsified prediction. The
judge nominated, ran the batteries, replayed D's six arms and A's gate,
and recorded the defence.

cites:: lanes/judge/2026-09-02-title-v33-mapunion-battery1.json, lanes/judge/2026-09-02-title-v33-mapunion-battery2.json, lanes/judge/theta-ng1-mapunion-challenger.json, lanes/D/2026-09-02-command-ungateable-result.json, exchange/2026-09-02-2258-laneD-judge-packet.md, exchange/2026-09-02-2149-laneB-judge-packet.md
uses-data:: CHAMPION.json, CONSTRAINT-REGISTRY.json, CALIBRATION-LEDGER.json
