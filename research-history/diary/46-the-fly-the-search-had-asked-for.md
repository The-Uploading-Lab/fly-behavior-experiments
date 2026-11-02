---
id: 46
date: 2026-09-02
title: The fly the search had asked for
tldr: The 18:30 correction had found that nl1 was crowned on the engine's default graded dials because main's runner never read the three values its search stated. Once lane A's pass-through was on main, the judge ruled an ordering clause (a challenger runs the holder's 24 paired seeds; the higher qualifying count takes the title), ran the stated-dials fly on nl1's seeds, and it qualified 19 of 24 against nl1's 15. ng1 is champion, differing from nl1 in exactly those three numbers. Lane A had registered 3-7 qualifying per battery and got 11, a prediction falsified from above. The same evening lane D found the first descending manipulation that produces standing on this substrate (DNg93 at 94 Hz prevents ignition 3 of 3, upright, motor pool at a tenth of its walking rate) and that the same neuron cannot stop a command already running.
era: standard
commit: a476f5663 (batteries), coronation commit tagged ng1
seeds: judge 30001-30024 paired with nl1's; D 60452-60454
n: two title batteries, one paired comparison, three stop probes replayed
wall: 712 s + 714 s battery wall; 19:14-19:49
reproduce: PYTHONPATH=. .venv/bin/python lanes/judge/title_match_v33.py --theta results-ng1-theta.json --seeds 30001-30012 --battery 1 --out receipt.json
video: walker_ng1_body.mp4
video_url: https://media.theuploadinglab.ai/walker_ng1_body.mp4
---

**Question.** The 18:30 correction (entry 45's tail) left two flies
sharing one theta file: nl1, which the engine ran at the default graded
dials, and the fly lane A's search had optimised, with `graded_every 5`,
`graded_v50 -49.38 mV`, `graded_slope 4.60 mV` applied. Which one holds
the title, and by what rule, when both can pass the same bar?

**Method.** The bar had no ordering rule, so the judge wrote one before
running anything (14eb4d71b, 19:21): the challenger runs the holder's 24
paired judge seeds on the holder's engine at spike_reg 44; the title
goes to the higher qualifying count over the 24; a tie keeps the holder;
the challenger must still pass each twelve-seed battery on its own;
displacement and speed order nothing. Pairing is valid because the theta
was frozen at 06:50, the seeds were first used at 12:00, and the 18:54
gate reproduced nl1's battery-1 row 26 of 26 fields on the current
engine. Lane A registered its prediction at 19:12: 3-7 qualifying per
battery, falsified outside 2-8, from a Metal VNC-cut screen where the
stated-dials fly scored 41% in band against nl1's 50%.

**Result.** Battery 1 (30001-30012): standing 12/12, qualifying 11/12.
Battery 2 (30013-30024): standing 10/12, qualifying 8/12. Paired, 19
against nl1's 15: seven seeds qualify for ng1 only, three for nl1 only.
**ng1 is champion.** Median displacement of standing runs 4.375 and
3.85 mm (nl1 3.79 and 4.165 on the same seeds); median tip cadence of
qualifying runs 5.455 Hz, no commanded step frequency in the theta. Lane
A's prediction was falsified high: 11 against a ceiling of 8. The VNC
cut at the screen's duration ranks these two flies in the opposite
order from the whole-CNS six-second battery.

Lane D, in parallel: DNg93 is the largest direct descending projection
onto leg motor neurons (6,793 synapses, GABA-predicted). Driven at 94.26
Hz with no walking command it prevents ignition 3 of 3 and leaves the
fly upright with the motor pool at a tenth of its walking rate, the
first manipulation on this substrate that produces standing rather than
a differently-walking fly. Added on top of a live DNg100 command, at 1x
and 2x rate, it does not stop the fly: path speed 7.1 against 6.9-8.4
mm/s, 0 of 3 stand. Silencing DNg93 with no command does not ignite the
fly either (onsets identical to intact to the bin). Judge replays on
main, Metal, of the stop probe and the silenced arm: every speed bin,
pool rate and active-motor-neuron count identical. So the model has a
start inhibitor and, as far as one rate on one neuron shows, no stop
signal for a running command.

**Interpretation.** The three graded dials are the only difference
between the two champions, and 7-vs-3 discordant seeds (McNemar
two-sided p about 0.34) is not by itself strong evidence they matter;
the title moves on the count rule the clause states, which is what the
clause is for. The stronger result is procedural: a fly can now be
crowned on the values its search chose, and the pass-through that was
missing is the kind of shape defect the 2026-09-01 measured-structure
clauses exist to catch. `inh_gain 1.0` is stated in the theta of record
for the same reason, changing no run, and `lanes/A/shape_gate.py` passes
on this record where it failed on nl1's.

**Problems.** Two seeds in battery 2 fall outright at 155.6 degrees and
one lifts a foot only twice; the body is not fall-proof. The graded
dials are one searched point in a four-dial space, not a map. Lane A's
screen predicted the wrong direction, so the Metal VNC-cut screen is not
yet a proxy for the title instrument at this duration, and that needs
its own row before another prediction is registered from it. The
judge's own timestamps ran a minute ahead of its commits twice today.

**What changed in the model.** `CHAMPION.json`: ng1, whole CNS, v3.3,
nl1 to lineage. `HALL-OF-FAME.json`: ng1 record, throne holder ng1, the
ordering clause in the v3.3 bar text, nl1's clip re-scored at tier 3.
`results-ng1-theta.json`: the judge-owned theta of record.
Registry: `cmd.DNg93_stop` gains the stop-probe, dose and silenced
readings. Video: https://media.theuploadinglab.ai/walker_ng1_body.mp4

**Attribution.** Robin set the standing demand for the day ("send
reports to telegram and video if there is. If progress isn't there,
suggest how to get it") and had delegated the bar. Lane A built the
search, found and folded the missing pass-through, and registered the
prediction that the batteries falsified. Lane D designed and ran the
DNg93 probes and the LAYER1 start/stop table. Lane E's serotonin map
entered the ledger. The judge found the stated/applied split, ruled the
ordering clause, ran both batteries, replayed D's probes, and crowned.

cites:: MODELCARD-ng1.md, lanes/judge/2026-09-02-title-v33-ng1-battery1.json, lanes/judge/2026-09-02-title-v33-ng1-battery2.json, lanes/D/2026-09-02-dng93-stop-probe-result.json, lanes/D/2026-09-02-dng93-stop-dose-result.json, lanes/D/2026-09-02-dng93-silenced-ignition-result.json
uses-data:: CHAMPION.json, HALL-OF-FAME.json, results-ng1-theta.json
