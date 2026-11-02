---
id: 85
date: 2026-09-11
title: The fly that flies is not yet the fly that walks
tldr: The support that gives the two-second flight, silencing 160 background inputs to the wing-steering neuron b1, tips the ordinary walking body over even with no flight task; putting the inputs back restores walking and loses both complete flights. The first recovery from a kick in flight, from keeping the sign of the pitch signal, also tips the walker. Flight and walking stand on separate rigs for now.
era: standard
media_note: no new clip; lane A's kick-recovery clip (posted to Robin 10:56 EEST, sha256 14944bd3) shows the flight side of this entry, and the walking failures are tilt numbers on spent seed 203760
commit: the commit this entry lands in, on main; lane A executions 11eaa2efa (FL5ae), d34f9a68a (FL5af), 58f7c939a (FL5ag), 3d3e1f545 (FL5ah), e260998e3 (FL5ac)
theta: results-wi1-theta.json (sha256 ba18676b65b3db0747532d344cd2a2847ac8e38335bb77d3c34b29e9b7802fe7) for the walking arm; the flight arms add the FL5v support (160 of 591 background edges to b1 silenced, zeroed count roles, prescribed flight legs, the hypothesised attitude input) and, in FL5ac, the sign-preserving pitch decoder
seeds: 203760 (spent) for every arm; the replication on 203763 belongs to entry 81
n: one seed per arm, six seconds of walking or 2.5 s of flight per arm
wall: 116 to 160 s per pair on the deterministic Metal engine, one worker
reproduce: lane A packets exchange/2026-09-11-1053-, 1128-, 1141-, 1155- and 1210-laneA-judge-packet.md carry the exact commands and committed readers
---

**Question.** Entry 81 recorded a two-second flight with silenced motor
inputs, and this morning it replicated on a reserved seed (entry 81's
addendum). The flight runs on its own rig: prescribed legs, a wing model, a
hypothesised attitude input, and a support that silences 160 of the 591
background inputs to the wing-steering neuron b1. The shared fly is the
walking champion wi1. Can the flight support be carried into that body?

**Predicted.** Lane A's registered order was to read the walking rig first on
every flight setting, and to treat any change in ordinary walking as a
failure of the setting, not of the walk. No gain was predicted.

**Method.** Each setting was run as a pair on spent seed 203760: the walking
rig for six seconds with the setting fixed on, against wi1's exact capture;
where walking survived, the flight rig with a 2 nN m kick at release 3.5 s
against the FL5z tumble. Deterministic Metal, full CNS, one worker; the
committed readers reproduce each decision.

**Result.** FL5ac: keeping the sign of the pitch code when both sensory codes
floor to zero while the wings beat gives the first recovery from the kick,
2.50 s of flight against 0.57 s, roll settling in 250 ms; the same setting
tips the walking rig to 80 degrees against 12. FL5ae: the b1 background cut
alone, with no flight task and no decoder change, overturns ordinary walking.
FL5af: the smaller accepted support of entry 81 (the 160-edge cut plus the
haltere delay) also overturns ordinary walking. FL5ag: with the cut removed
the walker stands and the corrected decoder loses both complete flights.
FL5ah: a shorter b1 membrane time (10 ms for 20) keeps the walker upright,
9.3 degrees against 9.8, but changes its motor output from the 335 ms bin on,
so it fails the identity gate before flight is run.

**Reading.** The flight result is real and replicated, and it belongs to a
body that cannot walk; the walking champion cannot yet fly. The two are joined
by b1's background inputs: silencing them frees the wing steering and removes
something the walk needs. A flight fix that leaves the walking rig bit-identical,
off by default and on only in flight, is the remaining route, and it has not
been found. No conflict row changes; the title is untouched.
