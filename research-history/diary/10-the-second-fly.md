---
id: 10
date: 2026-08-17
title: The second fly agrees with the first
tldr: We compared our walking circuit's wiring against a second individual fly, connection by connection. The strong wiring we simulate is 87% shared; the weak wiring we discard is a coin flip.
era: standard
commit: see "THE SECOND FLY part 1" commit
theta: none — pure data analysis
seeds: none — deterministic
n: 2,918,999 testable connections; 24,206 matched neurons
wall: ~4 min
reproduce: .venv/bin/python3 mancagree1.py
---
**Question.** Everything we simulate comes from ONE fly's wiring
diagram. Are our conclusions about "the fly connectome" — or about
this one fly?

**Predicted (pre-registered, quoted).** "MA1: conservation rises with
BANC synapse count, and >=10-synapse edges are substantially more
conserved than 1-9. MA2: conservation ~flat -> the cut stands on its
behavioural result only." Written before the comparison ran — the
day had already produced one retraction about this very threshold's
history, so the bar was set blind.

**Method.** A second complete fly nerve cord exists: MANC, a male,
published by Janelia in 2024 (public download, URL recorded this
time). The BANC team had already matched 24,206 of our walking-core
neurons to their MANC counterparts, cell by cell. For every
connection in our core where both endpoints have a match, we asked:
do the same two cells connect in the other fly, and how strongly?

**Result (numbers).** Conservation climbs smoothly with connection
strength: 1-synapse connections are shared between the flies 38% of
the time; 2 synapses, 54%; 5-9, 78%; 10-19, 85%; 100+, 94%. Our
simulated core (10+ synapses): **87.1% shared. The tail we discard:
49.9%.** Where a strong connection exists in both animals, its
strength agrees too (correlation 0.47 on 200,448 shared
connections).

**Interpretation.** The wiring we simulate is, to first order, the
wiring every fly has; the wiring we drop is closer to individual
handwriting. The same weak-varies/strong-conserved law was known
from a worm study (Witvliet 2021); this is, to our knowledge, its
first direct confirmation across two fly nerve cords — and it is a
male and a female agreeing, which makes the shared core more
striking, not less.

**Problems / what bit us.** Earlier today the agent claimed the
10-synapse cut was "a noise threshold, not an individuality filter"
— the human remembered otherwise, the ledger proved the human right,
and the claim was retracted in writing. This experiment then tested
what the ledger asserted. Caveats that stand: the matches are
morphology-based, not perfect; sex differences are folded in; the
other dataset's own detection threshold makes weak-edge conservation
a lower bound. The separation survives all three.

**What changed in the model.** Nothing — and that is the point: the
substrate every result stands on just went from assumption to
measurement. Part 2 (registered): port the rig to the male cord and
see whether the SAME parameters walk.

**Attribution.** Human: "btw another fly connectome exists... WE
should prolly run tests on both LOL" — the entire experiment, in one
message; also the memory-correction that motivated testing the
threshold's story. Agent: the download, the join, the analysis, and
the earlier wrong claim this experiment repaid.
