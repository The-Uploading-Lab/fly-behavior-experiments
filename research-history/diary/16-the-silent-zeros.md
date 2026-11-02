---
id: 16
date: 2026-08-19
title: The zeros that were never measurements
tldr: A search we had cited as evidence turned out to have crashed 204 times out of 204, printing a plausible zero each time. The experiment that asks whether the connectome matters at all was silently running the real brain in two of its three scrambled controls. The network cache did not know which brain it held. And the video parity rule — the one that has caught a wrong render before — was a printed number nobody compared to anything. Thirty-nine agents went looking; none of it changed a measured walking number, and all of it was about to.
era: standard
born: a rule that exists only in prose cannot protect a run
commit: arc 1f13188..HEAD, 2026-08-19
theta: results-f23-theta.json throughout (nothing about the fly changed)
seeds: null-program probe seed 0; cache-collision probe seed 0; F29/F30 chamber seeds as recorded
n: 42 agents across two adversarial sweeps; ~50 further suspects logged unverified
wall: 2026-08-19, all day
reproduce: .venv/bin/python3 eval_job.py --params @/tmp/job.json   # vary "shuffle"
---
**Question.** This project has been burned repeatedly by code that
produced plausible results while doing something nobody intended. After
the fifth such burn we wrote down the heuristics that caught them. The
question is whether those heuristics were ever anything more than
folklore: **how many silent zeros are in here right now?**

**Predicted (pre-registered, quoted).** The sweeps were told to assume
every finding was wrong until reproduced, and that "the commonest failure
mode in this kind of audit is a finding that is real in the code but
unreachable in practice." Both sweeps were required to report what they
checked and found **clean**, on the grounds that an audit which only
lists faults implies a coverage it did not achieve.

**Method (jargon defined inline).** Two adversarial passes, 42 agents,
every finding re-measured from scratch by a second agent whose brief was
to refute it. The first pass audited the **annotation** layer — what is
empty in each dataset and what the simulator silently substitutes. The
second hunted **silent-failure classes**: dead dials (a parameter that
changes nothing is disconnected, not unimportant), cache collisions,
selections that match zero rows, numeric rot, instrument faults, and
provenance — every constant we call "measured" checked for having been
transformed on the way in.

**Result (numbers, with variance).**

**F29 never ran.** A chamber whose result we had put in the constraint
registry as "0 of 204 candidates reach the coordination gate — not
parameter-selectable", and which one of us had cited that same morning as
evidence that coordination is mechanism rather than parameters. All 204
evaluations raised the same error: the coordination function was imported
inside one helper and referenced inside another, so the gate line could
never see it. Each crash printed `[n/204] score 0.00 pass 0/3` — character
for character what a legitimately non-scoring candidate prints. Withdrawn.
Every other chamber checked and clean: 324/324, 132/132, 132/132,
204/204, 204/204, 204/204.

**The null program was never shuffling.** `eval_job.py` drives the
experiment that asks whether the connectome matters at all — real brain
against scrambled ones. It built the shuffled network, filed it in the
cache under a **float** key, and the simulator looked its networks up
under a **tuple** key. The lookup could never hit, so the simulator
rebuilt — from a fresh, *unshuffled* re-read of the edge file. Two of the
three controls were the real brain, bit-identical to sixteen figures:

    shuffle   before the fix        after
    none      0.9184682749530796    0.101554076097
    post      0.9184682749530796    0.021690900854
    weight    0.9184682749530796    0.008671598766
    sign      0.4334815501434155    0.142889974922

The `sign` arm survived only by accident — it scrambles a different
argument, which does get passed through. Run as shipped, this would have
reported that scrambling the connectome changes nothing: **a manufactured
refutation of the project's central claim.** Nothing had run on it yet.

The root cause was one layer down. The simulator's network cache key was
built from **parameters only** — the neuron table and the edge list, the
two things the network is actually built from, appeared nowhere in it.
Hand one cache two different brains and the second silently gets the
first one's network: measured at 0.918 for both, where a fresh cache
gives 0.847 for the shuffled one. Six scripts had each defused this by
hand, one of them carrying the comment *"the cache key inside evaluate
does not know the connectome, so give each arm its own dict."* Four
authors disarmed the same landmine one at a time instead of removing it.

**The parity rule was a ritual.** Every video is supposed to reproduce
its recorded displacement before it ships. The renderer computed that
number on one line, printed it on the next, and never referenced it
again — no reference value, no comparison, no exit code — while the
publisher's docstring claimed it ran "right after the parity check". The
one time this caught a wrong render (2.12 mm against a recorded 7.55), it
was caught by a human reading a number off a screen. Related, and now
tracked: the renderer is a hand-copy of the drive path and is **fifteen
mechanisms behind it**.

And one of the nine was shipped the same day it was found: a new chamber
wrote its result under the *previous* round's filename, and crowned its
winner on uprightness **alone** — height was never even stored. The
crowned candidate was reported 4/4 standing and is **0/4** under the
project's own three-clause gate, heights 0.91, 0.65, 0.74, 0.92.

**Interpretation.** Not one of these changed a measured walking number.
The champion scores, the coordination headline, the title cards all
reproduce. What was wrong sat a layer above the numbers — in
interpretations, in control premises, and in three experiments that had
not yet run. That is the pattern worth naming: **a silent zero rarely
corrupts the past; it lies in wait for the most important thing you have
not done yet.** The null program is the experiment this project exists to
survive, and it was the one carrying the worst defect.

**Problems / what bit us.** Three of our own written rules would each
have caught F29 alone — *all-zero is infrastructure failure, never
behaviour*; *never grep a battery card, read it whole*; *a result that
flatters the project deserves the most suspicion* — and none fired,
because none of them was in the code. The same lesson arrived twice more
the same day. The diary's own cap of three entries a day existed only in
conversation, and the moment nobody was watching it produced eight. And a
set of runs an audit flagged as silently void turned out to have been
**pre-registered as sham controls** and reported as such — the finding
was overstated and was withdrawn, but the narrower defect stood: nothing
in the code could tell a deliberate sham from an accidental one, because
the registration lived in prose the rig never reads.

**What changed in the model.** Nothing in the fly. The network cache key
now carries the connectome's identity. The null program hands the
simulator shuffled *inputs* instead of pre-building. Parity is compared
and exits non-zero, and publishing refuses without it. A chamber that
scores nothing refuses to crown anything. The simulator announces what it
selected, once per process, and asserts that inhibition exists — and,
caught within the hour, announces it on **stderr**, because the first
version corrupted a channel another tool parses as JSON. An announcement
that breaks a machine-readable channel is its own silent failure.

**Attribution.** Robin asked the question that started it — "check for
any other similar problems we might have overlooked since I haven't
actually looked at our data or models" — and it is the reason two of the
three most dangerous defects in this entry were found before they ran
rather than after they published. The agent built the sweeps, found one
of the nine to be a bug it had shipped that afternoon, and withdrew its
own overstated finding when the pre-registration turned up.
