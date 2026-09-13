# Global neuronal aging experiment, 13 September 2026

Owner: Robin, latest live instruction in the second hackathon lane. Run
aging across all neurons at days 0, 10, ..., 90. Deliver walking and escape
statistics first, then escape videos at days 0, 30 and 90. This authorizes
this bounded hackathon experiment after the earlier general research stop.
It does not resume lanes A–E or allocate their seeds.

## Model fixed before running

Baseline candidate: fixed-native-walk-visual125-lplc2-quarter-complete-ttm,
successful source commit 4cd49ae74223b6616c73466e3e71f277527fd52d. The
current analysis checkout begins at 65fcf0b5d0e2133963c3456bfad1524b2de0844f.
Candidate and input hashes are frozen in freeze.json before execution.

The new hypothesis is **global synaptic efficacy decline**:

    q(d) = 2^(-d / 45)
    W(d) = q(d) W(0)
    Wgraded(d) = q(d) Wgraded(0)
    G(d) = q(d) G(0), degree(d) = q(d) degree(0)

Every one of the 188,508 represented neuron IDs receives the same q.
W contains signed chemical outputs, including inhibitory outputs. G
contains the represented electrical connections. Scaling occurs after
normalization; intrinsic cell parameters, anatomical counts, topology,
input rate, muscle strength, geometry and flight controller stay fixed.
Zero age takes the unchanged code path and must reproduce baseline events.

The **45-day efficacy half-life is assumed**, not inferred from animal
behavior or calibrated to a neural measurement. Day 30 gives q=0.629961;
day 90 gives q=0.25. Model days are an explicit testable day-to-parameter
hypothesis. The study cannot establish a biological aging clock, survival
to day 90, or animal population uncertainty. These are freshly initialized
age-parameter snapshots, not a continuous 90-day physical simulation.

Why test it: declining transmitter release has been observed at selected
adult motor terminals (Banerjee et al., 2021,
https://doi.org/10.1038/s41467-021-24490-1). That supports examining synaptic
function but does not establish this global curve. The strongest opposing
evidence is selective vulnerability in the escape circuit: many components
remain functional while LC4-to-GF transmission declines (Gaitanidis et al.,
https://pmc.ncbi.nlm.nih.gov/articles/PMC12707685/). Early brain-wide active
zone increases and compensation are also reported (Huang et al.,
https://pmc.ncbi.nlm.nih.gov/articles/PMC9721493/). Thus uniform weakening is
a simplifying hypothesis that a future experiment can reject, not a claim
that every real neuron ages equally. No neuron deletion or molecular-aging
mechanism is asserted.

## Fixed runs and ownership

Use owned development seed 0 only, paired across ages and protocols. No
new nonzero or held-out seed is allocated. Each age has one walking case,
one mechanical-input case and one no-input control. These give per-case
statistics, not a reliable estimate of population escape probability.
The baseline is selected development evidence. Existing wet-lab values
were seen in the previous comparison; no future age-series animal outcomes
exist here and no parameter selection uses those earlier outcomes.

Walking duration: 6,000 ms. Mechanical escape: 3,000 ms. Sham: 1,100 ms
(existing longer day-zero sham may be reused). Pulse is 450 Hz on JO-A and
JO-B, duration 30 ms, onset 600 ms. It is an antennal-input proxy, not a
calibrated box tap. Capture poses and neural activity during these exact
runs. Later films replay captures without changing outcomes.

Day-zero puff with the wrapper is the behavior-neutral parity check.
Day-zero walking and sham may reuse saved complete-model captures after
source/parameter verification. An old 6 s walking prefix is not another
trial. Each requested age remains in the output even after standing loss;
no additional seed or tuning is used to repair failure.

## Readouts fixed before outcomes

Walking: apply the existing both-body standing gate first. Censor speed,
path and gait after standing failure, retaining the case and failure.
For eligible trials use displayed thorax xy from 0.300 to 5.990 s, including
pauses. Linearly sample saved poses at 120 Hz, apply a five-frame median
filter (reflect edges), sample path every three frames and include the
final endpoint. Speed is path / full observation duration in mm/s; net
displacement and straightness are also reported. This is a temporal
processing approximation on ~119 Hz saved poses, not 240 Hz measurement.
Internal-body speed is separately labelled. Fixed 1 s windows describe
within-trial temporal variability; they are not independent trials.

Escape: report first GF spike after 600 ms, first TTM event after input,
native support-loss/departure time, and departure by 416.666667 ms after
input. Require displayed support and uprightness in the 100 ms before
input for the eligible reflex denominator. Preserve pre-input departures,
GF-free motor activity, no departure, crashes and later clearance failures.
No departure is right-censored, never assigned a latency of 416.7 ms.
GF latency is not observable movement latency. Sham has the same horizon.
Flight clearance and final posture are descriptive because wing power,
attitude, leg folding and landing retain programmed support.

Directional hypothesis: weakening transmission can reduce supported walking
and delay or abolish escape. Inhibitory and excitatory outputs both weaken,
so nonmonotonic movement, false triggers or falls also remain possible.
Do not force a monotonic curve. Flat outputs require checking application
hashes before biological interpretation. Failures are results, not reasons
to change q, stimulation or controller settings.

## Execution

One dedicated self-closing tmux session; one own engine maximum; host cap
two, fresh admission before launch and between trials. Retain the existing
NumPy exception for gap junctions and per-neuron input regularity. Time the
first full case; local Metal lacks these model features, so no accelerator
port or unqualified backend is introduced. No public scientific acceptance
or champion nomination is requested by this experiment.
