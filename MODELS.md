# Models and interventions

## Shared fly

The baseline is `fixed-native-walk-visual125-lplc2-quarter-complete-ttm`, selected in the hackathon development work. Its original successful source commit is `4cd49ae74223b6616c73466e3e71f277527fd52d` in [fly-wbe](https://github.com/The-Uploading-Lab/fly-wbe). The packaged configuration is [here](simulator/lanes/longevity/ageing/candidate-fixed-native-walk-visual125-lplc2-quarter-complete-ttm.json).

BANC v888 supplies neuron identities and anatomical connectivity. The represented network contains 188,508 neurons, 13,620,865 directed edges and 42,316,236 retained synapses. It uses conductance-based integrate-and-fire neurons with adaptation, synaptic delays, selected graded-release cells and explicit electrical connections. These physiological parameters and connection-strength mappings are model assumptions; a structural connectome alone does not determine them.

One neural network is coupled to two bodies. The internal walking body supplies sensory feedback. Its neural motor output drives the displayed winged body. Joint-target transfer connects them; each body's standing is checked. The displayed body uses FlyGym/FlyBody geometry and MuJoCo physics. Native giant-fibre/TTM events initiate jumping, including the continuing muscle twitch after takeoff. Wing power, wingbeat pattern, attitude support, leg folding and landing retain programmed control. A departure therefore does not establish an entirely neural flight controller.

Every reproduced case uses development seed 0. The cases were selected during model development; repeated time windows and ages are not independent animals or held-out validation.

## Ibuprofen: Karolina's A2 adaptation mapping

The treatment changes all represented neurons' adaptation parameters through the same runtime path used for the original comparison:

| Parameter | Control | Ibuprofen |
|---|---:|---:|
| `adapt_a` | 2.0 | 1.0 |
| `adapt_b` | 1.0418521601266768 | 2.0 |
| `tau_w` (ms) | 270.97700471360525 | 100.0 |

The adaptation state follows `dw/dt = (a*(V - V_rest) - w)/tau_w`, increases by `b` on spikes, and subtracts from membrane drive. The mapping lowers subthreshold adaptation, raises the spike-triggered increment and shortens its decay. All other model parameters are fixed. The runner verifies the actual adaptation values on all 188,508 neurons before stepping the network.

This is the supplied phenomenological neuron-function hypothesis. It has no ibuprofen concentration, absorption, molecular target or exposure-time model. Its output is a prediction under this parameter mapping, not evidence that the same mapping occurs in the experimental flies.

Implementation: [apply_intervention](simulator/lanes/longevity/production/karolina_experiment.py), runtime application audit in each run's `manifest.json`.

## Calorie restriction / starvation: Karolina's arm-B circuit hypothesis

Chemical LC4 and LPLC2 input onto both giant fibres, `DNp01`, is multiplied by **0.5 relative to the selected baseline**. Canonical neuron IDs define the source and target populations. This composes with the baseline's existing connection gains. Intrinsic neuron parameters, antennal input and muscle parameters remain fixed.

The mapping represents a proposed change in visual escape transmission. It contains no food, caloric budget, metabolism, body-weight loss or hours-since-food-removal dynamics. The wet-lab label is calorie restriction; this circuit perturbation is only a mechanism scenario for comparison. The experimental hand tap has visual and mechanical components, while the simulated escape protocol uses antennal input. An unchanged simulated tap response can therefore follow from a mismatch between the perturbed pathway and the tested input.

## Aging: global synaptic-efficacy decline

At model day `d`, `q(d) = 2**(-d/45)`. After network normalization, the aging wrapper scales signed chemical weights `W`, graded-release weights `Wg`, and electrical weights `G` by `q`. Electrical degrees and pair weights receive the same scaling. Both excitatory and inhibitory outputs weaken. Topology, neuron count, intrinsic parameters, inputs, muscle strength and programmed flight control are fixed. Day zero takes an unchanged numerical path.

The **45-day half-life is assumed**. Thus day 45 assigns efficacy 0.5, and day 90 assigns 0.25. Each age starts a fresh simulation; no 90-day physical trajectory or mortality process is simulated. This model tests uniform weakening, without cell-type-specific vulnerability or compensation. The other hackathon lane's [frozen protocol](experiments/aging-protocol.md) records the motivation, opposing evidence and fixed readout rules.

Implementation: [global_ageing.py](simulator/lanes/longevity/ageing/global_ageing.py). Per-run manifests record neuron coverage and weight-array hashes before and after the change.

## What the outputs compare

Walking in the wet-lab comparison uses **projected fitted silhouette lengths per second** on both sides. Model poses are projected with a fixed overhead orthographic camera. A silhouette's length is four times the square root of its largest pixel-coordinate covariance eigenvalue. A five-frame median filter and a six-frame path stride at 240 Hz match the wet-lab estimator; strides 3 and 12 report processing sensitivity. Each model window starts at 0.300 simulated seconds and matches the corresponding measured interval's duration. Camera pose and mask quality still differ between the two sources. Physical mm/s cannot be assigned to the animal recordings without a scale.

The separate aging series reports displayed-body path speed in mm/s over 0.300–5.990 s, using its frozen 120 Hz sampling, median filter and three-frame stride. Its within-trial window SD is temporal variation, not population uncertainty. Both-body standing failures censor speed and gait.

The simulated escape input is a 450 Hz, 30 ms JO-A/JO-B antennal pulse beginning at 600 ms. Giant-fibre latency is a neural output; native support-loss latency is a body output. Neither is the same endpoint as the wet-lab frame interval for visible departure. Measured nondepartures are retained as right-censored observations through 416.666667 ms. No p-values or animal treatment-effect estimates are computed from selected clips with missing fly identities and denominators.

## Portability changes

The portable copy replaces research-fleet Git-reference checks with a local source/input hash manifest. It permits importing the unused Metal wrapper on Windows; execution remains CPU NumPy because this model uses gap junctions and per-neuron input regularity. A standalone capture wrapper and portable paths replace the machine-specific launch and delivery wrappers. Scientific model equations and the selected configuration are preserved. Original hashes and tested numerical comparisons are in `provenance/`.
