# Neuron, muscle and sensory model review

This folder exposes the parameters and implementation of the research walking model as checked on 19 October 2026. It is a source-review snapshot with component tests. The [September hackathon experiments](../README.md) remain the runnable end-to-end distribution; their treatment and flight results do not describe this newer configuration.

## Which model

| Configuration | Status | Parameters |
|---|---|---|
| `wm1`, 12 September | Formal walking champion of record | [champion-wm1.json](config/champion-wm1.json) |
| Build `7167953a`, retained 15 October | Current retained research model, including RL7's motor-neuron changes; not crowned as champion | [retained-build.json](config/retained-build.json) |

The research source is copied byte-for-byte from run commit `90872f6a70f44db3738c0c6830b167dae2686ad1`. The retained configuration comes from commit `4c2331df28c84f640b345920141cefdf2b9c2e1c`. Its runtime keys equal those of the tested RL7 arm, whose original SHA-256 is `8938ec2181f2665225ec73bcd4bc57dd25f666c5b7e6c8fff2cf91fad2ff5a49`. Only top-level underscore-prefixed research annotations were removed from the configuration. [manifest.json](manifest.json) records the source paths, hashes and transformations.

The formal title and research retention use different acceptance rules. RL7 was retained after a pooled two-block comparison; a title match stopped at a configuration-key gate on `e_inh_mv`. Retention is not a new title or proof of general biological fidelity.

## Where to read the implementation

| Question | Source entry point |
|---|---|
| What is a neuron, and what are its defaults? | [cns.py](source/cns.py): `SimParams`, `CNS.__init__`, `CNS.run` |
| How does theta become actual neural parameters and weights? | [walk_search.py](source/walk_search.py): `evaluate`, `size_factors`, `mn_reset_offset_by_id_apply`, `mn_exact_by_id_apply`, delay-vector functions |
| What code ran on the GPU? | [cns_metal.py](source/cns_metal.py), [cns_engine.m](source/gpu/cns_engine.m), [cns_engine.h](source/gpu/cns_engine.h) |
| Which motor neuron drives which joint? | [motormap.py](source/motormap.py), then `finalize_motor_matrix` in [walk_search.py](source/walk_search.py) |
| How do motor spikes move the body? | [body.py](source/body.py): `Body.step2`; [muscle_split.py](source/muscle_split.py): `StiffnessController`; `evaluate` in [walk_search.py](source/walk_search.py) applies the MuJoCo actions |
| How is sensory input encoded? | [body.py](source/body.py): `Proprioceptors.__init__` and `rates`; [load_loop.py](source/load_loop.py): contact loads; `evaluate` adds load, hair-plate and other input channels |

Source comments record the history of development. Some opening docstrings describe early defaults that the retained theta overrides. The active keys and branches take precedence. For example, the body currently uses second-order joint-target dynamics and searched gains, despite its early first-order, equal-gain description.

## Single-neuron model and parameters

The default neural unit is a single-compartment leaky integrate-and-fire model. The retained configuration enables conductance-based excitation and inhibition, shunting, adaptation and presynaptic delays. A simplified statement of the membrane update is:

`tau_m * dV/dt = V_rest - V + g_E*(E_E - V) + g_I*(E_I - V) - w + other configured currents`.

Threshold crossing produces an event, followed by reset and refractoriness. Adaptation follows `dw/dt = (a*(V - V_rest) - w)/tau_w` with a spike-triggered increment. These are reduced dynamics, without reconstructed dendritic compartments or a full ion-channel model. The engine also contains experimental alternatives which are inactive unless configured.

Selected cells use continuous voltage-dependent release: `graded_gain / (1 + exp(-(V - graded_v50)/graded_slope))`. Membership comes from [graded_map.json](source/graded_map.json), the [alignment-derived list](source/gpu/lamina_walk/s5585.json) and explicit theta additions. Delays, transmitter identity and connection multipliers are also configurable. An anatomical synapse count is not treated as a measured synaptic conductance.

Important shared theta values are `delay = 3.1200 ms` before type, region and row overrides, `adapt_a = 2`, `adapt_b = 1.04185`, `tau_w = 270.977 ms`, `e_inh_mv = -70`, and `bal_target = 0.43519`. These shared values do not describe every cell. The underlying defaults are in `SimParams`; `evaluate` applies theta and cell-specific changes. In particular, it starts chemical gain at `0.65` and multiplies it by `w_syn_scale`, rather than using the dataclass's `w_syn` default unchanged.

The [motor-neuron parameter table](motor-neuron-parameters.json) lists all 391 leg motor rows, their assigned classes and declared overrides. It is a join of theta with the retained study's [class map](source/gpu/b6890108/classes.json), not a new full-network parameter readback. Class assignments and transfer across cells are model assumptions; these are not 391 independently measured cells.

| Assigned class | Rows | Retained settings |
|---|---:|---|
| Fast | 84 | Rest −68 mV; membrane time constant 4.08 ms; reset at rest; adaptation increment 2.45 with 50 ms decay. The study's fast-class physiological target remains open. |
| Intermediate | 223 | Rest −60 mV on 210 rows; 13 retain −52 mV. Membrane constants 4.32–11.27 ms; reset 2 mV above rest; spike-triggered adaptation removed. |
| Slow | 84 | Rest −48 mV; membrane constant 13.1 ms; reset 18 mV below rest; adaptation increment 2.45 with 50 ms decay; an additional post-spike conductance and exact membrane update. |

The post-spike conductance parameters are explicit in `holddown_by_id`: for the slow class, increment 10, decay 8 ms, reversal at rest, with a 2 ms / 2 mV latch. Other cells have separate overrides. The configuration also includes selected electrical pairs, a scoped serotonin state and presynaptic inhibition. None amounts to a complete molecular or neuromodulatory annotation of the fly.

## Muscle and body model

Motor output is assigned to muscles and 42 leg joint axes using BANC peripheral-target annotations, model-specific sign conventions, secondary moment arms and explicit transfer corrections. Size-dependent motor gains and four muscle-specific gains are included in theta. This mapping has anatomical constraints and fitted quantities.

`Body.step2` filters positive and negative motor channels separately, with a 6 ms activation constant on all 42 axes in this build. Their difference sets a bounded joint target through `tanh`; their sum supplies the co-contraction signal. Joint-target dynamics use `f0_hz = 25.688`, `zeta = 3`, and the exact integrator. These are searched model settings.

MuJoCo position actuators then apply forces toward the targets. For eligible antagonist-controlled axes, `k_mode = "cocon"` changes stiffness with activation through `StiffnessController.apply`. `k_min = 163.158`, `k_gain = 560.248`, and `cocon_scale = 2`. A separate adhesion rule controls foot attachment. This is a phenomenological muscle-to-joint model with position servos. It does not contain measured force-length and force-velocity curves for each fly muscle or a detailed tendon model. Passive and controller dynamics can contribute to apparent gait timing.

## Sensory encoders and command

The femoral chordotonal organ, a leg position and movement sensor, is represented by claw, hook and club groups. Before optional filtering, the code uses clipped position-dependent claw rates, direction-selective rectified velocity for hook cells, and absolute velocity for club cells. Hair-plate channels respond near joint extremes. Default encoder constants are 5 Hz baseline, 120 Hz scale and 8 rad/s velocity scale. The retained theta uses BANC-type polarity and exact-type modality overrides, and applies its high-pass transform selectively through `phasic = 1.25` and `proprioceptor_phasic_mode`.

There is an important feedback distinction: `prop.rates(bod)` reads the reduced joint-target body's angles and velocities. Those variables are not a direct readback of all achieved MuJoCo joint angles. Campaniform input separately uses contact forces from MuJoCo, with rate `camp_gain * camp_match_factor * clip(load/F_ref, 0, 1)` in the retained `load_matched` mode. Here the gain product is approximately 276 Hz. `hp_dropped` adds another explicit hair-plate input channel; the name does not mean all hair-plate input is disabled.

The command is externally imposed DNg100 activity at 103.688 Hz. In this configuration, command events are forced spikes, so membrane inhibition cannot veto them. Their rate-to-event conversion uses `spike_reg = 44`, a regularized renewal process. The retained walking theta contains no visual stimulus or visual-controller configuration. Visual pathways and experimental decoders present elsewhere in the source are not evidence that this walking configuration navigates by vision.

## What has and has not been validated

[retention.json](evidence/retention.json) preserves the scientific portion of the recorded RL7 comparison. On seeds 219203–219226, both the prior body and RL7 stood and qualified on 24/24 runs. The pooled objective changed from 10 to 9 open items because motor-class targets improved while the counted gait conflicts increased from 7 to 8. Pooled simulated speed decreased from 2.5001 to 2.0185 mm/s. Footfall separation worsened beyond its registered band. These are historical simulated results, not reruns performed for this export.

The current research agenda is to obtain rest without command and command-dependent start/stop. Prior bodies walked with no command and after a neck cut. The 16 October study opening explicitly marked the newest `7167953a` body unread on its headless floor and air-rest tests. Its retained walking card does not resolve those failures. A successful walking video therefore does not establish appropriate sensory control, causal necessity of the real wiring, or accurate animal emulation. Those require matched interventions, trained null models and independent measurements.

## Check this snapshot

From the repository root:

```bash
python3 model-review/inspect_model.py
uv sync --locked
uv run python model-review/test_components.py
uv run python -m unittest discover -s tests -v
```

The first command verifies snapshot hashes and reports model settings without downloading data. Component tests exercise neuron dynamics, antagonist activation and sensory encoding on small synthetic fixtures. They do not validate the whole animal or reproduce the 24-run retention result.

[Export checks](evidence/export-checks.json) record the checks performed for this publication: six component/integrity checks, the eight existing hackathon tests, source/input-configuration identity, documentation links and a native host-library compilation. Native GPU execution and whole-network numerical parity were not tested in this export.

This folder is a selected source export, not the complete research runner dependency tree. It includes the core neural, motor, sensory and coupling implementation plus active annotation inputs; optional research rigs and private database services are not packaged. `walk_search.evaluate` is supplied for inspection. A standalone end-to-end replay of this October build, including complete inputs, run wrapper and native-engine verification, remains separate work. Use the original repository commands for the runnable September experiments.

Original Uploading Lab code uses the repository's MIT license. BANC-derived identity and annotation inputs retain CC BY 4.0. Body runtime dependencies retain their upstream licenses. See [THIRD_PARTY.md](../THIRD_PARTY.md).
