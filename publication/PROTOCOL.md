# Controlled reproduction protocol

Protocol written on 3 November 2026 before executing the controls. The frozen September qh1 walker is the experimental model. This new comparison does not reproduce the confounded August muscle comparison or the October inhibitory replacement study. Those historical result sets retain their separate provenance.

## Question and endpoints

The question is how the selected model changes when stiffness modulation, delay assignment, inhibitory weight normalization or neural motor drive is altered. All arms retain the same command, motor map, neural parameters, actuator force limits, initial state, six-second duration and paired noise seed, except for the named intervention. No parameters are refitted. This tests conditional sensitivity in one selected model. It does not compare optimally fitted model families or establish physiological validity.

Standing under the archived `standing_v32` definition and post-transient net displacement are the main descriptive endpoints. All trials, including falls, remain in the output. Speed and gait fields from the historical battery are conditional on standing and must not be averaged after treating missing values as zeros. Mechanical diagnostics include sustained posture loss, displacement before that loss, target tracking error, actuator saturation, contact-tip motion and sampled actuator work. Definitions are in `mechanical_metrics` in `reproduce.py`. Work uses FlyGym model units and control-tick sampling; it is not metabolic energy or a validated efficiency estimate. Contact-tip speed includes motion of the lowest selected tip geometry, which can change identity between ticks.

## Arms

| Arm | Intervention | Purpose |
|---|---|---|
| `base` | Original load-dependent stiffness, delay vector and inhibitory normalization | Shared paired reference |
| `stiffness-replay` | Replay the reference stiffness sequence at its original ticks | Identity control for instrumentation and replay; mechanical arrays must match exactly |
| `stiffness-mean` | Fixed per-joint temporal mean of reference stiffness | Remove modulation while matching each joint's mean gain |
| `activation` | Switch load-dependent stiffness to the implemented antagonist-activation law | Compare implemented regulation laws with other parameters fixed |
| `stiffness-shift` | Circularly shift the reference stiffness sequence by a seeded offset of 25–75% of its duration, shared across joints | Retain each joint's gain distribution and circular temporal structure while breaking alignment to the ongoing loads |
| `delay-uniform` | Every presynaptic delay equals the original vector's exact mean | Remove delay dispersion without changing its mean |
| `delay-shuffle` | Permute the original delay vector across neurons with an explicit assignment seed | Preserve the complete delay distribution while breaking its association with neuron identity |
| `inhibition-unscaled` | Set `bal_target=0`; retain transmitter signs and baseline inhibitory weights | Remove the static postsynaptic inhibitory rescaling |
| `motor-zero` | Set both leg antagonist drives and abdominal neural drive to zero; retain body mechanics and load feedback | Disconnect neural motor output and quantify passive/mechanical movement |

The muscle controls preserve the two-channel motor interface. Setting `k_gain=0` in this historical implementation would also disable that interface, so it is not used as the fixed-stiffness control. Mean and shifted stiffness use the reference trial's full trace, including future samples: they are retrospective interventions, not deployable feedback policies. The activation comparison does not match mean stiffness; the separate mean control addresses that confound for the load law. A temporal null that performs equally well or better remains a result.

## Replication units and execution

One neural noise seed forms a paired block. Assignment seeds specify independent delay permutations and stiffness time offsets. Shared base, uniform and other deterministic controls are executed once per noise seed, not repeated and counted as new observations for each assignment. A time shift can repeat at different seeds because the allowable offsets are finite; actual offsets are saved. Simulation seeds are not animals.

```sh
uv sync --locked
uv run python publication/reproduce.py
```

The default executes nine full six-second trials on one noise seed (201303) and one assignment seed (9001). It is an execution pilot with no population inference. Outputs are in `outputs/publication/controls/`. Existing outputs are never overwritten. Approximately 417 MB of public data are downloaded once and checked against `experiments/data.json`. The CPU backend runs serially; a previous base trial took about 111 seconds on the release machine. A GPU is not needed. Mac execution is tested; Linux headless rendering follows the root installation instructions.

An expanded exploratory comparison can be run explicitly:

```sh
uv run python publication/reproduce.py --seeds 201303 201304 201305 201306 201307 201308 --null-seeds 9001 9002 9003 9004 --out outputs/publication/controls-expanded
```

This executes 90 trials, approximately three CPU hours at the previously timed base cost. Do not infer that this larger comparison has run from the availability of the command. Each trial saves the full parameter set, body and neural traces, applied stiffness, delay vector, inhibitory factors, motor drive, joint targets and achieved angles, actuator forces and limits, log, runtime and output hashes. The summary records dependency versions, source hashes and status. A failed invariant or interrupted run remains marked failed, with completed rows preserved.

## Analysis and limits

Report each trial and paired differences against its matching base seed. Keep assignment variability separate from neural noise variability. The pilot has no significance threshold, confidence interval or claim of superiority. A later confirmatory study needs a fixed sample size informed by pilot variability, independent evaluation targets, delay-by-inhibition interactions and equal-budget refitting of competing model families. Retain null outcomes and controls that outperform the original.

The archived August, October and calcium tables can be recalculated using `publication/analyse.py`; this is distinct from regenerating their simulations. The release must not describe those historical pipelines as fully reproduced until their source, data, environment and execution receipts are supplied.
