# Publication evidence and reproduction

This package accompanies the review draft *Construction and retrospective evaluation of an embodied Drosophila connectome model*. It contains the frozen simulator, muscle interface, executable null controls, historical result rows and reanalysis. The manuscript is still under scientific review. The records describe several frozen builds; their results cannot be combined as validation of one model.

The [claim-to-command map](REPRODUCIBILITY.md) distinguishes complete simulation routes, recalculation from saved observations and pipelines still needing export.

## Run the muscle model and null controls

```sh
uv sync --locked
uv run python publication/reproduce.py
uv run python publication/analyse_controls.py outputs/publication/controls
```

This runs nine paired six-second trials: the original model, an identity replay, fixed mean stiffness, activation-driven stiffness, shifted stiffness, uniform delays, shuffled delays, disabled inhibitory rescaling and zero neural motor drive. Both leg and abdominal neural drive are removed in the motor null. The fixed-stiffness arm preserves the two-channel antagonist interface. See the [frozen protocol](PROTOCOL.md) for intervention definitions, endpoints and expanded seed commands.

Every case saves its full configuration, simulated body and neural traces, applied stiffness and delay vectors, inhibitory factors, motor drive, achieved and target joint angles, actuator forces, limits and checksums. The analyser rejects missing cases, changed outputs or incomplete runs. It reports paired displacement changes for all cases, including falls. The default is a one-seed execution pilot, not a statistical claim about flies or model superiority. Null controls that equal or exceed the original remain in the results.

Run the smaller implementation checks without downloading the connectome:

```sh
uv run python publication/test_reproduce.py
uv run python publication/timing_fixture.py
```

These check the actual muscle laws and actuator equilibrium, inhibitory rescaling, null transformations and a two-neuron delay circuit. The delay fixture tests 64 combinations of timestep, source delay, target delay and queue/bucket delivery; its output is `outputs/publication/timing-fixture.json`. These are implementation checks, distinct from the whole-model null experiments.

## Recalculate the reported comparisons

From the repository root, using Python 3.12:

```sh
python3 publication/analyse.py
```

This command needs no third-party libraries, data download or credentials. It verifies the history and evidence manifests, then writes `outputs/publication/summary.json`. It recomputes the paired muscle standing counts, the exact discordance test, all six delay-arm speed medians, paired speed differences, calcium feature errors and the count of saved walking speed-bin support decisions. It also reconstructs the October inhibitory-arm standing counts and paired discordances from saved failure-seed identities; those speed medians remain quoted archive summaries.

To generate the two manuscript figures:

```sh
uv sync --locked
uv run python publication/plot.py
```

The PNG, SVG and PDF figures are written to `outputs/publication/figures/`.

## Run the historical delay simulator

The repository's pinned dependencies and public BANC data are used. The download is approximately 417 MB, checked against SHA-256 digests in `experiments/data.json`. The source is reconstructed at development revision `88fc005eecd0e181f44f7e828750605d36ef8745`. The supported entry point runs the NumPy CPU backend, so no GPU or lab credentials are required.

```sh
uv sync --locked
uv run python publication/replay_delay.py --arm base --seed 201303
```

The command creates a separate working copy under `outputs/publication/delay/base-201303/`, runs six simulated seconds and writes `result.json` plus compressed body and neural traces. It refuses to overwrite an existing trial. Other arms are `flat6`, `flat15`, `noshort`, `nolong` and `perm`; the archived comparison uses seeds 201303 through 201308 in every arm. Each invocation is one trial. The default takes substantially longer than six wall-clock seconds and loads the whole connectome.

The manuscript's independent replay used the Metal backend. CPU random-number generation and numerical differences can change trajectories, so successful execution of this portable release is not a claim of identical CPU and Metal outcomes. The archived replay rows remain the source for its numerical tables. The included historical Metal adapter is part of the unchanged import closure; the supported launchers select NumPy and do not build or require the native Metal library. The `delay-model` files implement many historical optional modes; the frozen delay experiment and controlled reproduction protocol above are the supported entry points in this export.

## Reconstruct the August muscle comparison

```sh
uv run python publication/replay_muscle.py --seeds 46
```

This runs both original configurations on one shared seed. To reconstruct the complete 60-trial comparison, use `--seeds 46-75 --out outputs/publication/muscle-history-all`. The output records the four original metrics, archived values and their numerical differences for every trial. Exceptions stop the run and remain failures; they do not become zero-valued observations. A run refuses to overwrite existing output.

The eleven source modules are unchanged from `4f024efc52119717c1add15517e175b5b00fde7d`; results were committed at `57289be60c147c8d109e8b3cefb78d4a55a50c42`. The joint-range input was initially untracked and was recovered and first committed on 30 August. Its later source and digest are explicit in `muscle-model/manifest.json`. The inspected Git history supplies no 11 August range-file hash; original environment backups are an additional route for checking that identity. The current pinned environment is also later than the original experiment. The launcher therefore compares computed outcomes with the archive instead of assuming numerical equality.

This historical experiment changed command identity and several other parameters together with the muscle interface. Its replay cannot isolate a stiffness effect. The September protocol above supplies matched interventions on a separate frozen model.

For the separate September hackathon experiments, follow the [root README](../README.md). The [October model review](../model-review/README.md) describes a later implementation. Its component tests do not reproduce every whole-network experiment in that later build.

## What the evidence supports

| Comparison | Available evidence | Interpretation limit |
|---|---|---|
| Controlled September muscle, delay and inhibition interventions | Executable paired protocol, raw-output export and integrity-checked analysis | One frozen configuration; no refitting or independent-animal validation |
| August muscle interface | 60 stored rows, original source and configurations, serial replay launcher, paired reanalysis | 16/30 versus 8/30 archived standing; command and other parameters changed; recovered range input and later environment limit historical equivalence |
| September delay assignment | 36 independently replayed rows and historical source | One selected model, six noise seeds and one shuffled assignment; 28/36 trials matched five original outcomes exactly |
| October inhibitory scaling | Extracted 24-seed arm summaries and parameter definitions | Summaries were not independently rerun; speed is conditional on standing |
| H4 visual calcium | Saved model and animal summary features | Exposed development data, a separate model, no raw fluorescence trace error or independent-animal inference |
| Walking observation | Per-trace observer diagnostics | No simulated run supplied half a second within one reference speed bin; no gait-accuracy ranking |

`manifest.json` records hashes and any export transformations. Original absolute paths in numeric metadata were reduced to basenames, and resource-admission logs were excluded. Numerical rows were preserved. `delay-replay-summary.json` retains the earlier comparison against archived outcomes; `analyse.py` independently recomputes the new-replay summaries from the released rows. The original run environment and raw traces for every other historical study are not part of this export. Recreating the calcium model or the October inhibitory study requires further source and data packaging. The available routes are the cited original datasets, the archived study configurations and the October source review; those pipelines are not represented here as complete reproductions.

## Source datasets and software

BANC anatomy: Bates et al., *Distributed control circuits across a brain-and-cord connectome*, [Nature 2026](https://doi.org/10.1038/s41586-026-10735-w); [data deposit](https://doi.org/10.7910/DVN/7WTH1N), CC BY 4.0. Public input hashes are in `experiments/data.json`.

Walking observations: DeAngelis, Zavatone-Veth and Clark, [eLife 2019](https://doi.org/10.7554/eLife.46409); [Dryad data](https://doi.org/10.5061/dryad.3p9h20r). Only derived observer summaries are included here.

Calcium observations: Ketkar et al., [eLife 2022](https://doi.org/10.7554/eLife.74937). The released numbers are source-normalized summary features used in earlier model development. L2 response direction and the L1 observation model remain interpretive limitations.

The model uses FlyGym and MuJoCo. Existing dependency license notices and vendored compatibility code are documented in [THIRD_PARTY.md](../THIRD_PARTY.md) and `provenance/dependency-licenses.json`. Project-authored code and records follow the repository license; upstream data and software retain their original licenses. The private scientific-review manuscript and meeting notes are not distributed in this package.

## Release checks

The [verification receipt](verification.json) records a fresh clone and dependency installation, public input downloads, all eight repository tests and all six model-component tests. The reanalysis and figure commands completed. A six-second NumPy trial also completed in each of two installations, with identical body and neural trace hashes. Its speed was 0.9413 mm/s; the archived Metal result for the same nominal seed was 1.0226 mm/s. Only this one CPU arm and seed was rerun for the export. The source commit tested and command receipts are recorded explicitly.

## Dependency security status

GitHub reported 82 open alert entries covering 41 distinct Pillow advisories on 2 November 2026, including [an image-decoding advisory](https://github.com/python-pillow/Pillow/security/advisories/GHSA-5x8w-8jj5-q52h). The existing reproduction environment pins Pillow 12.3.0. The reported fixes require 13.0.0; a resolver check rejected that upgrade because FlyGym 2.1.0 requires Pillow below 13. A compatible FlyGym/Pillow migration and regression checks remain separate work. The standard-library numeric reanalysis command does not install these dependencies.
