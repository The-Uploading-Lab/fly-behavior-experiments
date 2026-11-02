# Publication evidence and reproduction

This package accompanies the review draft *Construction and retrospective evaluation of an embodied Drosophila connectome model*. It contains historical result rows, executable reanalysis, the September delay simulator and a later evaluation supplement. The manuscript is still under scientific review. The records describe several frozen builds; their results cannot be combined as validation of one model.

## Recalculate the reported comparisons

From the repository root, using Python 3.12:

```sh
python3 publication/analyse.py
```

This command needs no third-party libraries, data download or credentials. It verifies the history and evidence manifests, then writes `outputs/publication/summary.json`. It recomputes the paired muscle standing counts, the exact discordance test, all six delay-arm speed medians, paired speed differences, calcium feature errors and the walking speed-bin support check.

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

The manuscript's independent replay used the Metal backend. CPU random-number generation and numerical differences can change trajectories, so successful execution of this portable release is not a claim of identical CPU and Metal outcomes. The archived replay rows remain the source for its numerical tables. The included historical Metal adapter is part of the unchanged import closure; the supported launcher selects NumPy and does not build or require the native Metal library. The `delay-model` files implement many historical optional modes; only this frozen delay experiment is the supported entry point in this export.

For the separate September hackathon experiments, follow the [root README](../README.md). The [October model review](../model-review/README.md) describes a later implementation. Its component tests do not reproduce every whole-network experiment in that later build.

## What the evidence supports

| Comparison | Available evidence | Interpretation limit |
|---|---|---|
| August muscle interface | 60 stored rows, both configurations, paired reanalysis | 16/30 versus 8/30 standing; command identity and several parameters changed with stiffness |
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
