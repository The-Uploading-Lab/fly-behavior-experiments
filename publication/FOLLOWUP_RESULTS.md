# Replicated frozen-model control results

All 138 prespecified six-second simulations completed on 3 November 2026. The [full analysis](followup-analysis.json) passed complete-grid, case-record, output-hash, environment and control-identity checks. The [verification receipt](followup-verification.json) records the source pin, scheduler completion and archive checks. The model and analysis source were frozen at `c855d05c5e3656ee70e2333d46e538de5af9f2e4` before execution.

The design uses six noise seeds crossed with four null assignments. Fixed arms have six trials and randomized arms have 24 trials, with shared references. These are simulation realizations, not animals. All outcomes are retained, including falls. The [protocol](FOLLOWUP_PROTOCOL.md) fixed the endpoints, sample, stopping rule and exploratory bootstrap before the results were available. The earlier Mac pilot is separate.

| Arm | Late posture | Mean net displacement mm | Mean paired change mm [95% interval] |
|---|---:|---:|---:|
| `base` | 5/6 | 6.166 | reference |
| `stiffness-replay` | 5/6 | 6.166 | 0.000 [0.000, 0.000] |
| `stiffness-mean` | 6/6 | 5.059 | -1.107 [-2.375, 0.118] |
| `activation` | 6/6 | 5.069 | -1.097 [-3.374, 1.097] |
| `stiffness-shift` | 23/24 | 4.830 | -1.336 [-3.229, 0.400] |
| `stiffness-shift-after-startup` | 21/24 | 4.908 | -1.258 [-3.091, 0.455] |
| `delay-uniform` | 5/6 | 2.958 | -3.208 [-4.517, -1.912] |
| `delay-narrow-clock-mean` | 23/24 | 2.801 | -3.365 [-5.177, -1.851] |
| `delay-shuffle` | 23/24 | 2.328 | -3.838 [-5.143, -2.619] |
| `inhibition-unscaled` | 0/6 | 2.305 | -3.861 [-5.450, -2.225] |
| `motor-zero` | 6/6 | 0.000 | -6.166 [-7.842, -4.767] |

Late posture uses final-second mean tilt below 51.68 degrees and mean height above 1 mm, without a movement threshold. Net displacement includes falls and is not a measure of successful walking by itself. Changes are arm minus matched baseline. Intervals use 20,000 percentile bootstrap resamples; rows and columns of randomized arms are resampled independently to preserve the crossed design. The intervals are approximate with six noise seeds and four assignments. No significance or equivalence decision is made. Full paired differences, historical gate outcomes and conditional speeds are retained in the JSON.

## Interpretation

Both shuffled delays and the narrow delay distribution reduced displacement after averaging assignments within every noise block. The narrow control exactly preserved the timestep-rounded mean, so this contrast is not explained by a changed mean delivery slot. The original vector remains a whole-cell cable-length proxy; these results do not identify physiological propagation delays or compare optimally refitted model families.

Shuffled delays retained late posture in 23 of 24 trials, compared with five of six for the original assignment. The earlier pilot's shuffled trial fell. The original delay assignment is therefore not necessary for late posture in these runs, even though it produced greater mean displacement. Neither comparison establishes anatomical specificity of the connectome.

Fixed-mean stiffness and activation-driven stiffness retained late posture in all six trials. Both temporal shifts also retained movement in most trials. Their displacement intervals include zero; the follow-up does not establish a consistent advantage from the original load-to-stiffness timing. This is not an equivalence result. The startup-preserving shift matched all saved control-tick state arrays through the first 500 ms in all 24 cases.

With inhibitory normalization removed, late posture failed in all six trials. That is conditional on the retained weights and parameters; viable replacement factors in the separate October study remain relevant. Removing motor drive retained upright, stationary posture in all six trials, separating support from locomotion. The historical posture-and-motion gate rejected every motor-null trial because its displacement was negligible.

## Reproduce the analysis

Download [the complete raw-trace archive](https://github.com/The-Uploading-Lab/fly-behavior-experiments/releases/tag/publication-followup-2026-11-03), verify its checksum and extract it into an empty directory. The archive contains all six shard directories. From the repository root:

```sh
uv sync --locked
uv run python publication/analyse_followup.py \
  PATH/followup-2026110301 PATH/followup-2026110302 \
  PATH/followup-2026110303 PATH/followup-2026110304 \
  PATH/followup-2026110305 PATH/followup-2026110306
```

Archive: `fly-followup-20261103.tar.gz`, 1,101,707,969 bytes. SHA256: `e1c8df7b682dfcbb6eaac3ec4ada87d66d5782b1498b12105e285e1ac4d329ea`. All 840 archived files were read back and compared with the collected data before upload. Simulation code and large pinned inputs are obtained through the source repository; model directories are excluded from the archive.
