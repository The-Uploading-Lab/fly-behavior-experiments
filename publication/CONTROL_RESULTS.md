# Controlled reproduction pilot

All nine six-second trials completed on neural noise seed 201303. Randomized controls used assignment seed 9001. These are conditional simulation results from one selected September configuration. The identity replay reproduced all saved mechanical arrays and neural activity exactly. The same baseline speed was obtained by the previously released CPU runner.

| Arm | Historical gate | Net displacement mm | Conditional speed mm/s |
|---|---|---:|---:|
| base | pass | 5.177094 | 0.941290 |
| stiffness-replay | pass | 5.177094 | 0.941290 |
| stiffness-mean | pass | 5.045477 | 0.917360 |
| activation | pass | 6.510874 | 1.183795 |
| delay-uniform | pass | 3.083869 | 0.560703 |
| inhibition-unscaled | fail | 1.334960 | not scored |
| motor-zero | fail | 0.000000 | not scored |
| stiffness-shift | pass | 5.225294 | 0.950053 |
| delay-shuffle | fail | 4.749187 | not scored |

The historical gate named `standing_v32` requires mean tilt below 51.68 degrees and mean height above 1 mm during the final second, plus at least 0.3 mm displacement. The zero-motor model remains upright at approximately 1.128 mm height and 2.10 degrees mean late tilt, but has approximately 3.27e-9 mm net motion. Its gate failure is inactivity, not a fall. Unscaled inhibition and shuffled delays produce mean late tilts of 139.49 and 170.95 degrees respectively.

The time-shifted stiffness null slightly exceeds the original displacement on this seed. This pilot does not support an advantage for the original temporal alignment of stiffness and load. Activation-based stiffness is faster than the original on this seed, while fixed mean stiffness is slightly slower. These alternatives were not refitted; a single paired block supplies no general ranking or equivalence inference. The shifted sequence includes startup, and both fixed-mean and shifted controls use the original trial's full trace rather than a deployable feedback law.

The uniform-delay intervention matches the nominal delay mean, 6.240070568 ms. At the model's 0.1 ms neural timestep, source delays are rounded to delivery slots. The resulting means are 6.238163367 ms for the original vector and 6.200000000 ms for the uniform vector. The shuffled assignment preserves both distributions exactly. The original protocol is retained unchanged; this rounding difference was identified during the pilot and is reported by the analyser.

The reconstructed September motor map modulates 36 of 42 leg axes. The earlier August map's 30-axis statement describes a different build. Mechanical diagnostics retain transient posture loss separately from the historical late-window gate. For example, the original model drops below 1 mm briefly after 0.565 s and later passes the historical gate.

The [execution receipt](controls-verification.json) records source hashes, tests and the raw archive checksum. [Computed comparisons](expected/controls-pilot-analysis.json) and the [complete run summary](expected/controls-pilot-summary.json) are included in Git. The [raw trace archive](https://github.com/The-Uploading-Lab/fly-behavior-experiments/releases/download/publication-controls-2026-11-03/fly-controls-pilot-20261103.tar.gz) includes every trial's parameters, output arrays, logs and hashes. After extracting it, recompute the comparison with:

```sh
uv run python publication/analyse_controls.py controls-pilot
```

The archive checksum is `1d1a234d3e6ba7493c34bd854b6db9af2ec2fa63bafdf2c69730ede60edf8c70`. Re-running `publication/reproduce.py` regenerates the model outputs using public inputs. The reference platform is macOS arm64; exact whole-trajectory equality on other platforms has not been established.
