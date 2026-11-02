# Follow-up protocol: replicated controls

Written on 3 November 2026 before any follow-up simulation. The earlier one-seed pilot informed this design. This is an exploratory follow-up on the frozen September model, with no parameter fitting or biological validation claim.

## Decisions this experiment can inform

The experiment asks whether the effects of fixed stiffness, activation stiffness, temporal shifts, delay changes and inhibitory rescaling persist across noise realizations. Two additions address specific pilot limitations: the original temporal null also changed startup, and equal nominal delay means became unequal after rounding to the neural clock.

The original nine intervention definitions and nuisance parameters in `PROTOCOL.md` are retained. Two additional randomized controls are used:

* `stiffness-shift-after-startup` preserves the first 200 body ticks (500 ms), then circularly shifts the remaining 2,200 ticks by a seeded offset between 25% and 75% of that remainder. The same offset applies to all joints. The per-joint distribution after startup is preserved exactly. The shift can introduce a discontinuity at 500 ms and still uses future reference values; it is a retrospective intervention.
* `delay-narrow-clock-mean` first computes the original integer delivery slots using `max(1, rint(delay/0.1))`. If their sum is S and the vector length is N, the new vector contains N minus (S mod N) slots of floor(S/N), and S mod N slots of floor(S/N)+1. Assignment to neurons is permuted with the stated null seed. This preserves the exact integer slot sum and gives the narrowest possible distribution on this clock. It is a two-valued delay control, not a uniform delay. Both it and the original nominal-uniform arm are retained.

## Fixed sample and execution

Noise seeds: 2026110301, 2026110302, 2026110303, 2026110304, 2026110305, 2026110306. Assignment seeds: 2026110311, 2026110312, 2026110313, 2026110314. These values were fixed without examining their follow-up outcomes. They are simulation realizations, not independent animals or independent connectomes.

Each noise block contains seven deterministic arms and four randomized arm families at four assignments each: 23 trials per block, 138 trials total. A shared base is run once per noise block. The four assignment seeds are crossed with the six noise seeds, so the 24 randomized trials in one family are not 24 independent reference pairs. No outcome-dependent stopping, substitution of favorable seeds or exclusion of falls is permitted. Infrastructure failures retain their partial outputs; a repaired rerun receives a new output directory and a receipt stating the departure.

```sh
uv sync --locked
uv run python publication/followup.py \
  --seeds 2026110301 2026110302 2026110303 2026110304 2026110305 2026110306 \
  --null-seeds 2026110311 2026110312 2026110313 2026110314 \
  --out outputs/publication/followup-complete
uv run python publication/analyse_followup.py outputs/publication/followup-complete
```

The present execution uses the FR8 normal CPU partition, three serial workers at most, each with one CPU and 6 GB requested memory, and one noise block per array task. The cluster allocation is checked against the shared-account ceiling before submission. Each paired block stays on one host; `OPENBLAS_CORETYPE=Haswell` and one BLAS/OpenMP thread are set explicitly. The source commit, Python, packages, host and scheduler job are recorded. The Mac pilot is reported separately and is not pooled with the Linux follow-up. A small engine timing fixture and a complete baseline portability run precede the array. New numerical trajectories on Linux are possible and are reported as such.

The GPU question was considered: changing to the historical native GPU backend would introduce another numerical and random-number implementation into this follow-up. The present comparison keeps the released NumPy backend and pairs all arms on the same CPU family. GPU acceleration requires a separate equivalence check. Runtime is timed on the baseline probe before the array is submitted; no speedup is assumed.

## Outcomes and analysis fixed before execution

The main descriptive outcomes are post-transient net displacement for every trial and late posture success. Late posture success uses the final second's mean tilt below 51.68 degrees and mean body height above 1 mm, without a movement threshold. The historical `standing_v32` posture-and-motion gate is reported separately. Thus a stationary upright motor-null trial remains upright, and a fall with large displacement is not called successful locomotion. Sustained 50 ms posture loss, joint tracking, actuator saturation, contact-tip motion and sampled work remain diagnostic outputs with the limitations in `PROTOCOL.md`.

For each intervention report the raw paired differences from its matching base. For a randomized arm, first average the four assignments within each noise block for the six block summaries. Report arm means, posture counts, mean and median paired displacement differences, all six block differences and the range across assignments. Keep missing conditional speed values missing.

An exploratory 95% percentile bootstrap interval for the mean displacement difference uses 20,000 resamples and analysis seed 2026110321. For deterministic arms, resample the six noise blocks. For randomized arms, independently resample six noise indices and four assignment indices, keeping their crossed matrix. This interval depends on the sampled noise and assignment distributions and is approximate with six by four realizations. It cannot establish animal-level uncertainty, equivalence, optimized-family superiority or connectome specificity. No significance threshold or p-value decision is used.

Before combining shards, the reader verifies the complete planned grid, case records, output hashes, source hashes, environment and control identities. Each identity replay must match all saved mechanical arrays and neural activity. The startup null must preserve startup; the clock-mean null must preserve the exact slot sum. A failed invariant stops interpretation and remains in the execution record. Raw traces and unsuccessful outcomes are published with the analysis.

## Cluster setup

The cluster's live storage notice requires data and environments under `/data/robin`, not `/home`. Prepare a private checkout there. The first bootstrap attempt (Slurm 37826) failed while installing uv because package-index DNS was unavailable on the compute node; no simulation ran. Environment installation is therefore performed on the login node, where editing and installation are allowed, with simulation restricted to Slurm workers. The failed log is retained. The setup commands from the checkout root are:

```sh
python3 -m venv outputs/bootstrap-env
outputs/bootstrap-env/bin/python -m pip install --index-url=https://pypi.org/simple uv==0.12.0
UV_CACHE_DIR="$PWD/outputs/uv-cache" outputs/bootstrap-env/bin/uv sync --locked --python python3
mkdir -p outputs/cluster-logs
sbatch --partition=normal --cpus-per-task=1 --mem=6G --time=00:25:00 \
  --array=0-0%1 --job-name=publication-bootstrap \
  --output=outputs/cluster-logs/%x-%A_%a.out publication/cluster_bootstrap.sh
```

After the bootstrap receipt and runtime are inspected, submit `publication/cluster_followup.sh` as array `0-5%3` with one CPU and 6 GB per task and a wall limit based on the probe. No model evaluation runs on the login node. The bootstrap verifies the two existing public input files by their released hashes before linking them into the private checkout.
