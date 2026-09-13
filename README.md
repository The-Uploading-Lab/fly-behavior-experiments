# Fly behavior experiments

Run the two hackathon experiments locally: the comparison with 25 wet-lab recording records, and a global neuronal-aging hypothesis. Both use the same 188,508-neuron BANC-based model. Optional videos replay the computed body poses.

**[Watch the 2:18 recorded demo](https://github.com/The-Uploading-Lab/fly-behavior-experiments/releases/download/v1.0.0/hackathon-demo.mp4).**

## Install

Install [Git](https://git-scm.com/downloads) and [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/The-Uploading-Lab/fly-behavior-experiments.git
cd fly-behavior-experiments
uv sync --locked
```

`uv` installs the pinned Python 3.12 environment. No GPU, cloud account, research checkout or private credentials are required. The first experiment command downloads **416.7 MB** of pinned connectome inputs and verifies their SHA-256 hashes. After installation and that download, runs work offline.

Allow roughly **5 GB of disk space** for the environment, data and generated captures. The tested 3-second control simulation used a peak footprint of **3.24 GB** and took **79 seconds** on the development MacBook Pro; walking cases run longer. Runs execute sequentially.

The reference platform is macOS arm64. The dependencies also support Linux and Windows, but cross-platform trajectories can differ numerically. On headless Ubuntu/Debian, install `libosmesa6` before running: silhouette statistics and videos use MuJoCo's offscreen renderer. The runner selects OSMesa when no display is present. See [MuJoCo rendering documentation](https://mujoco.readthedocs.io/en/stable/programming/visualization.html).

## Run the 25-record comparison

```bash
uv run python run.py wetlab
```

This runs control, ibuprofen and the calorie-restriction circuit hypothesis, each with a 6-second walking trial and a 3-second escape trial. It then computes a comparison for every measured clip. The **25 clips contain 21 movement windows and 11 tap observations**; two movement windows are not walking, and one clip has two taps. The six seed-zero simulations supply condition/protocol references. Repeated comparisons are not independent simulated animals.

Open these files in `outputs/wetlab/`:

| File | Contents |
|---|---|
| `records-25.csv` | One row per clip, including its available comparisons |
| `movement.csv` | Speed in fitted silhouette lengths/s, path, net displacement, straightness, ratios and processing sensitivity |
| `taps.csv` | Every departure interval or censored nondeparture, corresponding simulated support-loss time, and disagreement |
| `summary.csv` | Descriptive walking summaries by condition |
| `statistics.json` | The same results plus projection settings and provenance |

Capture time labels are retained. The model has no treatment exposure-time kinetics. Wet-lab physical scale, treatment-start times, fly identities and complete trial denominators were unavailable, so these comparisons do not estimate treatment effects. The wet-lab measurements are exploratory, unreviewed candidates.

## Run aging

```bash
uv run python run.py aging
```

Runs days **0, 10, ..., 90**, with walking, antennal escape input and a no-input control at each age. Results are in `outputs/aging/aging.csv` and `statistics.json`. Walking values are censored after standing failure; nondeparture remains censored rather than receiving an invented latency.

A shorter subset uses the same model and protocols:

```bash
uv run python run.py aging --days 0 30 90
```

These are freshly initialized age-parameter snapshots. The **45-day synaptic-efficacy half-life is assumed**, not a calibrated biological clock. [MODELS.md](MODELS.md) describes the model and [the frozen protocol](experiments/aging-protocol.md) gives the original experiment rules.

## Optional videos

```bash
uv run python run.py wetlab --video
uv run python run.py aging --video
```

MP4s and preview PNGs appear in each experiment's `videos/` folder. Walking plays at real simulated speed; escape and sham play at one-eighth speed, with a simulated-time label. Aged or treated cases that do not jump remain nonjumping. The flight/return failures remain labelled.

Completed runs are checked and reused when adding videos or resuming. To repeat into a fresh directory, use `--out outputs/my-repeat`. An incomplete run is not silently reused; choose a fresh output directory after an interrupted simulation. Films follow the displayed body; the internal sensory body is checked separately for standing. Wingbeats, attitude support and landing contain programmed control.

## Check the implementation

```bash
uv run python -m unittest discover -s tests -v
```

Tests cover the exact ibuprofen parameter change, deprivation targeting, age-zero identity, global signed-weight scaling, electrical-current conservation, silhouette scale invariance, all 25 records, and censored nondepartures. Small reference result files are in `expected/`; simulation code and calibration inputs are in `simulator/`. Historical module paths are retained to preserve the tested runtime.

The source/input hashes, original source identities and reproduction-check results are in `provenance/`. [MODELS.md](MODELS.md) explains the intervention assumptions. [THIRD_PARTY.md](THIRD_PARTY.md) credits the connectome and body-model sources.
