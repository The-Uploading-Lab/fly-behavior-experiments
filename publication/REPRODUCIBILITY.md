# Claim to command map

Reproduction has two distinct tasks: recalculating a result from released observations and regenerating the simulation that produced them. The table states which is available. A source snapshot or saved result table alone does not establish that the original experiment can be rerun.

| Result or implementation | Public command | Null or comparison | Coverage |
|---|---|---|---|
| Antagonist motor map and stiffness equations | `uv run python publication/test_reproduce.py` | Single-channel axes retain stock gain; equilibrium is unchanged when stiffness changes | Exact implementation checks |
| Matched muscle interventions on qh1 | `uv run python publication/reproduce.py` | Identity replay, fixed per-joint mean, circular time shift, zero motor drive; activation alternative | Full serial simulation and raw trace export; one-seed default pilot |
| Delay timing | `uv run python publication/timing_fixture.py` | Change target delay; compare queue with bucket delivery | 64 two-neuron CPU cases; no full-network backend equivalence claim |
| Delay assignment in the controlled protocol | `uv run python publication/reproduce.py` | Uniform nominal mean and distribution-preserving shuffled assignments | Full simulation; explicit noise and assignment seeds; analysis also reports timestep-rounded delivery means |
| Static inhibitory input scaling | `uv run python publication/test_reproduce.py` and `uv run python publication/reproduce.py` | Same graph with scaling disabled | Exact equation check and whole-model ablation on September source |
| August two-configuration standing comparison | `uv run python publication/replay_muscle.py --seeds 46-75` | Original predecessor and candidate, paired seeds | Historical source reconstruction with later recovered range input; report numerical discrepancies from archive |
| September six-arm delay table | `python3 publication/analyse.py`; `uv run python publication/replay_delay.py --arm base --seed 201303` and the other named arms/seeds | Uniform, clipped and shuffled delays | Archived Metal table recalculation; CPU simulation available, cross-backend numerical differences documented |
| October inhibitory replacement study | `python3 publication/analyse.py` | Original F, M and other replacement summaries | Recomputes standing counts and paired discordances from saved failure identities; speed medians remain quoted summaries; original simulation pipeline still needs export |
| H4 calcium error | `python3 publication/analyse.py` | Saved model and animal features, including disagreements | Recalculates feature errors; no raw-fluorescence pipeline or independent-animal validation |
| Walking speed-bin support | `python3 publication/analyse.py` | Preserve every failed support check | Recounts saved observer decisions; raw-trace observation pipeline still needs export |
| October motor-neuron refinement | `uv run python model-review/test_components.py` | Earlier configuration records in model review | Implementation and parameter checks; full comparative whole-model experiment still needs export |

The complete controlled protocol is in [PROTOCOL.md](PROTOCOL.md). It holds nuisance parameters fixed for component sensitivity, without a search over the alternatives. Anatomical specificity would require randomized-connectome controls with equal fitting budgets. Neither a single-seed ablation nor a null that fails without refitting establishes that specificity.

The next packaging work is the October inhibitory study and the raw walking/calcium observation pipelines. Their archived summaries remain usable as historical context while the executable central model and its controls are reviewed.
