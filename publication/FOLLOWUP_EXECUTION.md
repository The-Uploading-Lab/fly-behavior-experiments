# Follow-up execution record

At the initial status check on 3 November 2026, three workers were running the first three noise blocks of the 138-trial [follow-up protocol](FOLLOWUP_PROTOCOL.md). The remaining three blocks were queued under the concurrency limit. No complete follow-up comparison is reported here.

The scientific source is frozen at `c855d05c5e3656ee70e2333d46e538de5af9f2e4`. Later commits add tests and execution records; the running checkout is unchanged. Slurm array 37883 has six tasks, at most three concurrent workers, one requested CPU and 6 GB per task, and a 4 hour 45 minute limit per task. There is no automatic requeue. Each task runs all 23 interventions for one noise seed. The complete experiment contains six noise seeds and four crossed assignment seeds.

## Completed checks

The Linux bootstrap, Slurm 37839_0, completed successfully. Ten component tests and all 64 two-neuron timing cases passed. Its full-model baseline took 479.5 seconds, with peak resident memory of 2,763,172 KiB. A separate local test of the full analysis reader passed using 138 synthetic records; it checks paired effects, crossed assignment aggregation, posture scoring, duplicate shards and altered output hashes. Synthetic records are implementation fixtures, not simulation findings.

The saved neural trace from the Linux baseline exactly matched the Mac pilot for seed 201303. The largest absolute difference in saved body positions was 2.30 × 10⁻⁹ in the stored position units; saved contact loads differed by up to 3.73 × 10⁻⁶ in their stored units. The computed speeds were 0.9412898722727095 and 0.9412898723620272 mm/s, respectively. Array-level comparisons, runtime versions and transport checksums are in [the bootstrap verification record](followup-bootstrap-verification.json). One baseline comparison does not establish equivalence for every trial. The Mac pilot will remain separate from the Linux follow-up.

## Setup departures and continuation

The first bootstrap job, 37826_0, failed during package installation because the compute node could not resolve the package index. No simulation ran. Installation was moved to the login node. An initial `uv` invocation selected Python 3.8.20 and failed before installation; the successful command specified `/usr/bin/python3` explicitly:

```sh
UV_CACHE_DIR="$PWD/outputs/uv-cache" outputs/bootstrap-env/bin/uv sync \
  --locked --python /usr/bin/python3 --index-url https://pypi.org/simple
```

The runtime receipt reports Python 3.12.13 on the compute node, while the login node reports 3.12.14. The analysis reader requires consistent runtime records across all six blocks. Installation logs and partial failures are retained. Model evaluations run only on Slurm compute nodes.

After all six blocks finish, the analysis command is:

```sh
.venv/bin/python publication/analyse_followup.py \
  outputs/publication/followup-2026110301 \
  outputs/publication/followup-2026110302 \
  outputs/publication/followup-2026110303 \
  outputs/publication/followup-2026110304 \
  outputs/publication/followup-2026110305 \
  outputs/publication/followup-2026110306
```

The reader must pass the complete-grid, provenance, output-hash and control-identity checks before interpretation. All planned outcomes, including falls and successful null controls, belong in the release. These comparisons estimate conditional sensitivity of one frozen model; they do not provide independent animal validation.
