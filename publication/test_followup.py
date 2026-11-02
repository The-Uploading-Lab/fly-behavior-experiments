"""Check the full crossed-grid reader with explicitly synthetic observations."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest

import numpy as np

from analyse_followup import followup, SEEDS, NULL_SEEDS, RANDOM_MODES


class FollowupReaderTests(unittest.TestCase):
    def test_complete_grid_effects_posture_and_integrity(self):
        fixed = ["base", "stiffness-replay", "stiffness-mean", "activation", "delay-uniform", "inhibition-unscaled", "motor-zero"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for seed in SEEDS:
                for mode, assignment in ([(m, None) for m in fixed] + [(m, a) for m in sorted(RANDOM_MODES) for a in NULL_SEEDS]):
                    case = f"{mode}-{seed}-{assignment}"
                    path = root / case
                    path.mkdir()
                    body = np.tile([0., 0., 1.2], (2400, 1))
                    quat = np.tile([1., 0., 0., 0.], (2400, 1))
                    if mode == "inhibition-unscaled":
                        quat[:] = [0., 1., 0., 0.]
                    np.savez_compressed(path / "mechanics.npz", body=body, quat=quat,
                                        stiffness=np.ones((2400, 2)), delay_ms=np.array([6.2, 6.3]))
                    # These are synthetic reader values, not simulated displacements.
                    effect = 0. if mode in ("base", "stiffness-replay") else 2.
                    row = {"case": case, "mode": mode, "seed": seed, "null_seed": assignment,
                           "wall_seconds": 1., "neural_counts_sha256": "synthetic",
                           "historical_metrics": {"standing_v32": mode not in ("motor-zero", "inhibition-unscaled")},
                           "mechanical_metrics": {"net_displacement_mm": 1. + effect, "sustained_posture_loss_s": None},
                           "files": {"mechanics.npz": hashlib.sha256((path / "mechanics.npz").read_bytes()).hexdigest()}}
                    (path / "result.json").write_text(json.dumps(row))
                    rows.append(row)
            summary = {"status": "complete", "protocol_version": 2, "noise_seeds": SEEDS,
                       "null_seeds": NULL_SEEDS, "rows": rows, "release_commit": "synthetic",
                       "source_hashes": {}, "packages": {}, "python": "synthetic", "machine": "synthetic",
                       "numerical_environment": {}, "execution_host": "synthetic"}
            (root / "summary.json").write_text(json.dumps(summary))
            result = followup([root])
            self.assertEqual(result["trials"], 138)
            arms = {a["mode"]: a for a in result["arms"]}
            self.assertEqual(arms["activation"]["mean_paired_difference_mm"], 2.)
            self.assertEqual(arms["activation"]["exploratory_bootstrap_95_interval_mm"], [2., 2.])
            self.assertEqual(arms["delay-shuffle"]["block_differences_mm"], [2.] * 6)
            self.assertEqual(arms["motor-zero"]["posture_success_count"], 6)
            self.assertEqual(arms["motor-zero"]["historical_gate_success_count"], 0)
            self.assertEqual(arms["inhibition-unscaled"]["posture_success_count"], 0)
            with self.assertRaisesRegex(ValueError, "Duplicate case across shards"):
                followup([root, root])
            changed = root / rows[0]["case"] / "mechanics.npz"
            with changed.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "Changed output"):
                followup([root])


if __name__ == "__main__":
    unittest.main(verbosity=2)
