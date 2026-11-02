"""Algebraic and regression checks for the published intervention adapter."""
from pathlib import Path
import importlib.util
import contextlib
import io
import hashlib
import json
import tempfile
import sys
import unittest
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from reproduce import delay_control, stiffness_schedule, mechanical_metrics
from analyse_controls import analyse

spec = importlib.util.spec_from_file_location("frozen_muscle", HERE / "delay-model/muscle_split.py")
muscle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(muscle)


class ControlTests(unittest.TestCase):
    def test_analysis_rejects_changed_outputs_and_keeps_nonstanding_trials(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            modes = ["base", "stiffness-replay", "stiffness-mean", "activation", "delay-uniform", "inhibition-unscaled", "motor-zero", "stiffness-shift", "delay-shuffle"]
            rows = []
            for mode in modes:
                case = root / mode
                case.mkdir()
                (case / "trace.bin").write_bytes(b"synthetic integrity fixture")
                np.savez_compressed(case / "mechanics.npz", delay_ms=np.array([.14, 6.24]))
                row = {"case": mode, "mode": mode, "seed": 7,
                       "null_seed": 9001 if mode in ("stiffness-shift", "delay-shuffle") else None,
                       "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in case.iterdir()},
                       "wall_seconds": 1., "historical_metrics": {"standing_v32": mode != "motor-zero"},
                       "mechanical_metrics": {"net_displacement_mm": 1., "sustained_posture_loss_s": None}}
                (case / "result.json").write_text(json.dumps(row))
                rows.append(row)
            summary = {"status": "complete", "rows": rows, "noise_seeds": [7], "null_seeds": [9001]}
            (root / "summary.json").write_text(json.dumps(summary))
            result = analyse(root)
            self.assertEqual(result["trials"], 9)
            self.assertFalse(next(r for r in result["comparisons"] if r["mode"] == "motor-zero")["standing"])
            (root / "motor-zero/trace.bin").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Changed output"):
                analyse(root)
            summary["rows"] = rows[:-1]
            (root / "summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "Missing or duplicated"):
                analyse(root)

    def test_actual_inhibitory_scaling_changes_only_incoming_negative_weights(self):
        import pandas as pd
        sys.path.insert(0, str(HERE / "delay-model"))
        import cns
        meta = pd.DataFrame({"banc_888_id": ["exc", "inh", "target"],
                             "neurotransmitter_verified": ["acetylcholine", "gaba", "acetylcholine"],
                             "neurotransmitter_predicted": ["acetylcholine", "gaba", "acetylcholine"],
                             "super_class": ["other"] * 3, "cell_type": ["fixture"] * 3})
        edges = pd.DataFrame({"pre": ["exc", "inh"], "post": ["target", "target"], "count": [10, 5]})
        with contextlib.redirect_stdout(io.StringIO()):
            off = cns.CNS(meta, edges, cns.SimParams(balance_target=0.))
            on = cns.CNS(meta, edges, cns.SimParams(balance_target=.5))
        self.assertIsNone(off.balance_scale)
        before, after = off.W.toarray(), on.W.toarray()
        np.testing.assert_array_equal(before[before >= 0], after[before >= 0])
        self.assertAlmostEqual(float(after[2, 0] / -after[2, 1]), .5, places=6)
        self.assertAlmostEqual(float(after[2, 1] / before[2, 1]), float(on.balance_scale[2]), places=6)

    def test_delay_null_preserves_distribution_and_changes_assignment(self):
        reference = np.geomspace(.1, 23., 1000)
        a = delay_control(reference, "delay-shuffle", 9001)
        b = delay_control(reference, "delay-shuffle", 9002)
        np.testing.assert_array_equal(np.sort(a), reference)
        np.testing.assert_array_equal(a, delay_control(reference, "delay-shuffle", 9001))
        self.assertFalse(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, reference))
        flat = delay_control(reference, "delay-uniform", None)
        np.testing.assert_allclose(flat, reference.mean(), rtol=0, atol=0)

    def test_stiffness_null_preserves_joint_distributions_and_alignment(self):
        reference = np.arange(1200, dtype=float).reshape(400, 3)
        shifted, receipt = stiffness_schedule(reference, "stiffness-shift", 9001)
        offset = receipt["offset_ticks"]
        self.assertTrue(100 <= offset <= 300)
        np.testing.assert_array_equal(np.roll(shifted, -offset, axis=0), reference)
        np.testing.assert_array_equal(np.sort(shifted, axis=0), reference)
        mean, _ = stiffness_schedule(reference, "stiffness-mean", None)
        np.testing.assert_allclose(mean, np.broadcast_to(reference.mean(axis=0), reference.shape))
        identity, _ = stiffness_schedule(reference, "stiffness-replay", None)
        np.testing.assert_array_equal(identity, reference)
        self.assertFalse(np.shares_memory(identity, reference))

    def test_split_preserves_signed_map_and_single_channel_axes(self):
        m = np.array([[2., 3., 0., 0.], [-1., 4., -2., 0.]])
        flex, ext, both = muscle.split_map(m)
        np.testing.assert_array_equal(ext - flex, m)
        np.testing.assert_array_equal(both, [True, False, False, False])

    def test_actual_muscle_laws_preserve_servo_equilibrium(self):
        # Exercise the frozen implementation, with explicit actuator storage.
        c = muscle.StiffnessController.__new__(muscle.StiffnessController)
        c.ids, c.mask = np.array([0, 2]), np.array([0, 2])
        c.k_min, c.k_gain, c.drive_scale, c.kv_of_k = 40., 140., 2., 0.
        c.m = SimpleNamespace(actuator_gainprm=np.full((4, 10), 45.),
                              actuator_biasprm=np.zeros((4, 10)))
        c.apply(np.array([0., 1., 2., 3.]))
        expected = np.array([40., 40. + 140. * np.tanh(1.)])
        np.testing.assert_allclose(c.m.actuator_gainprm[c.ids, 0], expected)
        np.testing.assert_array_equal(c.m.actuator_biasprm[c.ids, 1], -expected)
        np.testing.assert_array_equal(c.m.actuator_gainprm[[1, 3], 0], [45., 45.])
        c.set_leg_map(["lf_a", "lf_b", "rh_a", "rh_b"], 10.)
        c.apply_load({"lf": -5., "rh": 30.})
        np.testing.assert_array_equal(c.m.actuator_gainprm[c.ids, 0], [40., 180.])
        target = np.array([.4, -.2])
        force_at_target = c.m.actuator_gainprm[c.ids, 0] * target + c.m.actuator_biasprm[c.ids, 1] * target
        np.testing.assert_array_equal(force_at_target, [0., 0.])

    def test_posture_loss_metrics_keep_a_fallen_trial(self):
        n, dt = 400, .0025
        body = np.column_stack([np.arange(n) * dt, np.zeros(n), np.full(n, 2.)])
        quat = np.tile([1., 0., 0., 0.], (n, 1))
        quat[240:260] = [np.sqrt(.5), np.sqrt(.5), 0., 0.]
        a = {"body": body, "quat": quat, "actuator_force": np.ones((n, 2)),
             "achieved_joint_angle": np.zeros((n, 2)), "target_joint_angle": np.full((n, 2), .2),
             "force_limits": np.tile([-65., 65.], (2, 1)), "loads": np.full((n, 6), 2.),
             "tips": np.zeros((n, 6, 3))}
        out = mechanical_metrics(a, dt)
        self.assertAlmostEqual(out["sustained_posture_loss_s"], .6025)
        self.assertAlmostEqual(out["displacement_before_posture_loss_mm"], .1)
        self.assertAlmostEqual(out["joint_tracking_rmse_rad"], .2)
        self.assertEqual(out["positive_actuator_work_model_units"], 0.)
        self.assertEqual(out["actuator_force_limit_fraction"], 0.)


if __name__ == "__main__":
    unittest.main(verbosity=2)
