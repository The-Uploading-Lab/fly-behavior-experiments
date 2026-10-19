"""Small implementation checks, not a whole-animal or retention replay."""
from pathlib import Path
import json
import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))
import body
import cns
import muscle_split
from inspect_model import verify


class SnapshotTests(unittest.TestCase):
    def test_integrity_and_syntax(self):
        manifest = verify()
        self.assertTrue(manifest['runtime_config_equals_tested_arm'])

    def test_motor_table_matches_theta(self):
        th = json.loads((ROOT / 'config/retained-build.json').read_text())
        rows = json.loads((ROOT / 'motor-neuron-parameters.json').read_text())['rows']
        self.assertEqual(len(rows), 391)
        self.assertEqual(len({r['banc_888_id'] for r in rows}), 391)
        self.assertEqual({c: sum(r['class'] == c for r in rows)
                          for c in ('fast', 'intermediate', 'slow')},
                         {'fast': 84, 'intermediate': 223, 'slow': 84})
        for r in rows:
            bid = r['banc_888_id']
            self.assertEqual(r['membrane_override'], th['mn_membrane_by_id'][bid])
            self.assertEqual(r['rest_override_mV'], th['mn_vrest_by_id'].get(bid))
            self.assertEqual(r['reset_offset_mV'], th['mn_reset_offset_by_id'].get(bid, 0.0))
        self.assertEqual(sum(r['rest_override_mV'] is None for r in rows), 13)

    def test_theta_keys_are_recognized_by_snapshot(self):
        import walk_search
        theta = json.loads((ROOT / 'config/retained-build.json').read_text())
        with patch.dict(os.environ, {'FLY_WBE_DECISION_GRADE': '1'}):
            walk_search._refuse_unknown_theta_keys(theta)

    def test_antagonist_split_and_activation(self):
        matrix = np.array([[1., -2.], [-3., 4.]])
        flex, ext, both = muscle_split.split_map(matrix)
        np.testing.assert_array_equal(ext - flex, matrix)
        self.assertTrue(both.all())
        b = body.Body(tau_act_ms=6, ranges=(np.full(42, -1.), np.ones(42)),
                      drive_scale=1., exact_filters=True, f0_hz=25.688, zeta=3.,
                      joint_integrator='exact')
        b.step2(np.ones(42), np.ones(42), 2.5)
        activation = 1 - np.exp(-2.5 / 6)
        np.testing.assert_allclose(b.a_flex, activation)
        np.testing.assert_allclose(b.cocon, 2 * activation)
        np.testing.assert_allclose(b.theta, 0.)

    def test_neuron_conductance_response_and_replay(self):
        meta = pd.DataFrame({'banc_888_id': ['1', '2'],
                             'neurotransmitter_verified': [None, None],
                             'neurotransmitter_predicted': ['acetylcholine', 'gaba'],
                             'cell_type': ['synthetic_a', 'synthetic_b'],
                             'cell_class': ['', ''], 'super_class': ['', ''],
                             'side': ['left', 'left'], 'region': ['', '']})
        edges = pd.DataFrame({'pre': ['1'], 'post': ['2'], 'count': [20]})
        p = cns.SimParams(conductance_based=True, split_conductances=True)
        net = cns.CNS(meta, edges, p)
        quiet = net.run(20., seed=7, record_voltage=np.array([1]))
        driven = net.run(20., drive={(0,): 200.}, seed=7,
                         record_voltage=np.array([1]))
        repeat = net.run(20., drive={(0,): 200.}, seed=7,
                         record_voltage=np.array([1]))
        np.testing.assert_array_equal(driven['voltage'], repeat['voltage'])
        self.assertGreater(float(driven['voltage'].max()), float(quiet['voltage'].max()))

    def test_position_and_velocity_encoder_algebra(self):
        # Directly exercise rates() with explicit group assignments, isolating
        # the encoder equations from the separate anatomical selector.
        p = body.Proprioceptors.__new__(body.Proprioceptors)
        names = ('claw_flex', 'claw_ext', 'hook_flex', 'hook_ext', 'club', 'hair', 'campaniform')
        p.groups = {g: np.array([i]) for i, g in enumerate(names)}
        p.dof_of = {g: np.array([0]) for g in names}
        p.claw_gain, p.claw_code = 1., None
        p.r_rest_hook = p.r_rest_hair = 5.
        p.v_scale_hook = p.v_scale_club = 8.
        p.camp_src, p.type_modality_rows, p.phasic = 'rest', (), 0.
        b = SimpleNamespace(normalised=lambda: np.array([.25]),
                            vel=np.array([4.]), a=np.array([0.]), drive_scale=1.)
        _, positive = p.rates(b)
        np.testing.assert_allclose(positive, [35., 95., 65., 5., 65., 5., 5.])
        b.vel *= -1
        _, negative = p.rates(b)
        np.testing.assert_allclose(negative, [35., 95., 5., 65., 65., 5., 5.])


if __name__ == '__main__':
    unittest.main(verbosity=2)
