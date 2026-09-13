import run
import os
os.chdir(run.ROOT / "simulator")
import unittest
import pandas as pd
from karolina_experiment import apply_intervention, ADAPT


class InterventionTests(unittest.TestCase):
    def test_exact_delta_preserves_existing_model(self):
        base = {'adapt_a': 2., 'adapt_b': 1.0418521601266768, 'tau_w': 270.97700471360525,
                'row_pair_seams': [{'gain': 1.25}], 'g_gap': 3., 'other': {'gain': 7.}}
        result = apply_intervention(base, 'ibuprofen', None)
        self.assertEqual({k: result[k] for k in ADAPT}, ADAPT)
        self.assertEqual({k: v for k, v in result.items() if k not in ADAPT},
                         {k: v for k, v in base.items() if k not in ADAPT})
        result['other']['gain'] = 2.
        self.assertEqual(base['other']['gain'], 7.)

    def test_both_visual_pools_compose_with_existing_gains(self):
        meta = pd.DataFrame({'cell_type': ['LC4', 'LPLC2', 'DNp01', 'DNp01', 'other'],
                             'banc_888_id': [1, 2, 3, 4, 5]})
        base = {'row_pair_seams': [{'gain': .25}]}
        result = apply_intervention(base, 'deprivation_b_half', meta)
        self.assertEqual(result['row_pair_seams'], [{'gain': .25},
             {'pre_ids': ['1', '2'], 'post_ids': ['3', '4'], 'gain': .5}])
        self.assertEqual(apply_intervention(base, 'control', None), base)
        with self.assertRaises(ValueError):
            apply_intervention(base, 'deprivation_b_half', meta.iloc[:3])


if __name__ == '__main__':
    unittest.main()
