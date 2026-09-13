import run
import os
os.chdir(run.ROOT / "simulator")
"""Check the R2 estimator against a known linear path and changes of image scale."""
import unittest
import numpy as np
from projection import summarize


class ProjectionEstimator(unittest.TestCase):
    def test_known_path_and_scale_invariance(self):
        xy = np.column_stack([np.arange(121), np.zeros(121)])
        lengths = np.full(121, 20.)
        a = summarize(xy, lengths, 120)
        # R2's reflected 5-frame median trims one pixel from either endpoint.
        self.assertEqual(a['path_px'], 118.)
        self.assertEqual(a['net_px'], 118.)
        self.assertEqual(a['speed_silh_s'], 11.8)
        self.assertEqual(a['straightness'], 1.)
        b = summarize(xy*3+57., lengths*3, 120)
        self.assertAlmostEqual(a['speed_silh_s'], b['speed_silh_s'])
        self.assertAlmostEqual(a['straightness'], b['straightness'])
        self.assertEqual(a['sensitivity_min_silh_s'], a['sensitivity_max_silh_s'])


if __name__ == '__main__':
    unittest.main()
