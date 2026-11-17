"""The Metal engine reproduces the NumPy engine's spikes on a small network.

Covers the afferent draws ported to the GPU driver for this model: Poisson
drive, scalar Erlang renewal, and the per-neuron regularity vector that mixes
the two, both with a fixed drive and with a drive_fn callback. macOS only.
"""
import run
import contextlib
import io
import os
import subprocess
import sys
import unittest

import numpy as np
import pandas as pd

os.chdir(run.ROOT / "simulator")
GPU = run.ROOT / "simulator/gpu"


@unittest.skipUnless(sys.platform == "darwin", "Metal backend is macOS only")
class MetalMatchesNumpy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run(["make", "-s", "-C", str(GPU), "libcnsengine.dylib"], check=True)
        import cns
        cls.cns = cns
        rng = np.random.default_rng(7)
        n = 60
        ids = [str(i) for i in range(n)]
        nt = ["acetylcholine"] * 45 + ["gaba"] * 15
        cls.meta = pd.DataFrame(dict(
            banc_888_id=ids, neurotransmitter_verified=nt, neurotransmitter_predicted=nt,
            super_class=["other"] * n, cell_type=["fixture"] * n))
        pre = rng.integers(0, n, 600)
        post = rng.integers(0, n, 600)
        keep = pre != post
        cls.edges = (pd.DataFrame(dict(pre=pre[keep].astype(str), post=post[keep].astype(str),
                                       count=rng.integers(1, 12, keep.sum())))
                     .drop_duplicates(["pre", "post"]))
        cls.n = n

    def network(self, backend, **params):
        p = self.cns.SimParams(dt=0.1, backend=backend, **params)
        with contextlib.redirect_stdout(io.StringIO()):
            return self.cns.CNS(self.meta, self.edges, p)

    def compare(self, run_kw, **params):
        out = {}
        for backend in ("numpy", "metal"):
            net = self.network(backend, **params)
            with contextlib.redirect_stdout(io.StringIO()):
                out[backend] = net.run(400.0, seed=3, record_trace_every=1.0,
                                       first_spike=True, **run_kw())
        a, b = out["numpy"], out["metal"]
        self.assertGreater(a["n_spikes"], 0)
        self.assertEqual(a["n_spikes"], b["n_spikes"])
        np.testing.assert_array_equal(a["counts"], b["counts"])
        np.testing.assert_array_equal(a["t_first_ms"], b["t_first_ms"])
        np.testing.assert_array_equal(np.asarray(a["trace"]), np.asarray(b["trace"]))

    def fixed_drive(self):
        return dict(drive={tuple(range(20)): 90.0})

    def callback_drive(self):
        rows = np.arange(20)

        def drive_fn(t_ms, counts):
            return rows, np.full(len(rows), 40.0 + 0.2 * t_ms)
        return dict(drive_fn=drive_fn, feedback_rows=np.arange(20, 30), drive_every_ms=5.0)

    def mixed(self):
        reg = np.ones(self.n)
        reg[:10] = 8.0
        return reg

    def test_poisson(self):
        self.compare(self.fixed_drive)

    def test_scalar_renewal(self):
        self.compare(self.fixed_drive, spike_reg=8.0)

    def test_per_neuron_regularity(self):
        self.compare(self.fixed_drive, spike_reg_per_neuron=self.mixed())

    def test_per_neuron_regularity_with_callback(self):
        self.compare(self.callback_drive, spike_reg_per_neuron=self.mixed())


if __name__ == "__main__":
    unittest.main()
