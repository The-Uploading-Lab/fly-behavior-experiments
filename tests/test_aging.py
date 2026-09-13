import run
import os
os.chdir(run.ROOT / "simulator")
"""Tests for zero identity, signed full-network coverage and gap conservation."""
import unittest
from types import SimpleNamespace
import numpy as np
from scipy import sparse
from lanes.longevity.ageing.global_ageing import apply_to_network,efficacy

def fixture():
    w=sparse.csc_matrix(np.array([[0.,2.,-3.],[4.,0.,-5.],[6.,7.,0.]],dtype=np.float32))
    g=sparse.csr_matrix(np.array([[0.,4.,0.],[4.,0.,2.],[0.,2.,0.]],dtype=np.float32))
    return SimpleNamespace(N=3,ids=['a','b','c'],W=w,_data=w.data,Wg=w[:,:1].tocsr(),
        G=g,gap_deg=np.asarray(g.sum(axis=1)).ravel().astype(np.float32),gap_pair_weights=np.array([4.,2.]))

class GlobalAgeingTests(unittest.TestCase):
    def test_day_zero_is_bit_identical(self):
        n=fixture(); snapshots={k:getattr(n,k).data.copy() for k in ['W','Wg','G']}
        a=apply_to_network(n,0)
        for k,v in snapshots.items():np.testing.assert_array_equal(getattr(n,k).data,v)
        self.assertTrue(a['age_zero_noop'])
    def test_all_outputs_scale_after_balance(self):
        n=fixture();before=n.W.toarray();gr=n.Wg.toarray();g=n.G.toarray()
        a=apply_to_network(n,90)
        np.testing.assert_array_equal(n.W.toarray(),before*.25)
        np.testing.assert_array_equal(n.Wg.toarray(),gr*.25)
        np.testing.assert_array_equal(n.G.toarray(),g*.25)
        np.testing.assert_array_equal(n.G@np.ones(3)-n.gap_deg,np.zeros(3))
        self.assertEqual(a['assigned_neurons'],3)
        self.assertEqual(a['neurons_with_chemical_outputs'],3)
        self.assertTrue(np.shares_memory(n.W.data,n._data))
    def test_mapping_and_invalid_days(self):
        self.assertEqual(efficacy(0),1.);self.assertEqual(efficacy(45),.5);self.assertEqual(efficacy(90),.25)
        for age in [-1,91,float('nan')]:
            with self.assertRaises(ValueError):efficacy(age)

if __name__=='__main__':unittest.main()
