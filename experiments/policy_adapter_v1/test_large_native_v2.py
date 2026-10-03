"""Analytic calibration recovery and extreme/one-class support checks."""
import unittest
import numpy as np
from scipy.special import expit
from large_native_v2 import calibration


class RepairTests(unittest.TestCase):
    def test_recovery(self):
        z=np.linspace(-8,8,1000)
        coefficient,audit=calibration(z,expit(1.4*z+.7),ridge=1e-8)
        np.testing.assert_allclose(coefficient,[1.4,.7],atol=2e-5)
        self.assertLessEqual(audit['gradient_max'],1e-7)

    def test_extreme_and_single_class(self):
        for z in (np.full(128,-13.8),np.zeros(128),np.linspace(-13.8,13.8,128)):
            for y in (np.zeros(128),np.ones(128),np.arange(128)%2):
                for kind in ('intercept','platt'):
                    coefficient,audit=calibration(z,y,kind,.01)
                    self.assertTrue(np.isfinite(coefficient).all())
                    self.assertLessEqual(audit['gradient_max'],1e-7)


if __name__=='__main__':
    unittest.main()
