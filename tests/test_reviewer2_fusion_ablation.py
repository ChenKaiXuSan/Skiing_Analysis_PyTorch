import unittest

import numpy as np

from reviewer2_fusion_ablation import (
    bootstrap_ci,
    holm_adjust,
    paired_effect_size_dz,
    sign_flip_p_value,
    weighted_fusion_array,
)


class Reviewer2FusionAblationTests(unittest.TestCase):
    def test_weighted_fusion_array_matches_confidence_extremes(self):
        left = np.array([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])
        right = np.array([[[2.0, 0.0, 0.0], [20.0, 0.0, 0.0]]])
        wl = np.array([[100.0, 0.0]])
        wr = np.array([[0.0, 100.0]])

        fused = weighted_fusion_array(left, right, wl, wr)

        self.assertTrue(np.allclose(fused[0, 0], left[0, 0], atol=1e-6))
        self.assertTrue(np.allclose(fused[0, 1], right[0, 1], atol=1e-6))

    def test_sign_flip_p_value_detects_consistent_positive_difference(self):
        diffs = np.ones(12)

        p = sign_flip_p_value(diffs)

        self.assertGreater(p, 0.0)
        self.assertLess(p, 0.01)

    def test_bootstrap_ci_is_ordered_and_contains_mean(self):
        values = np.array([1.0, 2.0, 3.0, 4.0])

        lo, hi = bootstrap_ci(values, seed=0, n_boot=500)

        self.assertLessEqual(lo, values.mean())
        self.assertGreaterEqual(hi, values.mean())

    def test_paired_effect_size_dz_uses_difference_scale(self):
        before = np.array([3.0, 4.0, 5.0])
        after = np.array([2.0, 3.0, 4.0])

        dz = paired_effect_size_dz(before, after)

        self.assertGreater(dz, 0)

    def test_holm_adjust_monotonicity(self):
        adjusted = holm_adjust([0.01, 0.03, 0.20])

        self.assertLessEqual(adjusted[0], adjusted[1])
        self.assertLessEqual(adjusted[1], adjusted[2])
        self.assertTrue(all(0.0 <= p <= 1.0 for p in adjusted))


if __name__ == "__main__":
    unittest.main()
