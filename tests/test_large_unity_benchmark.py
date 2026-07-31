import unittest
from pathlib import Path

import numpy as np

from paper_resubmission_20260728.evaluate_large_unity_benchmark import (
    ALIGN_ROOT_POS,
    CANONICAL_LEFT_HIP_POS,
    CANONICAL_LEFT_SHOULDER_POS,
    CANONICAL_RIGHT_HIP_POS,
    CANONICAL_RIGHT_SHOULDER_POS,
    JOINT_NAMES,
    LOCAL_DATA_PREFIX,
    OLD_DATA_PREFIX,
    camera_pair_features,
    ema_sequence,
    normalize_camera_id,
    project_points,
    rebase_path,
    reprojection_gated_triangulate_sequence,
    triangulate_point,
    weighted_fusion_array,
)


class LargeUnityBenchmarkTests(unittest.TestCase):
    def test_rebase_path_maps_stale_absolute_dataset_prefix(self):
        stale = OLD_DATA_PREFIX / "skiing_unity_dataset/data_pole_ski/female/action/meta/sequence.json"

        rebased = rebase_path(stale)

        self.assertEqual(
            rebased,
            LOCAL_DATA_PREFIX / "skiing_unity_dataset/data_pole_ski/female/action/meta/sequence.json",
        )

    def test_camera_pair_features_uses_wrapped_angle_delta(self):
        layer_delta, angle_delta = camera_pair_features("capture_L0_A350", "capture_L4_A010")

        self.assertEqual(layer_delta, 4)
        self.assertEqual(angle_delta, 20)

    def test_normalize_camera_id_strips_capture_prefix(self):
        self.assertEqual(normalize_camera_id("capture_L2_A180"), "L2_A180")

    def test_triangulate_point_recovers_synthetic_point(self):
        p1 = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        p2 = np.array([[1.0, 0.0, 0.0, -1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        point = np.array([0.25, -0.10, 2.0])
        uv1 = np.array([point[0] / point[2], point[1] / point[2]])
        uv2 = np.array([(point[0] - 1.0) / point[2], point[1] / point[2]])

        recovered = triangulate_point(p1, uv1, p2, uv2)

        self.assertTrue(np.allclose(recovered, point, atol=1e-8))

    def test_weighted_fusion_array_respects_confidence_extremes(self):
        left = np.array([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])
        right = np.array([[[2.0, 0.0, 0.0], [20.0, 0.0, 0.0]]])
        q_left = np.array([[100.0, 0.0]])
        q_right = np.array([[0.0, 100.0]])

        fused = weighted_fusion_array(left, right, q_left, q_right)

        self.assertTrue(np.allclose(fused[0, 0], left[0, 0], atol=1e-6))
        self.assertTrue(np.allclose(fused[0, 1], right[0, 1], atol=1e-6))

    def test_canonical_anchor_positions_are_anatomical_joints(self):
        self.assertEqual(JOINT_NAMES[ALIGN_ROOT_POS], "neck_01")
        self.assertEqual(JOINT_NAMES[CANONICAL_LEFT_HIP_POS], "Thigh_L")
        self.assertEqual(JOINT_NAMES[CANONICAL_RIGHT_HIP_POS], "Thigh_R")
        self.assertEqual(JOINT_NAMES[CANONICAL_LEFT_SHOULDER_POS], "Upperarm_L")
        self.assertEqual(JOINT_NAMES[CANONICAL_RIGHT_SHOULDER_POS], "Upperarm_R")

    def test_project_points_applies_pinhole_projection(self):
        projection = np.array([[100.0, 0.0, 0.0, 0.0], [0.0, 100.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        xyz = np.array([[[1.0, 2.0, 10.0]]])

        uv = project_points(projection, xyz)

        self.assertTrue(np.allclose(uv[0, 0], [10.0, 20.0]))

    def test_reprojection_gate_can_invalidate_large_residuals(self):
        p1 = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        p2 = np.array([[1.0, 0.0, 0.0, -1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        uv1 = np.array([[[0.10, 0.05]]])
        uv2 = np.array([[[-0.30, 0.05]]])

        gated = reprojection_gated_triangulate_sequence(p1, p2, uv1, uv2, threshold_px=-1.0)

        self.assertFalse(np.isfinite(gated).all())

    def test_ema_sequence_smooths_and_carries_forward_missing_joints(self):
        seq = np.array([[[0.0, 0.0, 0.0]], [[10.0, 0.0, 0.0]], [[np.nan, np.nan, np.nan]]])

        smoothed = ema_sequence(seq, alpha=0.5)

        self.assertTrue(np.allclose(smoothed[0, 0], [0.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(smoothed[1, 0], [5.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(smoothed[2, 0], [5.0, 0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
