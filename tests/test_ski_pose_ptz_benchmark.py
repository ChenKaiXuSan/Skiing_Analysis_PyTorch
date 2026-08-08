import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "paper_array_20260730/array_submission/scripts/ski_pose_ptz/evaluate_ski_pose_ptz_benchmark.py"
)
spec = importlib.util.spec_from_file_location("ski_pose_ptz_benchmark", SCRIPT)
ptz = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ptz
spec.loader.exec_module(ptz)


class SkiPosePtzBenchmarkTests(unittest.TestCase):
    def test_joint_mapping_selects_paper_15_joint_subset_from_h36m_17(self):
        arr = np.arange(17 * 3, dtype=np.float64).reshape(17, 3)

        mapped = ptz.map_h36m17_to_paper15(arr)

        self.assertEqual(mapped.shape, (15, 3))
        self.assertTrue(np.array_equal(mapped[0], arr[9]))
        self.assertTrue(np.array_equal(mapped[-1], arr[8]))

    def test_camera_pair_record_encodes_public_dataset_subject_holdout(self):
        record = ptz.PtzPairRecord(split="test", seq=405, subj=5, cam1=0, cam2=3)

        self.assertEqual(record.pair_id, "test::seq405::subj5::cam00::cam03")
        self.assertEqual(record.person_id, "subj5")
        self.assertEqual(record.action_id, "seq405")
        self.assertEqual(record.layer_delta, 0)
        self.assertEqual(record.angle_delta_deg, 3)

    def test_camera_projection_uses_world_to_camera_from_r_cam_to_world(self):
        intrinsic = np.eye(3)
        cam_position = np.array([1.0, 2.0, 3.0])
        r_cam_to_world = np.eye(3)

        projection = ptz.camera_projection_matrix_from_labels(intrinsic, cam_position, r_cam_to_world)

        expected = np.array([[1.0, 0.0, 0.0, -1.0], [0.0, 1.0, 0.0, -2.0], [0.0, 0.0, 1.0, -3.0]])
        self.assertTrue(np.allclose(projection, expected))

    def test_extract_sam3d_paper_joints_uses_mhr70_keypoints(self):
        output = {
            "pred_joint_coords": np.arange(127 * 3, dtype=np.float32).reshape(127, 3),
            "pred_keypoints_3d": np.zeros((70, 3), dtype=np.float32),
            "pred_keypoints_2d": np.ones((70, 2), dtype=np.float32),
        }
        output["pred_keypoints_3d"][1] = np.array([1.0, 2.0, 3.0])

        joints3d, joints2d = ptz.extract_sam3d_paper_joints(output)

        self.assertEqual(joints3d.shape, (15, 3))
        self.assertEqual(joints2d.shape, (15, 2))
        self.assertTrue(np.array_equal(joints3d[0], output["pred_keypoints_3d"][1]))


if __name__ == "__main__":
    unittest.main()
