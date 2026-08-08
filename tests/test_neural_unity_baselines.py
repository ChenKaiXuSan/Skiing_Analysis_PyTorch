import unittest
from pathlib import Path

import numpy as np

from paper_array_20260730.array_submission.scripts.unity.evaluate_large_unity_benchmark import PairRecord
from paper_array_20260730.array_submission.scripts.unity import train_neural_unity_baselines as nub
from paper_array_20260730.array_submission.scripts.unity.train_neural_unity_baselines import (
    PairArrays,
    build_cross_view_fusion_features,
)


def make_record(pair_id: str, cam1: str, cam2: str, angle_delta: int, layer_delta: int = 1) -> PairRecord:
    return PairRecord(
        pair_id=pair_id,
        person_id="P",
        action_id="A",
        action_dir=Path("/tmp/action"),
        cam1=cam1,
        cam2=cam2,
        layer_delta=layer_delta,
        angle_delta_deg=angle_delta,
    )


class NeuralUnityBaselineTests(unittest.TestCase):
    def test_cross_view_features_can_omit_camera_metadata(self):
        n_frames = 2
        n_joints = 15
        arrays = {
            "sam2d_left": np.full((n_frames, n_joints, 2), 100.0),
            "sam2d_right": np.full((n_frames, n_joints, 2), 104.0),
            "sam3d_left": np.ones((n_frames, n_joints, 3)),
            "sam3d_right": np.ones((n_frames, n_joints, 3)) * 1.2,
            "gt3d": np.ones((n_frames, n_joints, 3)) * 1.1,
        }
        record = make_record("wide", "L0_A000", "L2_A120", angle_delta=120, layer_delta=2)
        item = PairArrays(record, arrays)

        with_camera = build_cross_view_fusion_features(Path("/tmp"), item, include_camera_metadata=True)
        without_camera = build_cross_view_fusion_features(Path("/tmp"), item, include_camera_metadata=False)

        self.assertEqual(with_camera.x.shape[1] - without_camera.x.shape[1], 2)
        self.assertEqual(without_camera.x.shape[0], n_frames * n_joints)

    def test_angle_generalization_split_holds_out_named_angle_bin(self):
        self.assertTrue(hasattr(nub, "filter_records_for_generalization"))
        near = make_record("near", "L0_A000", "L1_A030", angle_delta=30)
        mid = make_record("mid", "L0_A000", "L1_A060", angle_delta=60)
        wide = make_record("wide", "L0_A000", "L1_A150", angle_delta=150)
        records = [near, mid, wide]

        train_records = nub.filter_records_for_generalization(records, split="train", holdout_angle_bin="wide")
        test_records = nub.filter_records_for_generalization(records, split="test", holdout_angle_bin="wide")

        self.assertEqual([r.pair_id for r in train_records], ["near", "mid"])
        self.assertEqual([r.pair_id for r in test_records], ["wide"])

    def test_camera_generalization_split_holds_out_records_touching_camera(self):
        self.assertTrue(hasattr(nub, "filter_records_for_generalization"))
        touches_camera = make_record("touches", "L0_A000", "L1_A060", angle_delta=60)
        other = make_record("other", "L2_A120", "L3_A180", angle_delta=60)
        records = [touches_camera, other]

        train_records = nub.filter_records_for_generalization(records, split="train", holdout_camera_id="capture_L0_A000")
        test_records = nub.filter_records_for_generalization(records, split="test", holdout_camera_id="capture_L0_A000")

        self.assertEqual([r.pair_id for r in train_records], ["other"])
        self.assertEqual([r.pair_id for r in test_records], ["touches"])

    def test_selected_records_from_fold_applies_generalization_filter(self):
        self.assertTrue(hasattr(nub, "selected_records_from_fold"))
        fold = {
            "train": [
                {"person_id": "P", "action_id": "A", "sequence_meta_path": "/tmp/action/meta/sequence.json", "cam1_id": "capture_L0_A000", "cam2_id": "capture_L1_A020"},
                {"person_id": "P", "action_id": "A", "sequence_meta_path": "/tmp/action/meta/sequence.json", "cam1_id": "capture_L0_A000", "cam2_id": "capture_L1_A140"},
            ]
        }

        selected = nub.selected_records_from_fold(fold, "train", 8, seed=0, holdout_angle_bin="wide")

        self.assertEqual([r.angle_delta_deg for r in selected], [20])


if __name__ == "__main__":
    unittest.main()
