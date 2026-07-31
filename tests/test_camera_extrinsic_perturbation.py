import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = (
    Path(__file__).resolve().parents[1]
    / "archive/paper_packages/paper_ieee_access_20260729/ieeeaccess_submission/scripts"
)
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

spec = importlib.util.spec_from_file_location("access_unity_robustness", SCRIPT_DIR / "evaluate_large_unity_robustness.py")
robustness = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(robustness)


class CameraExtrinsicPerturbationTests(unittest.TestCase):
    def test_zero_extrinsic_perturbation_preserves_projection_matrix(self):
        k = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
        world_to_camera = np.array(
            [
                [1.0, 0.0, 0.0, 0.25],
                [0.0, 1.0, 0.0, -0.50],
                [0.0, 0.0, 1.0, 2.00],
                [0.0, 0.0, 0.0, 1.00],
            ]
        )

        projection = robustness.perturbed_projection_from_components(
            k,
            world_to_camera,
            rotation_deg=0.0,
            translation_units=0.0,
            rng=np.random.default_rng(7),
        )

        expected = robustness.projection_from_components(k, world_to_camera)
        self.assertTrue(np.allclose(projection, expected))

    def test_nonzero_extrinsic_perturbation_is_deterministic_and_rigid(self):
        world_to_camera = np.eye(4)

        perturbed_a = robustness.perturb_world_to_camera(
            world_to_camera,
            rotation_deg=2.0,
            translation_units=0.05,
            rng=np.random.default_rng(11),
        )
        perturbed_b = robustness.perturb_world_to_camera(
            world_to_camera,
            rotation_deg=2.0,
            translation_units=0.05,
            rng=np.random.default_rng(11),
        )
        perturbed_c = robustness.perturb_world_to_camera(
            world_to_camera,
            rotation_deg=2.0,
            translation_units=0.05,
            rng=np.random.default_rng(12),
        )

        rotation = perturbed_a[:3, :3]
        self.assertTrue(np.allclose(perturbed_a, perturbed_b))
        self.assertFalse(np.allclose(perturbed_a, perturbed_c))
        self.assertTrue(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8))
        self.assertTrue(np.allclose(perturbed_a[3], [0.0, 0.0, 0.0, 1.0]))
        self.assertAlmostEqual(np.linalg.norm(perturbed_a[:3, 3]), 0.05)


if __name__ == "__main__":
    unittest.main()
