#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/workspace/code/angle')

from main import (
    ID_TO_INDEX, UNITY_MHR70_MAPPING, visualize_3d_keypoints
)

# Test with run_3 only
input_path = Path("/workspace/data/fused_smoothed_results/run_3.npy")
output_dir = Path("/workspace/data/angle_outputs/run_3")

print(f"Loading keypoints from {input_path}")
kpts = np.load(input_path)
print(f"Keypoints shape: {kpts.shape}")
print(f"Data type: {kpts.dtype}")

print("Creating visualization...")
visualization_dir = output_dir / "skeleton_visualization"
try:
    visualize_3d_keypoints(kpts, ID_TO_INDEX, visualization_dir, num_frames_to_save=3)
    print("Visualization complete!")
except Exception as e:
    print(f"Error during visualization: {e}")
    import traceback
    traceback.print_exc()

# Check output
print(f"\nChecking output directory: {visualization_dir}")
if visualization_dir.exists():
    files = list(visualization_dir.glob("*.png"))
    print(f"Found {len(files)} PNG files")
    for f in files:
        print(f"  - {f.name}")
else:
    print("Output directory not created!")
