# Skiing Motion Analysis

This repository contains a research-oriented pipeline for skiing motion
analysis from monocular and dual-view videos. It focuses on 3D human pose
estimation, multi-view reconstruction, pose fusion, temporal smoothing,
biomechanical angle analysis, and visualization.

The codebase is organized as an experimental computer vision system rather than
a single packaged application. Most entry points are Hydra-based module scripts
that read configuration from `configs/`.

## What This Project Does

The main goal is to reconstruct and analyze skier motion from video. The
pipeline supports several complementary reconstruction routes:

- Preprocess raw videos into intermediate `.pt` files with detections, 2D
  keypoints, bounding boxes, depth, optical flow, and video metadata.
- Estimate 3D body pose with SAM3D Body in MHR70 format.
- Fuse left/right 3D pose estimates using confidence and cross-view
  consistency.
- Smooth fused sequences over time.
- Compute skiing-oriented kinematic metrics such as knee angles, hip angles,
  torso tilt, torso-knee angle, left/right knee difference, and elbow distance
  from the body midline.
- Generate visualizations and evaluation outputs for analysis.

## Pipeline Overview

```text
raw skiing videos
  -> prepare_dataset
  -> SAM3D Body
  -> dual-view alignment, fusion, and smoothing
  -> temporal smoothing
  -> angle analysis, metrics, and visualization
```

Typical data artifacts include:

- `.pt`: preprocessed video records with frames and detection outputs.
- `.npz`: model inference outputs such as SAM3D Body results.
- `.npy`: fused or smoothed 3D keypoint sequences.
- `.csv` and `.png`: angle-analysis tables and plots.

## Main Modules

| Path | Purpose |
| --- | --- |
| `prepare_dataset/` | Preprocess raw videos with YOLO, Detectron2, depth estimation, optical flow, tracking, and metadata export. |
| `prepare_side_results/` | Run SAM3D Body on video data and save 2D/3D body predictions. |
| `fuse/` | Fuse two SAM3D Body pose streams with rigid alignment, confidence weighting, and EMA smoothing. |
| `angle/` | Compute skiing motion metrics from 3D MHR70 keypoints. |
| `metrics/` | Compare predicted results with Unity or ground-truth data. |
| `vis_3d_kpt/` | Visualize 3D keypoint sequences and skeletons. |
| `analysis/` | Research notebooks for experiments and plotting. |

## Installation

Create a Python environment with PyTorch and install the project dependencies:

```bash
pip install -r requirements.txt
```

Some modules require additional model checkpoints and external packages. Common
checkpoint paths are configured under `configs/`, for example:

- YOLO checkpoints in `configs/prepare_dataset.yaml`
- SAM3D Body checkpoints in `configs/sam3d_body.yaml`

The default configs currently assume a `/workspace/data` and `/workspace/code`
layout. Update the relevant YAML files before running on a different machine.

## Common Commands

Run commands from the repository root. Most scripts should be launched with
`python -m` so Hydra resolves module imports and config paths correctly.

Preprocess raw videos:

```bash
python -m prepare_dataset.main
```

Run SAM3D Body inference:

```bash
python -m prepare_side_results.main
```

Fuse SAM3D Body left/right results:

```bash
python -m fuse.main_raw \
  --input-root /workspace/data/dual_view_pose/sam3d_body_results/person \
  --output-root /workspace/data/dual_view_pose/fused_smoothed_results
```

Run angle analysis:

```bash
python -m angle.main
```

## Configuration

The main Hydra configs are:

- `configs/prepare_dataset.yaml`
- `configs/sam3d_body.yaml`
- `configs/fuse.yaml`
- `configs/qwen_image_edit.yaml`

Use these files to set input/output paths, model checkpoints, GPU IDs,
visualization options, and optimization parameters.

## Notes

- This repository is under active research development, so some modules are
  experimental and may contain hard-coded assumptions about view names,
  subject-folder layout, or camera calibration.
- Left/right view ordering is module-dependent. Check the corresponding
  `main.py` before preparing new data.
- Large checkpoints, datasets, and generated logs are not expected to live in
  the Git repository.
- A more detailed process description is available in
  `doc/process_documentation.md`.
- A practical project inventory and cleanup guide is available in
  `docs/project_inventory.md`.

## License

See `LICENSE`.
