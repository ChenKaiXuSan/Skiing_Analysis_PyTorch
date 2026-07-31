# Skiing Analysis Project Inventory

Generated: 2026-07-31

This repository is best treated as a research workspace rather than a single
packaged application. It contains the dual-view skiing 3D pose pipeline, paper
experiment packages, result-generation scripts, and several experimental model
routes.

## Recommended Top-Level Organization

```text
core pipeline
  prepare_dataset/        raw video -> .pt preprocessing
  prepare_side_results/   SAM3D Body inference
  fuse/                   dual-view 3D fusion and smoothing
  angle/                  skiing kinematic analysis
  metrics/                Unity and no-GT real-world evaluation utilities
  vis_3d_kpt/             fused 3D skeleton visualization

paper packages
  paper_resubmission_20260728/
  paper_array_20260730/
  archive/paper_packages/paper_ieee_access_20260729/
  archive/paper_packages/paper_sport_engineering/
  reviewer2_fusion_ablation.py
  docs/reviewer2_fusion_ablation/

experimental or backup routes
  VideoPose3D/
  triangulation/
  vggt/
  bundle_adjustment/
  analysis/

environment and artifacts
  configs/
  tests/
  docker/
  ckpt/
  logs/
```

## Active Main Line

These components are still relevant to the current manuscript and benchmark
work.

| Path | Role | Current status |
| --- | --- | --- |
| `prepare_dataset/` | Converts raw videos into `.pt` records with frames, detections, depth, optical flow, and metadata. | Keep. Upstream data-production tool; uses Hydra and `/workspace`-style paths. |
| `prepare_side_results/` | Runs SAM3D Body on videos and saves 2D/3D predictions. | Keep. Core source for fusion and benchmark inputs. |
| `fuse/` | Aligns left/right SAM3D 3D streams, computes confidence, fuses joints, and applies EMA smoothing. | Keep. Core algorithm implementation. |
| `angle/` | Computes skiing turn segments, joint angles, torso-knee measures, and elbow-midline metrics. | Keep. Used for real-world kinematic analysis. |
| `metrics/` | Compares Unity predictions and evaluates real no-GT fused results. | Keep as utilities. Some functions are superseded by paper-package scripts. |
| `vis_3d_kpt/` | Batch visualization for fused/smoothed 3D sequences and pre-fusion views. | Keep as a utility. |
| `tests/` | Unit tests for current benchmark and ablation helpers. | Keep. Focus is paper experiment logic, not full model inference. |

## Current Paper Work

| Path | Role | Current status |
| --- | --- | --- |
| `paper_resubmission_20260728/` | Main manuscript package with Unity benchmark, robustness, supervised baselines, real-world metrics, tables, and PDF. | Keep if this remains the authoritative resubmission package. |
| `paper_array_20260730/` | Elsevier Array-format submission package with flattened figures/tables and copied scripts. | Keep if Array is the target venue package. |
| `archive/paper_packages/paper_ieee_access_20260729/` | IEEE Access-format package. | Archived from the top level; keep here unless actively submitting to IEEE Access. |
| `archive/paper_packages/paper_sport_engineering/` | Sports Engineering-style manuscript and figures. | Archived from the top level; keep here unless actively revising this venue version. |
| `reviewer2_fusion_ablation.py` | Standalone reviewer-response ablation/statistics script using `fuse.confidence`. | Keep near paper scripts or move under a paper package after updating imports/tests. |
| `docs/reviewer2_fusion_ablation/` | Generated ablation CSV and markdown report. | Keep as evidence if tied to reviewer response; otherwise archive with the relevant paper package. |

## Deprecated Code Routes

The user confirmed these routes are no longer needed for the active project. They should be moved to `archive/experimental_routes/` once root-owned directory permissions are available.

| Path | Role | Current status |
| --- | --- | --- |
| `VideoPose3D/` | COCO17 VideoPose3D route plus no-extrinsics fusion baseline. | Deprecated; move to `archive/experimental_routes/VideoPose3D/` with sudo. |
| `triangulation/` | Two-view triangulation from 2D keypoints and camera parameters. | Deprecated; move to `archive/experimental_routes/triangulation/` with sudo. |
| `vggt/` | VGGT single/multi-view reconstruction route. | Deprecated; move to `archive/experimental_routes/vggt/` with sudo. |
| `bundle_adjustment/` | Multi-modal geometric refinement using video, `.pt`, SAM3D, VGGT, and VideoPose3D inputs. | Deprecated; move to `archive/experimental_routes/bundle_adjustment/` with sudo. |
| `analysis/` | Notebooks for exploration and plotting. | Keep outside the main runnable path; archive old notebooks if not reproducible. |

## Cleanup Candidates

These should not be deleted blindly because the current worktree already has
many user-level changes. Confirm intent, then clean in a separate commit.

| Item | Reason | Suggested action |
| --- | --- | --- |
| `front_side/` | Git status shows this tracked directory as deleted. | If intentional, commit deletion. If not, restore from git before further cleanup. |
| `prepare_front_results/` | Git status shows this tracked directory as deleted. | Same as above. Likely superseded by newer SAM3D/fusion workflow. |
| `image_edit/` | Git status shows this tracked directory as deleted while `configs/qwen_image_edit.yaml` remains. | Either restore the module or remove/archive the orphan config. |
| `configs/qwen_image_edit.yaml` | Orphan config if `image_edit/` remains deleted. | Archive or delete with the image-edit route. |
| `ckpt/` | Large checkpoints and temporary files, some root-owned. | Keep out of git; consider moving to external data/cache storage. |
| `logs/` | Generated reports, root-owned. | Keep out of git; copy paper evidence into paper package only when needed. |
| `.mypy_cache/`, `__pycache__/` | Generated caches. | Safe to remove locally when permissions allow. |

## Suggested Final Layout

For a cleaner repo, use this target shape over time:

```text
configs/
docs/
paper/
  resubmission_20260728/
  array_20260730/
  archive/
src/
  pipeline/
  evaluation/
  visualization/
experimental/
  videopose3d/
  triangulation/
  vggt/
  bundle_adjustment/
tests/
```

Do the physical moves only after the current paper package is frozen, because
the LaTeX packages and tests currently use direct relative paths.

## Completed Moves

- Moved `paper_ieee_access_20260729/` to `archive/paper_packages/paper_ieee_access_20260729/`.
- Moved `paper_sport_engineering/` to `archive/paper_packages/paper_sport_engineering/`.
- Updated the IEEE robustness test to load from the archived path.
- Added `paper_*/` and `archive/paper_packages/` to `.gitignore` so paper packages are local artifacts by default.

## Blocked Moves

These directories are root-owned and normal `mv` failed with permission denied in this session:

```bash
sudo mv VideoPose3D triangulation vggt bundle_adjustment archive/experimental_routes/
```

## Immediate Low-Risk Actions

1. Fix broken test paths after paper package restructuring.
2. Add a README note pointing readers to this inventory.
3. Keep large data, checkpoints, logs, caches, and compiled LaTeX byproducts out
   of git unless they are deliberate submission artifacts.
4. Decide which paper package is authoritative before deleting or moving any
   other paper directory.
