#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Batch pairwise camera fusion for Unity SAM 3D keypoints."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from confidence import crossview_consistency_confidence, weakpersp_reproj_confidence
from fuse import fuse_frame_3d, temporal_smooth_ema
from save import save_smoothed_results


UNITY_MHR70_MAPPING = {
    1: "Bone_Eye_L",
    2: "Bone_Eye_R",
    5: "Upperarm_L",
    6: "Upperarm_R",
    7: "lowerarm_l",
    8: "lowerarm_r",
    9: "Thigh_L",
    10: "Thigh_R",
    11: "calf_l",
    12: "calf_r",
    13: "Foot_L",
    14: "Foot_R",
    41: "Hand_R",
    62: "Hand_L",
    69: "neck_01",
}
TARGET_IDS = list(UNITY_MHR70_MAPPING.keys())

# Indices for cross-view canonicalization in TARGET_IDS order.
IDX_PELVIS = 14
IDX_LHIP = 11
IDX_RHIP = 12
IDX_LSHO = 5
IDX_RSHO = 6


def _discover_npz_files(input_root: Path) -> Dict[str, List[Path]]:
    """Group camera output files by their parent folder relative to input root."""
    groups: Dict[str, List[Path]] = {}
    patterns = ["**/*sam_3d_body_outputs*.npz", "**/*.npz"]
    candidates: List[Path] = []
    for pat in patterns:
        candidates.extend(input_root.glob(pat))
        if candidates:
            break

    for path in sorted({p for p in candidates if p.is_file()}):
        group_key = str(path.parent.relative_to(input_root))
        groups.setdefault(group_key, []).append(path)
    return groups


def _camera_name_from_path(path: Path) -> str:
    stem = path.stem
    suffixes = ["_sam_3d_body_outputs", "_sam_3d_body_output", "_outputs"]
    for suffix in suffixes:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _as_joint_dict(arr: np.ndarray) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for idx, jid in enumerate(TARGET_IDS):
        if idx >= len(arr):
            continue
        out[jid] = np.asarray(arr[idx], dtype=np.float64)
    return out


def _load_sequence(npz_path: Path) -> List[Dict[str, Dict[int, np.ndarray]]]:
    data = np.load(npz_path, allow_pickle=True)
    key = "outputs" if "outputs" in data.files else data.files[0]
    frames = data[key]

    seq: List[Dict[str, Dict[int, np.ndarray]]] = []
    for frame in frames:
        p2d = np.asarray(frame["pred_keypoints_2d"], dtype=np.float64)
        p3d = np.asarray(frame["pred_keypoints_3d"], dtype=np.float64)
        seq.append(
            {
                "p2d": _as_joint_dict(p2d[TARGET_IDS]),
                "p3d": _as_joint_dict(p3d[TARGET_IDS]),
            }
        )
    return seq


def _fuse_pair(
    seq_a: List[Dict[str, Dict[int, np.ndarray]]],
    seq_b: List[Dict[str, Dict[int, np.ndarray]]],
    sigma_px: float,
    sigma_3d: float,
) -> List[Dict[int, np.ndarray]]:
    num_frames = min(len(seq_a), len(seq_b))
    fused_seq: List[Dict[int, np.ndarray]] = []

    for i in range(num_frames):
        p2d_a = seq_a[i]["p2d"]
        p3d_a = seq_a[i]["p3d"]
        p2d_b = seq_b[i]["p2d"]
        p3d_b = seq_b[i]["p3d"]

        conf_a, _, _, _ = weakpersp_reproj_confidence(p3d_a, p2d_a, sigma_px=sigma_px)
        conf_b, _, _, _ = weakpersp_reproj_confidence(p3d_b, p2d_b, sigma_px=sigma_px)

        conf_cross, _, _, _, _ = crossview_consistency_confidence(
            p3d_a,
            p3d_b,
            root_idx=IDX_PELVIS,
            left_hip_idx=IDX_LHIP,
            right_hip_idx=IDX_RHIP,
            left_shoulder_idx=IDX_LSHO,
            right_shoulder_idx=IDX_RSHO,
            sigma_3d=sigma_3d,
            scale_mode="hip",
        )

        q_a = np.sqrt(np.clip(conf_a * conf_cross, 0.0, 1.0))
        q_b = np.sqrt(np.clip(conf_b * conf_cross, 0.0, 1.0))
        fused_seq.append(fuse_frame_3d(p3d_a, p3d_b, q_a, q_b, TARGET_IDS))

    return fused_seq


def run_batch(
    input_root: Path,
    output_root: Path,
    sigma_px: float,
    sigma_3d: float,
    alpha: float,
    save_raw_fused: bool,
) -> Tuple[int, int]:
    groups = _discover_npz_files(input_root)
    total_pairs = 0
    total_saved = 0

    for group_name, files in groups.items():
        if len(files) < 2:
            continue

        out_dir = output_root / group_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for path_a, path_b in itertools.combinations(files, 2):
            cam_a = _camera_name_from_path(path_a)
            cam_b = _camera_name_from_path(path_b)
            pair_name = f"{cam_a}__{cam_b}"

            seq_a = _load_sequence(path_a)
            seq_b = _load_sequence(path_b)
            fused_seq = _fuse_pair(seq_a, seq_b, sigma_px=sigma_px, sigma_3d=sigma_3d)
            smooth_seq = temporal_smooth_ema(fused_seq, TARGET_IDS, alpha=alpha)

            if save_raw_fused:
                raw_path = out_dir / f"{pair_name}_fused.npy"
                save_smoothed_results(fused_seq, TARGET_IDS, raw_path)

            smooth_path = out_dir / f"{pair_name}_smoothed.npy"
            save_smoothed_results(smooth_seq, TARGET_IDS, smooth_path)

            print(f"[saved] {smooth_path}")
            total_pairs += 1
            total_saved += 1

    return total_pairs, total_saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pairwise fusion for Unity SAM 3D keypoints and save results."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/workspace/data/sam3d_body_results/unity"),
        help="Root folder containing SAM 3D npz outputs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/workspace/data/fused_smoothed_results/unity_pairs"),
        help="Output folder for pairwise fused results.",
    )
    parser.add_argument("--sigma-px", type=float, default=12.0)
    parser.add_argument("--sigma-3d", type=float, default=0.08)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument(
        "--save-raw-fused",
        action="store_true",
        default=True,
        help="Save pre-smoothing fused sequences (enabled by default).",
    )
    parser.add_argument(
        "--no-save-raw-fused",
        dest="save_raw_fused",
        action="store_false",
        help="Disable saving pre-smoothing fused sequences.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs, saved = run_batch(
        input_root=args.input_root,
        output_root=args.output_root,
        sigma_px=args.sigma_px,
        sigma_3d=args.sigma_3d,
        alpha=args.alpha,
        save_raw_fused=args.save_raw_fused,
    )
    if pairs == 0:
        print(f"No camera pairs found under: {args.input_root}")
        return
    print(f"Done. fused_pairs={pairs}, saved_files={saved}")


if __name__ == "__main__":
    main()
