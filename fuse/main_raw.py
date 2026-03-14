#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Batch fusion for real person data (pro_*/run_* formats)."""

import argparse
from pathlib import Path

import numpy as np

from .confidence import (
    crossview_consistency_confidence,
    weakpersp_reproj_confidence,
)
from .fuse import fuse_frame_3d, temporal_smooth_ema
from .load.load_raw import load_raw
from .save import save_smoothed_results

IDX_PELVIS = 14
IDX_LHIP = 11
IDX_RHIP = 12
IDX_LSHO = 5
IDX_RSHO = 6


def _resolve_person_paths(person_dir: Path):
    person_name = person_dir.name
    if person_name.startswith("pro"):
        sam_l_path = person_dir / "left"
        sam_r_path = person_dir / "right"
    elif person_name.startswith("run"):
        sam_l_path = person_dir / "osmo_1_sam_3d_body_outputs.npz"
        sam_r_path = person_dir / "osmo_2_sam_3d_body_outputs.npz"
    else:
        return None

    if not sam_l_path.exists() or not sam_r_path.exists():
        return None
    return {"sam_l": str(sam_l_path), "sam_r": str(sam_r_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Fuse real person SAM3D pairs and save raw+smoothed outputs.")
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/workspace/data/dual_view_pose/sam3d_body_results/person"),
        help="Root folder containing pro_*/run_* person directories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/workspace/data/fused_smoothed_results"),
        help="Output folder for fused results.",
    )
    parser.add_argument("--sigma-px", type=float, default=12.0)
    parser.add_argument("--sigma-3d", type=float, default=0.08)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument(
        "--no-adaptive-smooth",
        dest="adaptive_smooth",
        action="store_false",
        help="Disable speed/joint adaptive EMA and use fixed alpha.",
    )
    parser.set_defaults(adaptive_smooth=True)
    parser.add_argument("--smooth-alpha-min", type=float, default=0.45)
    parser.add_argument("--smooth-alpha-max", type=float, default=0.92)
    parser.add_argument("--smooth-speed-gain", type=float, default=0.25)
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
    args.output_root.mkdir(parents=True, exist_ok=True)

    total_people = 0

    for person in sorted(args.input_root.iterdir()):
        if not person.is_dir():
            continue

        person_name = person.name
        print(f"Processing person: {person_name}")

        # if "pro" in person_name:
        #     continue;

        paths = _resolve_person_paths(person)
        if paths is None:
            msg = "Skipped person (unsupported format or missing files):"
            print(msg, person_name)
            continue

        all_frame_results = load_raw(paths)
        if not all_frame_results:
            continue

        # 预先取第一帧来确定关节数量（SAM 输出按索引排列）
        first_frame = next(iter(all_frame_results.values()))
        p3d_first = first_frame["L_3D"]["pred"]
        all_joint_ids = list(range(len(p3d_first)))

        fused_seq = []

        for frame_data in all_frame_results.values():
            p2d_l_raw = frame_data["L_2D"]["pred"]
            p3d_l_raw = frame_data["L_3D"]["pred"]
            p2d_r_raw = frame_data["R_2D"]["pred"]
            p3d_r_raw = frame_data["R_3D"]["pred"]

            # 计算置信度
            p3d_l_conf1, _, _, _ = weakpersp_reproj_confidence(
                p3d_l_raw,
                p2d_l_raw,
                sigma_px=args.sigma_px,
            )

            p3d_r_conf1, _, _, _ = weakpersp_reproj_confidence(
                p3d_r_raw,
                p2d_r_raw,
                sigma_px=args.sigma_px,
            )

            conf2, _, _, _, _ = crossview_consistency_confidence(
                p3d_l_raw,
                p3d_r_raw,
                root_idx=IDX_PELVIS,
                left_hip_idx=IDX_LHIP,
                right_hip_idx=IDX_RHIP,
                left_shoulder_idx=IDX_LSHO,
                right_shoulder_idx=IDX_RSHO,
                sigma_3d=args.sigma_3d,
                scale_mode="hip",
            )

            q_l_data = np.sqrt(p3d_l_conf1 * conf2)
            q_r_data = np.sqrt(p3d_r_conf1 * conf2)

            fused_3d = fuse_frame_3d(
                p3d_l_raw, p3d_r_raw, q_l_data, q_r_data, all_joint_ids
            )
            fused_seq.append(fused_3d)

        smooth_seq = temporal_smooth_ema(
            fused_seq,  # type: ignore[arg-type]
            all_joint_ids,
            alpha=args.alpha,
            adaptive=args.adaptive_smooth,
            alpha_min=args.smooth_alpha_min,
            alpha_max=args.smooth_alpha_max,
            speed_gain=args.smooth_speed_gain,
        )

        if args.save_raw_fused:
            raw_path = args.output_root / f"{person_name}_fused.npy"
            save_smoothed_results(
                fused_seq,  # type: ignore[arg-type]
                all_joint_ids,
                raw_path,
            )
            print(f"[saved] {raw_path}")

        smooth_path = args.output_root / f"{person_name}_smoothed.npy"
        save_smoothed_results(
            smooth_seq,  # type: ignore[arg-type]
            all_joint_ids,
            smooth_path,
        )
        print(f"[saved] {smooth_path}")
        total_people += 1

    print(f"Done. processed_people={total_people}")


if __name__ == "__main__":
    main()
