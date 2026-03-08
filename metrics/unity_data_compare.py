#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/code/metrics/unity_data_compare copy.py
Project: /workspace/code/metrics
Created Date: Sunday March 8th 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Sunday March 8th 2026 4:14:05 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''
import argparse
import json
from pathlib import Path

import numpy as np

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

DEFAULT_PATHS = {
    "sam_l": "/workspace/data/sam3d_body_results/unity/male/left_sam_3d_body_outputs.npz",
    "sam_r": "/workspace/data/sam3d_body_results/unity/male/right_sam_3d_body_outputs.npz",
    "gt_2d_l": "/workspace/data/unity_data/RecordingsPose/cam_left camera/male_kpt2d_left camera_trimmed.jsonl",
    "gt_2d_r": "/workspace/data/unity_data/RecordingsPose/cam_right camera/male_kpt2d_right camera_trimmed.jsonl",
    "gt_3d": "/workspace/data/unity_data/RecordingsPose/male_pose3d_trimmed.jsonl",
}

DEFAULT_FUSED_NPY = (
    "/workspace/data/fused_smoothed_results/unity_pairs/male/left__right_smoothed.npy"
)
DEFAULT_RAW_FUSED_NPY = (
    "/workspace/data/fused_smoothed_results/unity_pairs/male/left__right_fused.npy"
)
DEFAULT_REPORT_PATH = (
    "logs/unity_before_after_fusion_report.txt"
)


def get_unity_gt_3d_dict(gt_2d_raw, gt_3d_raw, height=1080):
    """Build GT 3D dict in SAM3D coordinates for selected joints."""
    _ = gt_2d_raw, height  # kept for call-site compatibility
    name_to_id = {v: k for k, v in UNITY_MHR70_MAPPING.items()}
    return {
        name_to_id[item["name"]]: np.array(
            [-float(item["z"]), -float(item["y"]), float(item["x"])],
            dtype=np.float64,
        )
        for item in gt_3d_raw.get("joints3d", [])
        if item["name"] in name_to_id
    }


def get_unity_gt_2d_dict(gt_2d_raw, height=1080):
    """Build GT 2D dict in image pixel coordinates for selected joints."""
    name_to_id = {v: k for k, v in UNITY_MHR70_MAPPING.items()}
    return {
        name_to_id[item["name"]]: np.array(
            [float(item["u"]), height - float(item["v"])],
            dtype=np.float64,
        )
        for item in gt_2d_raw.get("joints2d", [])
        if item["name"] in name_to_id
    }


def calculate_mpjpe(pred_dict, gt_dict):
    common_ids = set(pred_dict.keys()) & set(gt_dict.keys())
    if not common_ids:
        return np.nan
    errors = [
        np.linalg.norm(np.asarray(pred_dict[j]) - np.asarray(gt_dict[j]))
        for j in common_ids
    ]
    return float(np.mean(errors))


def calculate_per_joint_errors(pred_dict, gt_dict):
    per_joint_err = {}
    common_ids = sorted(set(pred_dict.keys()) & set(gt_dict.keys()))
    for jid in common_ids:
        p = np.asarray(pred_dict[jid], dtype=np.float64)
        g = np.asarray(gt_dict[jid], dtype=np.float64)
        if not (np.isfinite(p).all() and np.isfinite(g).all()):
            per_joint_err[jid] = np.nan
        else:
            per_joint_err[jid] = float(np.linalg.norm(p - g))
    return per_joint_err


def init_joint_stat_container(target_ids):
    return {jid: [] for jid in target_ids}


def accumulate_joint_errors(stat_container, per_joint_err):
    for jid, err in per_joint_err.items():
        stat_container.setdefault(jid, []).append(err)


def summarize_joint_errors(stat_container):
    summary = {}
    for jid, arr in stat_container.items():
        a = np.asarray(arr, dtype=np.float64)
        a = a[np.isfinite(a)]
        if a.size == 0:
            summary[jid] = {"mean": np.nan, "std": np.nan, "median": np.nan, "n": 0}
        else:
            summary[jid] = {
                "mean": float(np.mean(a)),
                "std": float(np.std(a)),
                "median": float(np.median(a)),
                "n": int(a.size),
            }
    return summary


def _format_joint_error_table(summary, mapping_dict, title):
    lines = [
        f"\n--- {title} ---",
        f"{'ID':>4}  {'Joint':<16}  {'mean':>10}  {'std':>10}  {'median':>10}  {'n':>4}",
        "-" * 62,
    ]
    for jid in sorted(summary.keys()):
        s = summary[jid]
        lines.append(
            f"{jid:>4}  {mapping_dict.get(jid, 'Unknown'):<16}"
            f"  {s['mean']:>10.3f}  {s['std']:>10.3f}  {s['median']:>10.3f}  {s['n']:>4}"
        )
    return lines


def _safe_pct_improvement(baseline, target):
    if not np.isfinite(baseline) or baseline == 0:
        return np.nan
    return (baseline - target) / baseline * 100.0


def sam3d_3d_array_to_dict(pred_3d_array, target_ids):
    pred_3d_array = np.asarray(pred_3d_array)
    out = {}
    for i, jid in enumerate(target_ids):
        if i < len(pred_3d_array):
            out[jid] = pred_3d_array[i].astype(np.float64)
    return out


def sam3d_2d_array_to_dict(pred_2d_array, target_ids):
    pred_2d_array = np.asarray(pred_2d_array)
    out = {}
    for i, jid in enumerate(target_ids):
        if i < len(pred_2d_array):
            out[jid] = pred_2d_array[i].astype(np.float64)
    return out


def load_sequence_from_npy(npy_path, target_ids):
    arr = np.load(npy_path, allow_pickle=False)
    seq = []
    for t in range(arr.shape[0]):
        seq.append(sam3d_3d_array_to_dict(arr[t], target_ids))
    return seq


def run_before_after_fusion_analysis(
    fused_npy_path=DEFAULT_FUSED_NPY,
    raw_fused_npy_path=DEFAULT_RAW_FUSED_NPY,
    report_path=DEFAULT_REPORT_PATH,
    paths=None,
):
    if paths is None:
        paths = DEFAULT_PATHS

    sam_l = np.load(paths["sam_l"], allow_pickle=True)["arr_0"]
    sam_r = np.load(paths["sam_r"], allow_pickle=True)["arr_0"]
    raw_fused_seq = load_sequence_from_npy(raw_fused_npy_path, TARGET_IDS)
    fused_seq = load_sequence_from_npy(fused_npy_path, TARGET_IDS)

    gt_2d_l = [
        json.loads(line)
        for line in open(paths["gt_2d_l"], "r", encoding="utf-8-sig")
    ]
    gt_2d_r = [
        json.loads(line)
        for line in open(paths["gt_2d_r"], "r", encoding="utf-8-sig")
    ]
    gt_3d = [
        json.loads(line)
        for line in open(paths["gt_3d"], "r", encoding="utf-8-sig")
    ]

    num_frames = min(
        len(sam_l),
        len(sam_r),
        len(raw_fused_seq),
        len(fused_seq),
        len(gt_2d_l),
        len(gt_2d_r),
        len(gt_3d),
    )
    sam_l = sam_l[:num_frames]
    sam_r = sam_r[:num_frames]
    raw_fused_seq = raw_fused_seq[:num_frames]
    fused_seq = fused_seq[:num_frames]

    results = {
        "L_2D": [],
        "R_2D": [],
        "L_3D": [],
        "R_3D": [],
        "RAW_FUSED_3D": [],
        "FUSED_3D": [],
    }
    joint_stats = {
        "L_2D": init_joint_stat_container(TARGET_IDS),
        "R_2D": init_joint_stat_container(TARGET_IDS),
        "L_3D": init_joint_stat_container(TARGET_IDS),
        "R_3D": init_joint_stat_container(TARGET_IDS),
        "RAW_FUSED_3D": init_joint_stat_container(TARGET_IDS),
        "FUSED_3D": init_joint_stat_container(TARGET_IDS),
    }

    for i in range(num_frames):
        g2d_l = get_unity_gt_2d_dict(gt_2d_l[i])
        g2d_r = get_unity_gt_2d_dict(gt_2d_r[i])
        g3d = get_unity_gt_3d_dict(gt_2d_l[i], gt_3d[i])

        p2d_l = sam3d_2d_array_to_dict(
            np.asarray(sam_l[i]["pred_keypoints_2d"])[TARGET_IDS], TARGET_IDS
        )
        p2d_r = sam3d_2d_array_to_dict(
            np.asarray(sam_r[i]["pred_keypoints_2d"])[TARGET_IDS], TARGET_IDS
        )
        p3d_l = sam3d_3d_array_to_dict(sam_l[i]["pred_keypoints_3d"], TARGET_IDS)
        p3d_r = sam3d_3d_array_to_dict(sam_r[i]["pred_keypoints_3d"], TARGET_IDS)
        p3d_raw_f = raw_fused_seq[i]
        p3d_f = fused_seq[i]

        results["L_2D"].append(calculate_mpjpe(p2d_l, g2d_l))
        results["R_2D"].append(calculate_mpjpe(p2d_r, g2d_r))
        results["L_3D"].append(calculate_mpjpe(p3d_l, g3d))
        results["R_3D"].append(calculate_mpjpe(p3d_r, g3d))
        results["RAW_FUSED_3D"].append(calculate_mpjpe(p3d_raw_f, g3d))
        results["FUSED_3D"].append(calculate_mpjpe(p3d_f, g3d))

        accumulate_joint_errors(
            joint_stats["L_2D"], calculate_per_joint_errors(p2d_l, g2d_l)
        )
        accumulate_joint_errors(
            joint_stats["R_2D"], calculate_per_joint_errors(p2d_r, g2d_r)
        )
        accumulate_joint_errors(
            joint_stats["L_3D"], calculate_per_joint_errors(p3d_l, g3d)
        )
        accumulate_joint_errors(
            joint_stats["R_3D"], calculate_per_joint_errors(p3d_r, g3d)
        )
        accumulate_joint_errors(
            joint_stats["RAW_FUSED_3D"], calculate_per_joint_errors(p3d_raw_f, g3d)
        )
        accumulate_joint_errors(
            joint_stats["FUSED_3D"], calculate_per_joint_errors(p3d_f, g3d)
        )

    summary_l2d = summarize_joint_errors(joint_stats["L_2D"])
    summary_r2d = summarize_joint_errors(joint_stats["R_2D"])
    summary_l = summarize_joint_errors(joint_stats["L_3D"])
    summary_r = summarize_joint_errors(joint_stats["R_3D"])
    summary_raw_f = summarize_joint_errors(joint_stats["RAW_FUSED_3D"])
    summary_f = summarize_joint_errors(joint_stats["FUSED_3D"])

    left_2d = float(np.nanmean(results["L_2D"]))
    right_2d = float(np.nanmean(results["R_2D"]))
    left_3d = float(np.nanmean(results["L_3D"]))
    right_3d = float(np.nanmean(results["R_3D"]))
    raw_fused_3d = float(np.nanmean(results["RAW_FUSED_3D"]))
    fused_3d = float(np.nanmean(results["FUSED_3D"]))
    best_single = min(left_3d, right_3d)

    report_lines = [
        "=" * 84,
        "Unity Before/After Fusion Comparison Report",
        "=" * 84,
        f"Frames used: {num_frames}",
        f"Raw fused file: {raw_fused_npy_path}",
        f"Fused file: {fused_npy_path}",
        "",
        "[Overall 2D MPJPE]",
        f"  LEFT  2D MPJPE: {left_2d:.4f} px",
        f"  RIGHT 2D MPJPE: {right_2d:.4f} px",
        "",
        "[Overall 3D MPJPE]",
        f"  LEFT  (pre-fusion):  {left_3d:.6f}",
        f"  RIGHT (pre-fusion):  {right_3d:.6f}",
        f"  RAW_FUSED (pre-smooth): {raw_fused_3d:.6f}",
        f"  FUSED (post-smooth):    {fused_3d:.6f}",
        "",
        "[Relative Change of FUSED (post-smooth)]",
        (
            f"  vs LEFT       : {_safe_pct_improvement(left_3d, fused_3d):.3f}% "
            "(positive=improvement, negative=degradation)"
        ),
        (
            f"  vs RIGHT      : {_safe_pct_improvement(right_3d, fused_3d):.3f}% "
            "(positive=improvement, negative=degradation)"
        ),
        (
            f"  vs BEST_SINGLE: {_safe_pct_improvement(best_single, fused_3d):.3f}% "
            "(positive=improvement, negative=degradation)"
        ),
        (
            f"  vs RAW_FUSED  : {_safe_pct_improvement(raw_fused_3d, fused_3d):.3f}% "
            "(positive=improvement, negative=degradation)"
        ),
    ]

    report_lines.extend(
        _format_joint_error_table(
            summary_l2d, UNITY_MHR70_MAPPING, "LEFT View Per-Joint 2D Error"
        )
    )
    report_lines.extend(
        _format_joint_error_table(
            summary_r2d, UNITY_MHR70_MAPPING, "RIGHT View Per-Joint 2D Error"
        )
    )
    report_lines.extend(
        _format_joint_error_table(
            summary_l, UNITY_MHR70_MAPPING, "LEFT View Per-Joint 3D Error (Pre-Fusion)"
        )
    )
    report_lines.extend(
        _format_joint_error_table(
            summary_r, UNITY_MHR70_MAPPING, "RIGHT View Per-Joint 3D Error (Pre-Fusion)"
        )
    )
    report_lines.extend(
        _format_joint_error_table(
            summary_raw_f,
            UNITY_MHR70_MAPPING,
            "RAW_FUSED View Per-Joint 3D Error (Pre-Smooth)",
        )
    )
    report_lines.extend(
        _format_joint_error_table(
            summary_f,
            UNITY_MHR70_MAPPING,
            "FUSED View Per-Joint 3D Error (Post-Smooth)",
        )
    )

    report = "\n".join(report_lines)
    print(report)

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"\n[Saved] report: {report_path}")

    return {
        "num_frames": num_frames,
        "left_2d": left_2d,
        "right_2d": right_2d,
        "left_3d": left_3d,
        "right_3d": right_3d,
        "raw_fused_3d": raw_fused_3d,
        "fused_3d": fused_3d,
        "report_path": str(report_path),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Compare pre-fusion and post-fusion 3D results against GT")
    parser.add_argument("--fused-npy", default=DEFAULT_FUSED_NPY, help="Saved fused (T,J,3) npy path")
    parser.add_argument(
        "--raw-fused-npy",
        default=DEFAULT_RAW_FUSED_NPY,
        help="Saved raw fused (pre-smoothing) (T,J,3) npy path",
    )
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH, help="Output report text path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_before_after_fusion_analysis(
        fused_npy_path=args.fused_npy,
        raw_fused_npy_path=args.raw_fused_npy,
        report_path=args.report_path,
    )
