#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/code/metrics/true_data_compare.py
Project: /workspace/code/metrics
Created Date: Sunday March 8th 2026
Author: Kaixu Chen
-----
Comment:
No-GT quality evaluation for real fused human pose results.
Evaluates temporal smoothing effectiveness without ground truth by comparing
reprojection confidence, cross-view consistency, temporal metrics (speed/jerk),
and bone length stability between raw fused and smoothed results.

Have a good code time :)
-----
Last Modified: Sunday March 8th 2026 7:45:00 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''
import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

FUSE_DIR = Path(__file__).resolve().parents[1] / "fuse"
if str(FUSE_DIR) not in sys.path:
    sys.path.insert(0, str(FUSE_DIR))

from confidence import crossview_consistency_confidence, weakpersp_reproj_confidence

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

# Anatomical indices for cross-view consistency check
IDX_PELVIS = 14
IDX_LHIP = 11
IDX_RHIP = 12
IDX_LSHO = 5
IDX_RSHO = 6

# Kinematic chain edges for bone length stability analysis
BONE_EDGES = [
    (69, 5),
    (5, 7),
    (7, 62),
    (69, 6),
    (6, 8),
    (8, 41),
    (69, 9),
    (9, 11),
    (11, 13),
    (69, 10),
    (10, 12),
    (12, 14),
    (9, 10),
    (5, 6),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fused quality on real data without GT."
    )
    parser.add_argument(
        "--person-root",
        type=Path,
        default=Path("/workspace/data/sam3d_body_results/person"),
        help="Root folder containing pro_*/run_* source SAM outputs.",
    )
    parser.add_argument(
        "--fused-root",
        type=Path,
        default=Path("/workspace/data/fused_smoothed_results/person_pairs_test"),
        help="Folder containing *_fused.npy and *_smoothed.npy.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("logs/true_before_after_fusion_report.txt"),
        help="Output report path.",
    )
    parser.add_argument("--sigma-px", type=float, default=12.0)
    parser.add_argument("--sigma-3d", type=float, default=0.08)
    return parser.parse_args()


def resolve_person_paths(person_dir: Path) -> Optional[Dict[str, Path]]:
    """Resolve left/right SAM output paths for a person directory.
    
    Supports two formats:
    - pro_* format: left/ and right/ subdirectories with frame_*.npz files
    - run_* format: osmo_1_sam_3d_body_outputs.npz and osmo_2_sam_3d_body_outputs.npz
    
    Returns:
        Dict with 'sam_l' and 'sam_r' keys, or None if invalid format.
    """
    name = person_dir.name
    if name.startswith("pro"):
        sam_l = person_dir / "left"
        sam_r = person_dir / "right"
    elif name.startswith("run"):
        sam_l = person_dir / "osmo_1_sam_3d_body_outputs.npz"
        sam_r = person_dir / "osmo_2_sam_3d_body_outputs.npz"
    else:
        return None

    if not sam_l.exists() or not sam_r.exists():
        return None
    return {"sam_l": sam_l, "sam_r": sam_r}


def load_sam_data(path: Path):
    """Load SAM3D output data from either single npz or directory of frame npz files."""
    if path.is_file() and path.suffix == ".npz":
        data = np.load(str(path), allow_pickle=True)
        key = (
            "arr_0"
            if "arr_0" in data.files
            else ("outputs" if "outputs" in data.files else data.files[0])
        )
        outputs = data[key]
        if isinstance(outputs, np.ndarray) and outputs.ndim == 0:
            outputs = [outputs.item()]
        return list(outputs)

    # For pro format, frame-wise npz files under directory.
    frame_files = sorted(path.glob("frame_*_sam_3d_body_outputs.npz"))
    frames = []
    for f in frame_files:
        data = np.load(str(f), allow_pickle=True)
        key = (
            "outputs"
            if "outputs" in data.files
            else ("arr_0" if "arr_0" in data.files else data.files[0])
        )
        out = data[key]
        if isinstance(out, (list, np.ndarray)) and len(out) > 0:
            frames.append(out[0])
        else:
            frames.append(out)
    return frames


def sam3d_array_to_dict(arr: np.ndarray) -> Dict[int, np.ndarray]:
    """Convert SAM3D array to dict mapping joint IDs to coordinates."""
    arr = np.asarray(arr)
    out: Dict[int, np.ndarray] = {}
    for i, jid in enumerate(TARGET_IDS):
        if i < arr.shape[0]:
            out[jid] = np.asarray(arr[i], dtype=np.float64)
    return out


def load_lr_sequences(person_paths: Dict[str, Path]):
    """Load left and right SAM3D sequences for 2D and 3D keypoints."""
    sam_l = load_sam_data(person_paths["sam_l"])
    sam_r = load_sam_data(person_paths["sam_r"])
    n = min(len(sam_l), len(sam_r))

    seq_l_2d = []
    seq_r_2d = []
    seq_l_3d = []
    seq_r_3d = []
    for i in range(n):
        seq_l_2d.append(
            sam3d_array_to_dict(
                np.asarray(sam_l[i]["pred_keypoints_2d"])[TARGET_IDS]
            )
        )
        seq_r_2d.append(
            sam3d_array_to_dict(
                np.asarray(sam_r[i]["pred_keypoints_2d"])[TARGET_IDS]
            )
        )
        seq_l_3d.append(
            sam3d_array_to_dict(
                np.asarray(sam_l[i]["pred_keypoints_3d"])[TARGET_IDS]
            )
        )
        seq_r_3d.append(
            sam3d_array_to_dict(
                np.asarray(sam_r[i]["pred_keypoints_3d"])[TARGET_IDS]
            )
        )

    return seq_l_2d, seq_r_2d, seq_l_3d, seq_r_3d


def load_fused_sequence(npy_path: Path) -> List[Dict[int, np.ndarray]]:
    """Load fused pose sequence from npy file."""
    arr = np.load(str(npy_path), allow_pickle=False)
    out = []
    for t in range(arr.shape[0]):
        out.append(sam3d_array_to_dict(arr[t]))
    return out


def sequence_to_array(seq: List[Dict[int, Iterable[float]]]) -> np.ndarray:
    """Convert sequence of dict frames to (T, J, 3) array."""
    t_len = len(seq)
    arr = np.full((t_len, len(TARGET_IDS), 3), np.nan, dtype=np.float64)
    for t, frame in enumerate(seq):
        for j, jid in enumerate(TARGET_IDS):
            if jid in frame:
                arr[t, j] = np.asarray(frame[jid], dtype=np.float64)
    return arr


def mean_confidence(conf: np.ndarray) -> float:
    """Calculate mean of confidence values ignoring NaN."""
    conf = np.asarray(conf, dtype=np.float64)
    valid = np.isfinite(conf)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(conf[valid]))


def compute_temporal_metrics(seq: List[Dict[int, np.ndarray]]) -> Dict[str, float]:
    """Compute temporal smoothness metrics: speed and jerk."""
    x = sequence_to_array(seq)  # (T,J,3)
    if x.shape[0] < 3:
        return {"speed_mean": np.nan, "jerk_mean": np.nan}

    v = x[1:] - x[:-1]
    ok_v = np.isfinite(v).all(axis=2)
    speed = np.linalg.norm(np.where(np.isfinite(v), v, 0.0), axis=2)
    speed_vals = speed[ok_v]

    a = v[1:] - v[:-1]
    ok_a = np.isfinite(a).all(axis=2)
    jerk = np.linalg.norm(np.where(np.isfinite(a), a, 0.0), axis=2)
    jerk_vals = jerk[ok_a]

    return {
        "speed_mean": float(np.mean(speed_vals)) if speed_vals.size else float("nan"),
        "jerk_mean": float(np.mean(jerk_vals)) if jerk_vals.size else float("nan"),
    }


def compute_bone_length_cv(seq: List[Dict[int, np.ndarray]]) -> float:
    """Compute coefficient of variation for bone lengths across frames."""
    x = sequence_to_array(seq)
    id_to_idx = {jid: i for i, jid in enumerate(TARGET_IDS)}
    cvs = []

    for a, b in BONE_EDGES:
        if a not in id_to_idx or b not in id_to_idx:
            continue
        ia, ib = id_to_idx[a], id_to_idx[b]
        xa = x[:, ia, :]
        xb = x[:, ib, :]
        ok = np.isfinite(xa).all(axis=1) & np.isfinite(xb).all(axis=1)
        if not np.any(ok):
            continue
        lengths = np.linalg.norm(xa[ok] - xb[ok], axis=1)
        m = float(np.mean(lengths))
        s = float(np.std(lengths))
        if m > 1e-9:
            cvs.append(s / m)

    return float(np.mean(cvs)) if cvs else float("nan")


def safe_pct_improvement(baseline: float, target: float) -> float:
    """Calculate percentage improvement from baseline to target.
    
    Args:
        baseline: Original value before improvement
        target: New value after improvement
        
    Returns:
        Percentage improvement (positive = improvement, negative = degradation)
    """
    if not np.isfinite(baseline) or baseline == 0 or not np.isfinite(target):
        return float("nan")
    return (baseline - target) / baseline * 100.0


def evaluate_person(
    person_name: str,
    person_paths: Dict[str, Path],
    fused_root: Path,
    sigma_px: float,
    sigma_3d: float,
):
    """Evaluate fusion quality for one person without ground truth.
    
    Compares raw fused results vs smoothed results using:
    - Reprojection confidence (2D-3D consistency)
    - Cross-view consistency (left-right 3D agreement)
    - Temporal metrics (speed/jerk)
    - Bone length stability (coefficient of variation)
    
    Returns:
        Dict with quality metrics, or None if files not found.
    """
    raw_path = fused_root / f"{person_name}_fused.npy"
    smooth_path = fused_root / f"{person_name}_smoothed.npy"
    if not raw_path.exists() or not smooth_path.exists():
        return None

    l2d, r2d, l3d, r3d = load_lr_sequences(person_paths)
    raw_seq = load_fused_sequence(raw_path)
    sm_seq = load_fused_sequence(smooth_path)
    n = min(len(l2d), len(r2d), len(l3d), len(r3d), len(raw_seq), len(sm_seq))
    l2d, r2d, l3d, r3d, raw_seq, sm_seq = (
        l2d[:n],
        r2d[:n],
        l3d[:n],
        r3d[:n],
        raw_seq[:n],
        sm_seq[:n],
    )

    conf_lr_l = []
    conf_lr_r = []
    conf_cross = []
    conf_raw_l = []
    conf_raw_r = []
    conf_sm_l = []
    conf_sm_r = []

    for i in range(n):
        c_l, _, _, _ = weakpersp_reproj_confidence(l3d[i], l2d[i], sigma_px=sigma_px)
        c_r, _, _, _ = weakpersp_reproj_confidence(r3d[i], r2d[i], sigma_px=sigma_px)
        c_x, _, _, _, _ = crossview_consistency_confidence(
            l3d[i],
            r3d[i],
            root_idx=IDX_PELVIS,
            left_hip_idx=IDX_LHIP,
            right_hip_idx=IDX_RHIP,
            left_shoulder_idx=IDX_LSHO,
            right_shoulder_idx=IDX_RSHO,
            sigma_3d=sigma_3d,
            scale_mode="hip",
        )
        c_raw_l, _, _, _ = weakpersp_reproj_confidence(
            raw_seq[i], l2d[i], sigma_px=sigma_px
        )
        c_raw_r, _, _, _ = weakpersp_reproj_confidence(
            raw_seq[i], r2d[i], sigma_px=sigma_px
        )
        c_sm_l, _, _, _ = weakpersp_reproj_confidence(
            sm_seq[i], l2d[i], sigma_px=sigma_px
        )
        c_sm_r, _, _, _ = weakpersp_reproj_confidence(
            sm_seq[i], r2d[i], sigma_px=sigma_px
        )

        conf_lr_l.append(mean_confidence(c_l))
        conf_lr_r.append(mean_confidence(c_r))
        conf_cross.append(mean_confidence(c_x))
        conf_raw_l.append(mean_confidence(c_raw_l))
        conf_raw_r.append(mean_confidence(c_raw_r))
        conf_sm_l.append(mean_confidence(c_sm_l))
        conf_sm_r.append(mean_confidence(c_sm_r))

    raw_temporal = compute_temporal_metrics(raw_seq)
    sm_temporal = compute_temporal_metrics(sm_seq)
    raw_bone_cv = compute_bone_length_cv(raw_seq)
    sm_bone_cv = compute_bone_length_cv(sm_seq)

    return {
        "person": person_name,
        "frames": n,
        "left_reproj_conf": float(np.nanmean(conf_lr_l)),
        "right_reproj_conf": float(np.nanmean(conf_lr_r)),
        "crossview_conf": float(np.nanmean(conf_cross)),
        "raw_reproj_conf": float(
            np.nanmean([(a + b) * 0.5 for a, b in zip(conf_raw_l, conf_raw_r)])
        ),
        "smooth_reproj_conf": float(
            np.nanmean([(a + b) * 0.5 for a, b in zip(conf_sm_l, conf_sm_r)])
        ),
        "raw_speed": raw_temporal["speed_mean"],
        "smooth_speed": sm_temporal["speed_mean"],
        "raw_jerk": raw_temporal["jerk_mean"],
        "smooth_jerk": sm_temporal["jerk_mean"],
        "raw_bone_cv": raw_bone_cv,
        "smooth_bone_cv": sm_bone_cv,
        "reproj_gain_pct": safe_pct_improvement(
            1.0
            - float(
                np.nanmean([(a + b) * 0.5 for a, b in zip(conf_raw_l, conf_raw_r)])
            ),
            1.0
            - float(np.nanmean([(a + b) * 0.5 for a, b in zip(conf_sm_l, conf_sm_r)])),
        ),
        "jerk_gain_pct": safe_pct_improvement(
            raw_temporal["jerk_mean"], sm_temporal["jerk_mean"]
        ),
        "speed_gain_pct": safe_pct_improvement(
            raw_temporal["speed_mean"], sm_temporal["speed_mean"]
        ),
        "bone_cv_gain_pct": safe_pct_improvement(raw_bone_cv, sm_bone_cv),
    }


def format_value(v: float, ndigits: int = 6) -> str:
    """Format float value with specified digits, or 'nan' if non-finite."""
    return f"{v:.{ndigits}f}" if np.isfinite(v) else "nan"


def build_report(results: List[Dict[str, float]]) -> str:
    """Build formatted text report from evaluation results.
    
    Args:
        results: List of per-person evaluation result dicts.
        
    Returns:
        Formatted multi-line report string.
    """
    lines = []
    lines.append("=" * 92)
    lines.append("True Data Fusion Quality Report (No GT)")
    lines.append("=" * 92)
    lines.append("Metrics:")
    lines.append("  - reprojection confidence: higher is better")
    lines.append("  - cross-view confidence: higher is better")
    lines.append("  - speed/jerk/bone_cv: lower is better")
    lines.append("  - gain% = (raw_fused - smoothed) / raw_fused * 100")
    lines.append("")

    header = (
        f"{'person':<10} {'frames':>6} {'cross_conf':>11} "
        f"{'raw_conf':>10} {'sm_conf':>10} {'conf_gain%':>10} "
        f"{'raw_speed':>10} {'sm_speed':>10} {'speed_gain%':>11} "
        f"{'raw_jerk':>10} {'sm_jerk':>10} {'jerk_gain%':>10} "
        f"{'raw_bcv':>9} {'sm_bcv':>9} {'bcv_gain%':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        lines.append(
            f"{r['person']:<10} {int(r['frames']):>6} {format_value(r['crossview_conf'], 4):>11}"
            f" {format_value(r['raw_reproj_conf'], 4):>10} {format_value(r['smooth_reproj_conf'], 4):>10} {format_value(r['reproj_gain_pct'], 3):>10}"
            f" {format_value(r['raw_speed'], 4):>10} {format_value(r['smooth_speed'], 4):>10} {format_value(r['speed_gain_pct'], 3):>11}"
            f" {format_value(r['raw_jerk'], 4):>10} {format_value(r['smooth_jerk'], 4):>10} {format_value(r['jerk_gain_pct'], 3):>10}"
            f" {format_value(r['raw_bone_cv'], 4):>9} {format_value(r['smooth_bone_cv'], 4):>9} {format_value(r['bone_cv_gain_pct'], 3):>10}"
        )

    if results:
        lines.append("")
        lines.append("[Aggregate Mean]")
        keys = [
            "crossview_conf",
            "raw_reproj_conf",
            "smooth_reproj_conf",
            "reproj_gain_pct",
            "raw_speed",
            "smooth_speed",
            "speed_gain_pct",
            "raw_jerk",
            "smooth_jerk",
            "raw_bone_cv",
            "smooth_bone_cv",
            "jerk_gain_pct",
            "bone_cv_gain_pct",
        ]
        for k in keys:
            vals = np.array([r[k] for r in results], dtype=np.float64)
            lines.append(f"  {k:<18}: {format_value(float(np.nanmean(vals)), 6)}")

        lines.append("")
        lines.append("[Smoothed Better Count]")
        def _better_count(key: str) -> int:
            return int(np.sum(np.array([r[key] for r in results], dtype=np.float64) > 0))
        total = len(results)
        lines.append(f"  reproj_gain_pct  > 0 : {_better_count('reproj_gain_pct')}/{total}")
        lines.append(f"  speed_gain_pct   > 0 : {_better_count('speed_gain_pct')}/{total}")
        lines.append(f"  jerk_gain_pct    > 0 : {_better_count('jerk_gain_pct')}/{total}")
        lines.append(f"  bone_cv_gain_pct > 0 : {_better_count('bone_cv_gain_pct')}/{total}")

    return "\n".join(lines)


def main() -> None:
    """Main entry point: evaluate all persons and generate report."""
    args = parse_args()

    all_results = []
    for person_dir in sorted(args.person_root.iterdir()):
        if not person_dir.is_dir():
            continue
        paths = resolve_person_paths(person_dir)
        if paths is None:
            continue

        result = evaluate_person(
            person_name=person_dir.name,
            person_paths=paths,
            fused_root=args.fused_root,
            sigma_px=args.sigma_px,
            sigma_3d=args.sigma_3d,
        )
        if result is not None:
            all_results.append(result)

    report = build_report(all_results)
    print(report)

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(report, encoding="utf-8")
    print(f"\n[Saved] report: {args.report_path}")


if __name__ == "__main__":
    main()
