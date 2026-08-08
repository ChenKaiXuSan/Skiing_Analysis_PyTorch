#!/usr/bin/env python3
"""Reviewer 2 Unity fusion baselines, ablations, and paired statistics."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from fuse.confidence import crossview_consistency_confidence, weakpersp_reproj_confidence

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

IDX_PELVIS = 14
IDX_LHIP = 11
IDX_RHIP = 12
IDX_LSHO = 5
IDX_RSHO = 6
EPS = 1e-12

DEFAULT_DATA_ROOT = Path("/home/kaixu_chen/skiing/data/dual_view_pose")
DEFAULT_SAM_ROOT = Path("/home/kaixu_chen/skiing/data/sam3d_body_results/unity/male")
DEFAULT_FUSED_ROOT = DEFAULT_DATA_ROOT / "fused_smoothed_results_bak/unity_pairs/male"


@dataclass
class MethodResult:
    name: str
    frame_mpjpe: np.ndarray
    per_joint_error: np.ndarray

    @property
    def mean(self) -> float:
        return float(np.nanmean(self.frame_mpjpe))

    @property
    def std(self) -> float:
        return float(np.nanstd(self.frame_mpjpe))

    @property
    def median(self) -> float:
        return float(np.nanmedian(self.frame_mpjpe))


def softmax_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m = np.maximum(a, b)
    ea = np.exp(a - m)
    eb = np.exp(b - m)
    denom = ea + eb + EPS
    return ea / denom, eb / denom


def weighted_fusion_array(left: np.ndarray, right: np.ndarray, q_left: np.ndarray, q_right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    q_left = np.asarray(q_left, dtype=np.float64)
    q_right = np.asarray(q_right, dtype=np.float64)
    wl, wr = softmax_pair(q_left, q_right)
    fused = np.full_like(left, np.nan)
    ok_l = np.isfinite(left).all(axis=2)
    ok_r = np.isfinite(right).all(axis=2)
    both = ok_l & ok_r
    fused[both] = wl[both, None] * left[both] + wr[both, None] * right[both]
    fused[ok_l & ~ok_r] = left[ok_l & ~ok_r]
    fused[~ok_l & ok_r] = right[~ok_l & ok_r]
    return fused


def sign_flip_p_value(differences: Iterable[float], max_exact_n: int = 20) -> float:
    diffs = np.asarray(list(differences), dtype=np.float64)
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[np.abs(diffs) > EPS]
    n = diffs.size
    if n == 0:
        return 1.0
    observed = abs(float(np.mean(diffs)))
    if n <= max_exact_n:
        total = 1 << n
        count = 0
        for mask in range(total):
            signs = np.ones(n, dtype=np.float64)
            for bit in range(n):
                if mask & (1 << bit):
                    signs[bit] = -1.0
            if abs(float(np.mean(signs * diffs))) >= observed - EPS:
                count += 1
        return float(count / total)
    rng = np.random.default_rng(0)
    n_perm = 20000
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n))
    perm = np.abs(np.mean(signs * diffs[None, :], axis=1))
    return float((np.count_nonzero(perm >= observed - EPS) + 1) / (n_perm + 1))


def bootstrap_ci(values: Iterable[float], *, confidence: float = 0.95, seed: int = 0, n_boot: int = 5000) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = np.mean(arr[idx], axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def paired_effect_size_dz(before: Iterable[float], after: Iterable[float]) -> float:
    before = np.asarray(list(before), dtype=np.float64)
    after = np.asarray(list(after), dtype=np.float64)
    diff = before - after
    diff = diff[np.isfinite(diff)]
    if diff.size < 2:
        return float("nan")
    sd = float(np.std(diff, ddof=1))
    if sd < EPS:
        return float("inf") if float(np.mean(diff)) > 0 else 0.0
    return float(np.mean(diff) / sd)


def holm_adjust(p_values: Iterable[float]) -> List[float]:
    p_values = [float(p) for p in p_values]
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p_values[idx])
        running = max(running, val)
        adjusted[idx] = running
    return adjusted


def _jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f]


def _gt_3d(item: dict) -> np.ndarray:
    name_to_idx = {v: i for i, v in enumerate(UNITY_MHR70_MAPPING.values())}
    out = np.full((len(TARGET_IDS), 3), np.nan, dtype=np.float64)
    for joint in item.get("joints3d", []):
        idx = name_to_idx.get(joint["name"])
        if idx is not None:
            out[idx] = [-float(joint["z"]), -float(joint["y"]), float(joint["x"])]
    return out


def _sam_arrays(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    frames = data[data.files[0]]
    p2d = []
    p3d = []
    for frame in frames:
        p2d.append(np.asarray(frame["pred_keypoints_2d"], dtype=np.float64)[TARGET_IDS])
        p3d.append(np.asarray(frame["pred_keypoints_3d"], dtype=np.float64)[: len(TARGET_IDS)])
    return np.asarray(p2d, dtype=np.float64), np.asarray(p3d, dtype=np.float64)


def _array_to_dict(arr: np.ndarray) -> Dict[int, np.ndarray]:
    return {jid: np.asarray(arr[i], dtype=np.float64) for i, jid in enumerate(TARGET_IDS)}


def _confidence_arrays(left_2d: np.ndarray, right_2d: np.ndarray, left_3d: np.ndarray, right_3d: np.ndarray, sigma_px: float, sigma_3d: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = left_3d.shape[0]
    conf_l = np.zeros((n, len(TARGET_IDS)), dtype=np.float64)
    conf_r = np.zeros_like(conf_l)
    conf_cross = np.zeros_like(conf_l)
    for t in range(n):
        p3d_l = _array_to_dict(left_3d[t])
        p3d_r = _array_to_dict(right_3d[t])
        p2d_l = _array_to_dict(left_2d[t])
        p2d_r = _array_to_dict(right_2d[t])
        conf_l[t], _, _, _ = weakpersp_reproj_confidence(p3d_l, p2d_l, sigma_px=sigma_px)
        conf_r[t], _, _, _ = weakpersp_reproj_confidence(p3d_r, p2d_r, sigma_px=sigma_px)
        conf_cross[t], _, _, _, _ = crossview_consistency_confidence(
            p3d_l,
            p3d_r,
            root_idx=IDX_PELVIS,
            left_hip_idx=IDX_LHIP,
            right_hip_idx=IDX_RHIP,
            left_shoulder_idx=IDX_LSHO,
            right_shoulder_idx=IDX_RSHO,
            sigma_3d=sigma_3d,
            scale_mode="hip",
        )
    return conf_l, conf_r, conf_cross


def _mpjpe(pred: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    err = np.linalg.norm(pred - gt, axis=2)
    err[~(np.isfinite(pred).all(axis=2) & np.isfinite(gt).all(axis=2))] = np.nan
    return np.nanmean(err, axis=1), err


def _load_npy_sequence(path: Path, n: int) -> np.ndarray:
    return np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)[:n]


def _method(name: str, pred: np.ndarray, gt: np.ndarray) -> MethodResult:
    frame, per_joint = _mpjpe(pred, gt)
    return MethodResult(name=name, frame_mpjpe=frame, per_joint_error=per_joint)


def load_unity_data(args: argparse.Namespace) -> Tuple[np.ndarray, ...]:
    left_2d, left_3d = _sam_arrays(args.left_sam)
    right_2d, right_3d = _sam_arrays(args.right_sam)
    gt_items = _jsonl(args.gt_3d)
    gt_3d = np.asarray([_gt_3d(item) for item in gt_items], dtype=np.float64)
    n = min(left_3d.shape[0], right_3d.shape[0], gt_3d.shape[0])
    return left_2d[:n], right_2d[:n], left_3d[:n], right_3d[:n], gt_3d[:n]


def build_methods(left_2d: np.ndarray, right_2d: np.ndarray, left_3d: np.ndarray, right_3d: np.ndarray, gt_3d: np.ndarray, sigma_px: float, sigma_3d: float, raw_fused: np.ndarray | None = None, smoothed_fused: np.ndarray | None = None) -> List[MethodResult]:
    conf_l, conf_r, conf_cross = _confidence_arrays(left_2d, right_2d, left_3d, right_3d, sigma_px, sigma_3d)
    q_mul_l = np.sqrt(np.clip(conf_l * conf_cross, 0.0, 1.0))
    q_mul_r = np.sqrt(np.clip(conf_r * conf_cross, 0.0, 1.0))
    q_add_l = 0.5 * (conf_l + conf_cross)
    q_add_r = 0.5 * (conf_r + conf_cross)
    left_frame, _ = _mpjpe(left_3d, gt_3d)
    right_frame, _ = _mpjpe(right_3d, gt_3d)
    best_single = np.where(left_frame[:, None, None] <= right_frame[:, None, None], left_3d, right_3d)
    best_view = np.where((q_mul_l >= q_mul_r)[:, :, None], left_3d, right_3d)
    average = 0.5 * (left_3d + right_3d)
    median = np.median(np.stack([left_3d, right_3d], axis=0), axis=0)
    methods = [
        _method("Left single-view", left_3d, gt_3d),
        _method("Right single-view", right_3d, gt_3d),
        _method("Best single-view (oracle frame)", best_single, gt_3d),
        _method("Unweighted average fusion", average, gt_3d),
        _method("Two-view median fusion", median, gt_3d),
        _method("Confidence best-view selection", best_view, gt_3d),
        _method("2D confidence only", weighted_fusion_array(left_3d, right_3d, conf_l, conf_r), gt_3d),
        _method("3D consistency only", weighted_fusion_array(left_3d, right_3d, conf_cross, conf_cross), gt_3d),
        _method("2D x 3D multiplicative (recomputed)", weighted_fusion_array(left_3d, right_3d, q_mul_l, q_mul_r), gt_3d),
        _method("2D + 3D additive (recomputed)", weighted_fusion_array(left_3d, right_3d, q_add_l, q_add_r), gt_3d),
    ]
    if raw_fused is not None:
        methods.append(_method("Proposed confidence-guided fusion (saved raw)", raw_fused, gt_3d))
    if smoothed_fused is not None:
        methods.append(_method("Proposed fusion + smoothing (saved)", smoothed_fused, gt_3d))
    return methods


def add_sigma_sensitivity(methods: List[MethodResult], left_2d: np.ndarray, right_2d: np.ndarray, left_3d: np.ndarray, right_3d: np.ndarray, gt_3d: np.ndarray, sigma_px: float, sigma_3d: float) -> None:
    for factor in (0.5, 2.0):
        conf_l, conf_r, conf_cross = _confidence_arrays(left_2d, right_2d, left_3d, right_3d, sigma_px * factor, sigma_3d * factor)
        q_l = np.sqrt(np.clip(conf_l * conf_cross, 0.0, 1.0))
        q_r = np.sqrt(np.clip(conf_r * conf_cross, 0.0, 1.0))
        pred = weighted_fusion_array(left_3d, right_3d, q_l, q_r)
        methods.append(_method(f"2D x 3D multiplicative ({factor:.1f}x sigma, recomputed)", pred, gt_3d))


def comparison_rows(methods: List[MethodResult], reference_name: str) -> List[dict]:
    lookup = {m.name: m for m in methods}
    ref = lookup[reference_name]
    rows = []
    raw_p = []
    for method in methods:
        if method.name == reference_name:
            continue
        diff = method.frame_mpjpe - ref.frame_mpjpe
        diff = diff[np.isfinite(diff)]
        p = sign_flip_p_value(diff)
        raw_p.append(p)
        ci_lo, ci_hi = bootstrap_ci(diff)
        rows.append({
            "comparison": f"{method.name} - {reference_name}",
            "mean_delta": float(np.mean(diff)),
            "median_delta": float(np.median(diff)),
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "p_value": p,
            "cohen_dz": paired_effect_size_dz(method.frame_mpjpe, ref.frame_mpjpe),
        })
    for row, p_adj in zip(rows, holm_adjust(raw_p)):
        row["holm_p"] = p_adj
    return rows


def write_reports(methods: List[MethodResult], comparisons: List[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "fusion_ablation_summary.csv"
    stats_csv = out_dir / "fusion_ablation_statistics.csv"
    report_md = out_dir / "fusion_ablation_report.md"
    best_single = min(m.mean for m in methods if m.name in {"Left single-view", "Right single-view"})
    unweighted = next(m.mean for m in methods if m.name == "Unweighted average fusion")
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "mean_mpjpe", "std_mpjpe", "median_mpjpe", "improvement_vs_best_single_pct", "improvement_vs_unweighted_pct"])
        writer.writeheader()
        for m in methods:
            writer.writerow({
                "method": m.name,
                "mean_mpjpe": f"{m.mean:.6f}",
                "std_mpjpe": f"{m.std:.6f}",
                "median_mpjpe": f"{m.median:.6f}",
                "improvement_vs_best_single_pct": f"{(best_single - m.mean) / best_single * 100.0:.3f}",
                "improvement_vs_unweighted_pct": f"{(unweighted - m.mean) / unweighted * 100.0:.3f}",
            })
    with stats_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["comparison", "mean_delta", "median_delta", "ci_low", "ci_high", "p_value", "holm_p", "cohen_dz"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in comparisons:
            writer.writerow({k: row[k] if k == "comparison" else f"{row[k]:.8g}" for k in fieldnames})
    lines = [
        "# Reviewer 2 Unity Fusion Baseline and Ablation Report",
        "",
        f"n_frames = {len(methods[0].frame_mpjpe)}",
        "3D MPJPE is reported in Unity/avatar coordinate units using the same 15-joint matched evaluation convention as the manuscript. Saved proposed-fusion rows are loaded from the manuscript output files; rows marked recomputed are direct internal baseline/ablation replays.",
        "",
        "## Method Summary",
        "",
        "| Method | Mean MPJPE | SD | Median | vs best single | vs unweighted |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m in methods:
        lines.append(f"| {m.name} | {m.mean:.6f} | {m.std:.6f} | {m.median:.6f} | {(best_single - m.mean) / best_single * 100.0:.3f}% | {(unweighted - m.mean) / unweighted * 100.0:.3f}% |")
    lines.extend(["", "## Paired Frame-Level Statistics", "", "Positive deltas mean the comparison method has higher MPJPE than the reference method. Saved proposed-fusion rows use the existing manuscript output files; rows marked recomputed are direct replay baselines from the same left/right predictions.", "", "| Comparison | Mean delta | 95% CI | p | Holm p | Cohen dz |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for row in comparisons:
        lines.append(f"| {row['comparison']} | {row['mean_delta']:.6f} | [{row['ci_low']:.6f}, {row['ci_high']:.6f}] | {row['p_value']:.6g} | {row['holm_p']:.6g} | {row['cohen_dz']:.3f} |")
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-sam", type=Path, default=DEFAULT_SAM_ROOT / "left_sam_3d_body_outputs.npz")
    parser.add_argument("--right-sam", type=Path, default=DEFAULT_SAM_ROOT / "right_sam_3d_body_outputs.npz")
    parser.add_argument("--gt-3d", type=Path, default=DEFAULT_DATA_ROOT / "unity_data/RecordingsPose/male_pose3d_trimmed.jsonl")
    parser.add_argument("--sigma-px", type=float, default=12.0)
    parser.add_argument("--sigma-3d", type=float, default=0.08)
    parser.add_argument("--raw-fused", type=Path, default=DEFAULT_FUSED_ROOT / "left__right_fused.npy")
    parser.add_argument("--smoothed-fused", type=Path, default=DEFAULT_FUSED_ROOT / "left__right_smoothed.npy")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/reviewer2_fusion_ablation"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    left_2d, right_2d, left_3d, right_3d, gt_3d = load_unity_data(args)
    raw_fused = _load_npy_sequence(args.raw_fused, gt_3d.shape[0]) if args.raw_fused.exists() else None
    smoothed_fused = _load_npy_sequence(args.smoothed_fused, gt_3d.shape[0]) if args.smoothed_fused.exists() else None
    methods = build_methods(left_2d, right_2d, left_3d, right_3d, gt_3d, args.sigma_px, args.sigma_3d, raw_fused, smoothed_fused)
    add_sigma_sensitivity(methods, left_2d, right_2d, left_3d, right_3d, gt_3d, args.sigma_px, args.sigma_3d)
    reference = "Proposed confidence-guided fusion (saved raw)" if any(m.name == "Proposed confidence-guided fusion (saved raw)" for m in methods) else "2D x 3D multiplicative"
    comparisons = comparison_rows(methods, reference)
    write_reports(methods, comparisons, args.output_dir)
    print(f"[saved] {args.output_dir / 'fusion_ablation_report.md'}")


if __name__ == "__main__":
    main()
