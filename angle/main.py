#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/angle/main.py
Project: /workspace/code/angle
Created Date: Wednesday February 11th 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Wednesday February 11th 2026 1:41:43 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

# TODO: 需要按照滑雪者面向前方的位置来划分滑雪的trun。
# TODO: 因为run和pro的长度不一样，所以没有办法直接进行比较。我需要先把每个人的动作划分成几个turn，然后在每个turn里比较融合前后的角度变化情况。

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# --- 設定 ---
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

ID_TO_INDEX = {jid: idx for idx, jid in enumerate(TARGET_IDS)}

ANGLE_DEFS = {
    "knee_l": (9, 11, 13),
    "knee_r": (10, 12, 14),
    "elbow_l": (5, 7, 62),
    "elbow_r": (6, 8, 41),
    "shoulder_l": (69, 5, 7),
    "shoulder_r": (69, 6, 8),
    "hip_l": (69, 9, 11),
    "hip_r": (69, 10, 12),
}

# Elbow joint IDs
ELBOW_IDS = {
    "elbow_l": 7,  # lowerarm_l
    "elbow_r": 8,  # lowerarm_r
}

# Skeleton connections (bone pairs) for visualization
# Each tuple is (parent_joint_id, child_joint_id)
SKELETON_CONNECTIONS = [
    # Left arm
    (69, 5),  # neck -> shoulder_l
    (5, 7),  # shoulder_l -> elbow_l
    (7, 62),  # elbow_l -> hand_l
    # Right arm
    (69, 6),  # neck -> shoulder_r
    (6, 8),  # shoulder_r -> elbow_r
    (8, 41),  # elbow_r -> hand_r
    # Spine
    (69, 9),  # neck -> hip_l
    (69, 10),  # neck -> hip_r
    # Left leg
    (9, 11),  # hip_l -> knee_l
    (11, 13),  # knee_l -> foot_l
    # Right leg
    (10, 12),  # hip_r -> knee_r
    (12, 14),  # knee_r -> foot_r
]


def _center_from_ids(
    frame: np.ndarray,
    ids: Tuple[int, int],
    id_to_index: Dict[int, int],
) -> np.ndarray:
    points = []
    for jid in ids:
        p = frame[id_to_index[jid]]
        if np.all(np.isfinite(p)):
            points.append(p)
    if not points:
        return np.full((3,), np.nan, dtype=np.float64)
    return np.mean(np.stack(points, axis=0), axis=0)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0.0 or not np.isfinite(n):
        return np.full_like(v, np.nan)
    return v / n


def _fill_nan_linear(x: np.ndarray) -> np.ndarray:
    """Linearly fill NaN values for a 1D series."""
    y = np.asarray(x, dtype=np.float64).copy()
    n = y.shape[0]
    if n == 0:
        return y
    idx = np.arange(n)
    valid = np.isfinite(y)
    if not np.any(valid):
        return y
    if np.sum(valid) == 1:
        y[~valid] = y[valid][0]
        return y
    y[~valid] = np.interp(idx[~valid], idx[valid], y[valid])
    return y


def _smooth_1d(x: np.ndarray, window: int = 11) -> np.ndarray:
    if window < 3:
        return x.copy()
    if window % 2 == 0:
        window += 1
    kernel = np.ones((window,), dtype=np.float64)
    valid = np.isfinite(x).astype(np.float64)
    filled = np.where(np.isfinite(x), x, 0.0)
    num = np.convolve(filled, kernel, mode="same")
    den = np.convolve(valid, kernel, mode="same")
    out = np.full_like(x, np.nan, dtype=np.float64)
    mask = den > 0.0
    out[mask] = num[mask] / den[mask]
    return out


def compute_facing_heading(
    kpts: np.ndarray,
    id_to_index: Dict[int, int],
    up_axis: np.ndarray,
) -> np.ndarray:
    """Compute per-frame heading angle (degrees) from skier facing direction.

    Heading is computed in XZ ground plane from the forward vector.
    """
    if kpts.ndim != 3 or kpts.shape[2] != 3:
        raise ValueError("kpts must be (T,J,3)")

    T = kpts.shape[0]
    heading = np.full((T,), np.nan, dtype=np.float64)
    up_unit = _unit(up_axis)
    if not np.all(np.isfinite(up_unit)):
        return heading

    for t in range(T):
        frame = kpts[t]
        hip_l = frame[id_to_index[9]]
        hip_r = frame[id_to_index[10]]
        sho_l = frame[id_to_index[5]]
        sho_r = frame[id_to_index[6]]

        lr = None
        if np.all(np.isfinite(hip_l)) and np.all(np.isfinite(hip_r)):
            lr = hip_r - hip_l
        elif np.all(np.isfinite(sho_l)) and np.all(np.isfinite(sho_r)):
            lr = sho_r - sho_l

        if lr is None:
            continue

        lr_unit = _unit(lr)
        if not np.all(np.isfinite(lr_unit)):
            continue

        if up_axis[1] < 0:
            forward = _unit(np.cross(up_unit, lr_unit))
        else:
            forward = _unit(np.cross(lr_unit, up_unit))

        if not np.all(np.isfinite(forward)):
            continue

        # Ground-plane heading angle for consistent cross-subject comparison.
        heading[t] = np.degrees(np.arctan2(forward[0], forward[2]))

    return heading


def detect_turn_segments(
    heading_deg: np.ndarray,
    min_turn_frames: int = 12,
    min_heading_change_deg: float = 8.0,
) -> List[Dict[str, float]]:
    """Split a sequence into turns based on facing-direction extrema."""
    T = heading_deg.shape[0]
    if T == 0:
        return []

    valid = np.isfinite(heading_deg)
    if np.sum(valid) < 5:
        return []

    heading_filled = _fill_nan_linear(heading_deg)
    heading_unwrapped = np.degrees(np.unwrap(np.radians(heading_filled)))
    heading_smooth = _smooth_1d(heading_unwrapped, window=11)

    vel = np.gradient(heading_smooth)
    vel_smooth = _smooth_1d(vel, window=9)

    extrema = []
    for i in range(1, T):
        prev_v = vel_smooth[i - 1]
        curr_v = vel_smooth[i]
        if not np.isfinite(prev_v) or not np.isfinite(curr_v):
            continue
        if prev_v == 0.0 and curr_v == 0.0:
            continue
        if prev_v * curr_v < 0.0:
            extrema.append(i)

    boundaries = [0]
    for idx in extrema:
        if idx - boundaries[-1] >= min_turn_frames:
            boundaries.append(idx)
    if T - 1 - boundaries[-1] >= 1:
        boundaries.append(T - 1)
    elif boundaries[-1] != T - 1:
        boundaries[-1] = T - 1

    if len(boundaries) < 2:
        return []

    turns: List[Dict[str, float]] = []
    turn_id = 1
    for s, e in zip(boundaries[:-1], boundaries[1:]):
        if e - s + 1 < min_turn_frames:
            continue
        delta = float(heading_smooth[e] - heading_smooth[s])
        if abs(delta) < min_heading_change_deg:
            continue
        turns.append(
            {
                "turn_id": float(turn_id),
                "start_frame": float(s),
                "end_frame": float(e),
                "num_frames": float(e - s + 1),
                "heading_change_deg": delta,
                "direction": 1.0 if delta > 0.0 else -1.0,
            }
        )
        turn_id += 1

    return turns


def save_turn_reports(
    output_dir: Path,
    turns: List[Dict[str, float]],
    heading_deg: np.ndarray,
    series_dict: Dict[str, np.ndarray],
) -> None:
    """Save per-turn summary and per-turn angle statistics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    turns_csv = output_dir / "turn_summary.csv"
    with turns_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "turn_id",
                "start_frame",
                "end_frame",
                "num_frames",
                "heading_change_deg",
                "direction",
            ]
        )
        for turn in turns:
            direction_name = "left" if turn["direction"] > 0 else "right"
            writer.writerow(
                [
                    int(turn["turn_id"]),
                    int(turn["start_frame"]),
                    int(turn["end_frame"]),
                    int(turn["num_frames"]),
                    turn["heading_change_deg"],
                    direction_name,
                ]
            )

    metrics_csv = output_dir / "turn_metrics.csv"
    metric_names = list(series_dict.keys())
    with metrics_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["turn_id", "metric", "mean", "std", "min", "max"]
        writer.writerow(header)

        for turn in turns:
            s = int(turn["start_frame"])
            e = int(turn["end_frame"])
            for name in metric_names:
                values = series_dict[name][s : e + 1]
                finite = values[np.isfinite(values)]
                if finite.size == 0:
                    nan_row = [
                        int(turn["turn_id"]),
                        name,
                        "nan",
                        "nan",
                        "nan",
                        "nan",
                    ]
                    writer.writerow(nan_row)
                else:
                    writer.writerow(
                        [
                            int(turn["turn_id"]),
                            name,
                            float(np.mean(finite)),
                            float(np.std(finite)),
                            float(np.min(finite)),
                            float(np.max(finite)),
                        ]
                    )

    # Save heading series with turn boundaries for easier visual verification.
    heading_csv = output_dir / "turn_heading.csv"
    with heading_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "heading_deg", "turn_boundary"])
        boundaries = set()
        for turn in turns:
            boundaries.add(int(turn["start_frame"]))
            boundaries.add(int(turn["end_frame"]))
        for i, h in enumerate(heading_deg):
            writer.writerow([i, h, 1 if i in boundaries else 0])

    heading_png = output_dir / "turn_heading.png"
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    heading_smooth = _smooth_1d(_fill_nan_linear(heading_deg), window=11)
    ax.plot(heading_smooth, label="heading", linewidth=1.8)
    for turn in turns:
        s = int(turn["start_frame"])
        e = int(turn["end_frame"])
        ax.axvline(s, color="green", alpha=0.25, linestyle="--")
        ax.axvline(e, color="orange", alpha=0.25, linestyle="--")
    ax.set_xlabel("frame")
    ax.set_ylabel("heading (deg)")
    ax.set_title("Facing Heading with Turn Boundaries")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(heading_png, dpi=150)
    plt.close(fig)

    save_turn_detail_files(output_dir, turns, heading_deg, series_dict)


def save_turn_detail_files(
    output_dir: Path,
    turns: List[Dict[str, float]],
    heading_deg: np.ndarray,
    series_dict: Dict[str, np.ndarray],
) -> None:
    """Save per-turn detailed files for all analysis metrics."""
    turn_root = output_dir / "turn_details"
    turn_root.mkdir(parents=True, exist_ok=True)

    def _slice_series(
        src: Dict[str, np.ndarray],
        keys: List[str],
        start: int,
        end: int,
    ) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for k in keys:
            if k in src:
                out[k] = src[k][start : end + 1]
        return out

    joint_keys = list(ANGLE_DEFS.keys())
    knee_keys = ["knee_l", "knee_r"]
    torso_keys = ["torso_knee_angle"]
    knee_diff_keys = ["knee_diff_lr"]
    elbow_keys = ["elbow_distance_l", "elbow_distance_r"]
    body_keys = ["tilt_upper", "tilt_lower"]
    change_keys = [
        k for k in series_dict.keys() if k.endswith("_d") or k.endswith("_abs_d")
    ]

    metric_names = list(series_dict.keys())
    for turn in turns:
        turn_id = int(turn["turn_id"])
        s = int(turn["start_frame"])
        e = int(turn["end_frame"])

        turn_dir = turn_root / f"turn_{turn_id}_{s}_{e}"
        turn_dir.mkdir(parents=True, exist_ok=True)

        # Per-frame detailed values inside one turn.
        detail_csv = turn_dir / "series.csv"
        with detail_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["local_frame", "global_frame", "heading_deg"] + metric_names
            )
            local_idx = 0
            for g in range(s, e + 1):
                row = [local_idx, g, heading_deg[g]]
                row += [series_dict[name][g] for name in metric_names]
                writer.writerow(row)
                local_idx += 1

        # Summary stats for this turn only.
        summary_csv = turn_dir / "summary.csv"
        with summary_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "turn_id",
                    "start_frame",
                    "end_frame",
                    "num_frames",
                    "metric",
                    "mean",
                    "std",
                    "min",
                    "max",
                ]
            )
            for name in metric_names:
                values = series_dict[name][s : e + 1]
                finite = values[np.isfinite(values)]
                if finite.size == 0:
                    writer.writerow(
                        [
                            turn_id,
                            s,
                            e,
                            e - s + 1,
                            name,
                            "nan",
                            "nan",
                            "nan",
                            "nan",
                        ]
                    )
                else:
                    writer.writerow(
                        [
                            turn_id,
                            s,
                            e,
                            e - s + 1,
                            name,
                            float(np.mean(finite)),
                            float(np.std(finite)),
                            float(np.min(finite)),
                            float(np.max(finite)),
                        ]
                    )

        # Save the same report types as full-frame outputs, but for this turn only.
        joint_turn = _slice_series(series_dict, joint_keys, s, e)
        knee_turn = _slice_series(series_dict, knee_keys, s, e)
        torso_turn = _slice_series(series_dict, torso_keys, s, e)
        knee_diff_turn = _slice_series(series_dict, knee_diff_keys, s, e)
        elbow_turn = _slice_series(series_dict, elbow_keys, s, e)
        body_turn = _slice_series(series_dict, body_keys, s, e)
        change_turn = _slice_series(series_dict, change_keys, s, e)

        if joint_turn:
            save_angles_csv(turn_dir / "angles_joint.csv", joint_turn)
            plot_angles(turn_dir / "angles_joint.png", joint_turn)
        if knee_turn:
            save_angles_csv(turn_dir / "angles_knee.csv", knee_turn)
            plot_angles(turn_dir / "angles_knee.png", knee_turn)
        if torso_turn:
            save_angles_csv(turn_dir / "angles_torso_knee.csv", torso_turn)
            plot_angles(turn_dir / "angles_torso_knee.png", torso_turn)
        if knee_diff_turn:
            save_angles_csv(turn_dir / "angles_knee_diff.csv", knee_diff_turn)
            plot_angles(turn_dir / "angles_knee_diff.png", knee_diff_turn)
        if elbow_turn:
            save_angles_csv(turn_dir / "distance_elbow_midline.csv", elbow_turn)
            plot_angles(turn_dir / "distance_elbow_midline.png", elbow_turn)
        if body_turn:
            save_angles_csv(turn_dir / "angles_body_y_down.csv", body_turn)
            plot_angles(turn_dir / "angles_body_y_down.png", body_turn)
        if change_turn:
            save_angles_csv(turn_dir / "angles_change_fullframe.csv", change_turn)
            plot_angles(turn_dir / "angles_change_fullframe.png", change_turn)


def _series_turn_mean(
    series: np.ndarray,
    start_frame: int,
    end_frame: int,
) -> float:
    values = series[start_frame : end_frame + 1]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def compute_series_changes(
    series_dict: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Compute frame-to-frame changes for each metric series."""
    changes: Dict[str, np.ndarray] = {}
    for name, series in series_dict.items():
        if series.ndim != 1:
            continue
        delta = np.full_like(series, np.nan, dtype=np.float64)
        abs_delta = np.full_like(series, np.nan, dtype=np.float64)
        for i in range(1, series.shape[0]):
            prev = series[i - 1]
            curr = series[i]
            if np.isfinite(prev) and np.isfinite(curr):
                d = float(curr - prev)
                delta[i] = d
                abs_delta[i] = abs(d)
        changes[f"{name}_d"] = delta
        changes[f"{name}_abs_d"] = abs_delta
    return changes


def save_fullframe_change_reports(
    output_dir: Path,
    series_dict: Dict[str, np.ndarray],
    stem: str = "angles_change_fullframe",
) -> Dict[str, np.ndarray]:
    """Save full-frame per-metric changes before turn splitting."""
    change_series = compute_series_changes(series_dict)
    csv_path = output_dir / f"{stem}.csv"
    png_path = output_dir / f"{stem}.png"
    save_angles_csv(csv_path, change_series)
    plot_angles(png_path, change_series)
    print(f"Full-frame change csv saved to: {csv_path}")
    print(f"Full-frame change plot saved to: {png_path}")
    return change_series


def save_turn_comparison_report(
    out_csv: Path,
    before_turns: List[Dict[str, float]],
    after_turns: List[Dict[str, float]],
    before_series: Dict[str, np.ndarray],
    after_series: Dict[str, np.ndarray],
) -> None:
    """Compare fused before/after values by turn order."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    metrics = sorted(set(before_series.keys()) & set(after_series.keys()))
    num_pairs = min(len(before_turns), len(after_turns))

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "turn_pair_index",
                "before_turn_id",
                "after_turn_id",
                "metric",
                "before_mean",
                "after_mean",
                "delta_after_minus_before",
            ]
        )

        for i in range(num_pairs):
            t_before = before_turns[i]
            t_after = after_turns[i]

            s_b = int(t_before["start_frame"])
            e_b = int(t_before["end_frame"])
            s_a = int(t_after["start_frame"])
            e_a = int(t_after["end_frame"])

            for metric in metrics:
                mean_before = _series_turn_mean(before_series[metric], s_b, e_b)
                mean_after = _series_turn_mean(after_series[metric], s_a, e_a)
                delta = mean_after - mean_before
                writer.writerow(
                    [
                        i + 1,
                        int(t_before["turn_id"]),
                        int(t_after["turn_id"]),
                        metric,
                        mean_before,
                        mean_after,
                        delta,
                    ]
                )


def _compute_all_series(
    kpts: np.ndarray,
    up_axis: np.ndarray,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    np.ndarray,
    List[Dict[str, float]],
]:
    joint_angles = compute_angles(kpts, ANGLE_DEFS, ID_TO_INDEX)
    body_angles = compute_tilt_angles(kpts, ID_TO_INDEX, up_axis)
    torso_knee = compute_torso_knee_angle(kpts, ID_TO_INDEX)
    knee_diff = compute_knee_difference(kpts, ID_TO_INDEX)
    elbow_dist = compute_elbow_distance_from_midline(kpts, ID_TO_INDEX)
    heading_deg = compute_facing_heading(kpts, ID_TO_INDEX, up_axis)
    turns = detect_turn_segments(heading_deg)
    return (
        joint_angles,
        body_angles,
        torso_knee,
        knee_diff,
        elbow_dist,
        heading_deg,
        turns,
    )


def compute_tilt_angles(
    kpts: np.ndarray,
    id_to_index: Dict[int, int],
    up_axis: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute signed upper/lower body tilt angles (forward +, backward -).

    Args:
        kpts: (T,J,3) keypoints array
        id_to_index: joint ID to index mapping
        up_axis: vertical axis direction (e.g., [0,1,0] for Y-up, [0,0,1] for Z-up)
    """
    if kpts.ndim != 3 or kpts.shape[2] != 3:
        raise ValueError("kpts must be (T,J,3)")

    T = kpts.shape[0]
    upper = np.full((T,), np.nan, dtype=np.float64)
    lower = np.full((T,), np.nan, dtype=np.float64)

    for t in range(T):
        frame = kpts[t]

        hip_l = frame[id_to_index[9]]
        hip_r = frame[id_to_index[10]]
        sho_l = frame[id_to_index[5]]
        sho_r = frame[id_to_index[6]]

        pelvis = _center_from_ids(frame, (9, 10), id_to_index)
        shoulder = _center_from_ids(frame, (5, 6), id_to_index)
        knee = _center_from_ids(frame, (11, 12), id_to_index)

        lr = None
        if np.all(np.isfinite(hip_l)) and np.all(np.isfinite(hip_r)):
            lr = hip_r - hip_l
        elif np.all(np.isfinite(sho_l)) and np.all(np.isfinite(sho_r)):
            lr = sho_r - sho_l

        if lr is None:
            continue

        lr_unit = _unit(lr)
        up_unit = _unit(up_axis)
        if not np.all(np.isfinite(lr_unit)) or not np.all(np.isfinite(up_unit)):
            continue

        # Compute forward direction based on up_axis direction
        # If up_axis points down (Y < 0), reverse cross product order
        if up_axis[1] < 0:  # Y-axis down
            forward = _unit(np.cross(up_unit, lr_unit))
        else:  # Y-axis up
            forward = _unit(np.cross(lr_unit, up_unit))

        if not np.all(np.isfinite(forward)):
            continue

        def _tilt(v: np.ndarray) -> float:
            if not np.all(np.isfinite(v)):
                return float("nan")
            v_proj = v - np.dot(v, lr_unit) * lr_unit
            if not np.all(np.isfinite(v_proj)):
                return float("nan")
            v_unit = _unit(v_proj)
            if not np.all(np.isfinite(v_unit)):
                return float("nan")
            cos_theta = np.clip(np.dot(v_unit, up_unit), -1.0, 1.0)
            theta = np.degrees(np.arccos(cos_theta))
            sign = 1.0 if np.dot(v_unit, forward) >= 0.0 else -1.0
            return float(theta * sign)

        upper[t] = _tilt(shoulder - pelvis)
        lower[t] = _tilt(knee - pelvis)

    return {"tilt_upper": upper, "tilt_lower": lower}


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return angle ABC in degrees."""
    ba = a - b
    bc = c - b
    na = np.linalg.norm(ba)
    nc = np.linalg.norm(bc)
    if na == 0.0 or nc == 0.0:
        return float("nan")
    cos_theta = np.dot(ba, bc) / (na * nc)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def compute_angles(
    kpts: np.ndarray,
    angle_defs: Dict[str, Tuple[int, int, int]],
    id_to_index: Dict[int, int],
) -> Dict[str, np.ndarray]:
    """Compute angle time-series from (T,J,3) keypoints array."""
    if kpts.ndim != 3 or kpts.shape[2] != 3:
        raise ValueError("kpts must be (T,J,3)")

    angles: Dict[str, np.ndarray] = {}
    T = kpts.shape[0]

    for name, (a_id, b_id, c_id) in angle_defs.items():
        ai = id_to_index[a_id]
        bi = id_to_index[b_id]
        ci = id_to_index[c_id]

        series = np.full((T,), np.nan, dtype=np.float64)
        for t in range(T):
            a = kpts[t, ai]
            b = kpts[t, bi]
            c = kpts[t, ci]
            if (
                np.all(np.isfinite(a))
                and np.all(np.isfinite(b))
                and np.all(np.isfinite(c))
            ):
                series[t] = angle_deg(a, b, c)
        angles[name] = series

    return angles


def compute_torso_knee_angle(
    kpts: np.ndarray,
    id_to_index: Dict[int, int],
) -> Dict[str, np.ndarray]:
    """Compute angle between shoulder-pelvis and knee-pelvis lines.

    Args:
        kpts: (T,J,3) keypoints array
        id_to_index: joint ID to index mapping

    Returns:
        dict: angle time-series
    """
    if kpts.ndim != 3 or kpts.shape[2] != 3:
        raise ValueError("kpts must be (T,J,3)")

    T = kpts.shape[0]
    angle_series = np.full((T,), np.nan, dtype=np.float64)

    for t in range(T):
        frame = kpts[t]

        pelvis = _center_from_ids(frame, (9, 10), id_to_index)
        shoulder = _center_from_ids(frame, (5, 6), id_to_index)
        knee = _center_from_ids(frame, (11, 12), id_to_index)

        if (
            np.all(np.isfinite(pelvis))
            and np.all(np.isfinite(shoulder))
            and np.all(np.isfinite(knee))
        ):
            angle_series[t] = angle_deg(shoulder, pelvis, knee)

    return {"torso_knee_angle": angle_series}


def compute_knee_difference(
    kpts: np.ndarray,
    id_to_index: Dict[int, int],
) -> Dict[str, np.ndarray]:
    """Compute left-right knee angle difference over time.

    Args:
        kpts: (T,J,3) keypoints array
        id_to_index: joint ID to index mapping

    Returns:
        dict: difference time-series
    """
    if kpts.ndim != 3 or kpts.shape[2] != 3:
        raise ValueError("kpts must be (T,J,3)")

    T = kpts.shape[0]
    angle_l = np.full((T,), np.nan, dtype=np.float64)
    angle_r = np.full((T,), np.nan, dtype=np.float64)

    # Compute both knee angles
    for t in range(T):
        frame = kpts[t]

        # Left knee: 9 (Thigh_L), 11 (calf_l), 13 (Foot_L)
        a_l = frame[id_to_index[9]]
        b_l = frame[id_to_index[11]]
        c_l = frame[id_to_index[13]]
        if (
            np.all(np.isfinite(a_l))
            and np.all(np.isfinite(b_l))
            and np.all(np.isfinite(c_l))
        ):
            angle_l[t] = angle_deg(a_l, b_l, c_l)

        # Right knee: 10 (Thigh_R), 12 (calf_r), 14 (Foot_R)
        a_r = frame[id_to_index[10]]
        b_r = frame[id_to_index[12]]
        c_r = frame[id_to_index[14]]
        if (
            np.all(np.isfinite(a_r))
            and np.all(np.isfinite(b_r))
            and np.all(np.isfinite(c_r))
        ):
            angle_r[t] = angle_deg(a_r, b_r, c_r)

    # Compute difference (left - right)
    diff = np.full((T,), np.nan, dtype=np.float64)
    for t in range(T):
        if np.isfinite(angle_l[t]) and np.isfinite(angle_r[t]):
            diff[t] = angle_l[t] - angle_r[t]

    return {"knee_diff_lr": diff}


def compute_elbow_distance_from_midline(
    kpts: np.ndarray,
    id_to_index: Dict[int, int],
) -> Dict[str, np.ndarray]:
    """Compute horizontal distance from elbow to body midline.

    Body midline is the vertical plane passing through pelvis center.
    Distance is measured in the horizontal plane (ignoring Y/vertical axis).

    Args:
        kpts: (T,J,3) keypoints array
        id_to_index: joint ID to index mapping

    Returns:
        dict: distance time-series for left and right elbows
    """
    if kpts.ndim != 3 or kpts.shape[2] != 3:
        raise ValueError("kpts must be (T,J,3)")

    T = kpts.shape[0]
    dist_l = np.full((T,), np.nan, dtype=np.float64)
    dist_r = np.full((T,), np.nan, dtype=np.float64)

    for t in range(T):
        frame = kpts[t]

        # Get pelvis center (body midline point)
        pelvis = _center_from_ids(frame, (9, 10), id_to_index)

        if not np.all(np.isfinite(pelvis)):
            continue

        # Get left elbow position
        elbow_l_idx = id_to_index[7]  # lowerarm_l
        elbow_l = frame[elbow_l_idx]
        if np.all(np.isfinite(elbow_l)):
            # Distance in horizontal plane (XZ plane, excluding Y)
            horizontal_dist_l = np.sqrt(
                (elbow_l[0] - pelvis[0]) ** 2 + (elbow_l[2] - pelvis[2]) ** 2
            )
            dist_l[t] = horizontal_dist_l

        # Get right elbow position
        elbow_r_idx = id_to_index[8]  # lowerarm_r
        elbow_r = frame[elbow_r_idx]
        if np.all(np.isfinite(elbow_r)):
            # Distance in horizontal plane (XZ plane, excluding Y)
            horizontal_dist_r = np.sqrt(
                (elbow_r[0] - pelvis[0]) ** 2 + (elbow_r[2] - pelvis[2]) ** 2
            )
            dist_r[t] = horizontal_dist_r

    return {"elbow_distance_l": dist_l, "elbow_distance_r": dist_r}


def save_angles_csv(out_path: Path, angles: Dict[str, np.ndarray]) -> None:
    angle_names = list(angles.keys())
    T = len(next(iter(angles.values()))) if angles else 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame"] + angle_names)
        for t in range(T):
            row = [t] + [angles[name][t] for name in angle_names]
            writer.writerow(row)


def plot_angles(out_path: Path, angles: Dict[str, np.ndarray]) -> None:
    angle_names = list(angles.keys())
    if not angle_names:
        return

    def _smooth_series(series: np.ndarray, window: int = 11) -> np.ndarray:
        if window < 3:
            return series.copy()
        if window % 2 == 0:
            window += 1
        kernel = np.ones((window,), dtype=np.float64)
        valid = np.isfinite(series).astype(np.float64)
        filled = np.where(np.isfinite(series), series, 0.0)
        num = np.convolve(filled, kernel, mode="same")
        den = np.convolve(valid, kernel, mode="same")
        smoothed = np.full_like(series, np.nan, dtype=np.float64)
        mask = den > 0.0
        smoothed[mask] = num[mask] / den[mask]
        return smoothed

    rows = len(angle_names)
    fig, axes = plt.subplots(rows, 1, figsize=(10, max(3, rows * 2)), sharex=True)
    if rows == 1:
        axes = [axes]

    for ax, name in zip(axes, angle_names):
        series = angles[name]
        raw_line = ax.plot(
            series,
            label=f"{name} (raw)",
            linewidth=1.0,
            alpha=0.55,
            zorder=1,
        )[0]
        ax.plot(
            _smooth_series(series),
            linestyle="--",
            linewidth=2.4,
            color=raw_line.get_color(),
            alpha=0.95,
            zorder=2,
            label=f"{name} (smoothed)",
        )
        ax.set_ylabel("deg")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.3)

    axes[-1].set_xlabel("frame")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def visualize_elbow_position(
    kpts: np.ndarray,
    id_to_index: Dict[int, int],
    output_dir: Path,
) -> None:
    """Visualize elbow positions relative to body midline in horizontal plane.

    Creates a 2D plot showing the trajectory of left and right elbows around the body.

    Args:
        kpts: (T,J,3) keypoints array
        id_to_index: joint ID to index mapping
        output_dir: directory to save visualization
    """
    if kpts.ndim != 3 or kpts.shape[2] != 3:
        raise ValueError("kpts must be (T,J,3)")

    output_dir.mkdir(parents=True, exist_ok=True)
    T = kpts.shape[0]

    # Collect positions
    pelvis_positions = []
    elbow_l_positions = []
    elbow_r_positions = []

    for t in range(T):
        frame = kpts[t]

        # Get pelvis center
        pelvis = _center_from_ids(frame, (9, 10), id_to_index)

        # Get elbow positions
        elbow_l_idx = id_to_index[7]  # lowerarm_l
        elbow_l = frame[elbow_l_idx]

        elbow_r_idx = id_to_index[8]  # lowerarm_r
        elbow_r = frame[elbow_r_idx]

        if np.all(np.isfinite(pelvis)):
            pelvis_positions.append(pelvis)
            if np.all(np.isfinite(elbow_l)):
                elbow_l_positions.append(elbow_l)
            if np.all(np.isfinite(elbow_r)):
                elbow_r_positions.append(elbow_r)

    if not pelvis_positions:
        print("No valid pelvis positions found")
        return

    pelvis_pos = np.array(pelvis_positions)
    elbow_l_pos = np.array(elbow_l_positions) if elbow_l_positions else None
    elbow_r_pos = np.array(elbow_r_positions) if elbow_r_positions else None

    # Create overhead view (XZ plane)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Overhead view (XZ plane)
    ax1.scatter(
        pelvis_pos[:, 0], pelvis_pos[:, 2], c="black", s=20, label="Pelvis", alpha=0.5
    )

    if elbow_l_pos is not None:
        ax1.scatter(
            elbow_l_pos[:, 0],
            elbow_l_pos[:, 2],
            c="blue",
            s=20,
            label="Left Elbow",
            alpha=0.5,
        )

    if elbow_r_pos is not None:
        ax1.scatter(
            elbow_r_pos[:, 0],
            elbow_r_pos[:, 2],
            c="red",
            s=20,
            label="Right Elbow",
            alpha=0.5,
        )

    ax1.set_xlabel("X (Left-Right)")
    ax1.set_ylabel("Z (Front-Back)")
    ax1.set_title("Elbow Position Relative to Body - Top View")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis("equal")

    # Plot 2: Time series of horizontal distances
    elbow_l_dist = []
    elbow_r_dist = []

    for t in range(T):
        frame = kpts[t]
        pelvis = _center_from_ids(frame, (9, 10), id_to_index)

        if not np.all(np.isfinite(pelvis)):
            continue

        # Left elbow distance
        elbow_l = frame[id_to_index[7]]
        if np.all(np.isfinite(elbow_l)):
            dist_l = np.sqrt(
                (elbow_l[0] - pelvis[0]) ** 2 + (elbow_l[2] - pelvis[2]) ** 2
            )
            elbow_l_dist.append(dist_l)

        # Right elbow distance
        elbow_r = frame[id_to_index[8]]
        if np.all(np.isfinite(elbow_r)):
            dist_r = np.sqrt(
                (elbow_r[0] - pelvis[0]) ** 2 + (elbow_r[2] - pelvis[2]) ** 2
            )
            elbow_r_dist.append(dist_r)

    ax2.plot(elbow_l_dist, label="Left Elbow", color="blue", alpha=0.7)
    ax2.plot(elbow_r_dist, label="Right Elbow", color="red", alpha=0.7)
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Horizontal Distance (m)")
    ax2.set_title("Elbow Distance from Body Midline Over Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path = output_dir / "elbow_position_visualization.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Elbow position visualization saved to: {output_path}")


def visualize_3d_keypoints(
    kpts: np.ndarray,
    id_to_index: Dict[int, int],
    output_dir: Path,
    num_frames_to_save: int = 5,
) -> None:
    """Visualize 3D keypoints as a skeleton.

    Args:
        kpts: (T,J,3) keypoints array
        id_to_index: joint ID to index mapping
        output_dir: directory to save visualization images
        num_frames_to_save: number of keyframes to save (evenly distributed)
    """
    if kpts.ndim != 3 or kpts.shape[2] != 3:
        raise ValueError("kpts must be (T,J,3)")

    T = kpts.shape[0]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select frames to visualize (evenly distributed)
    if T <= num_frames_to_save:
        frame_indices = list(range(T))
    else:
        frame_indices = np.linspace(0, T - 1, num_frames_to_save, dtype=int).tolist()

    for frame_idx in frame_indices:
        frame = kpts[frame_idx]

        # Create 3D plot
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Plot skeleton connections
        for parent_id, child_id in SKELETON_CONNECTIONS:
            parent_idx = id_to_index.get(parent_id)
            child_idx = id_to_index.get(child_id)

            if parent_idx is None or child_idx is None:
                continue

            parent_pos = frame[parent_idx]
            child_pos = frame[child_idx]

            # Only plot if both joints are valid (finite values)
            if np.all(np.isfinite(parent_pos)) and np.all(np.isfinite(child_pos)):
                ax.plot(
                    [parent_pos[0], child_pos[0]],
                    [parent_pos[1], child_pos[1]],
                    [parent_pos[2], child_pos[2]],
                    "b-",
                    linewidth=2,
                    alpha=0.7,
                )

        # Plot keypoints
        valid_points = []
        for jid, name in UNITY_MHR70_MAPPING.items():
            idx = id_to_index.get(jid)
            if idx is None:
                continue

            pos = frame[idx]
            if np.all(np.isfinite(pos)):
                valid_points.append(pos)
                ax.scatter(*pos, c="r", s=50, alpha=0.8)

        # Set axis labels and title
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"3D Skeleton - Frame {frame_idx}")

        # Set equal aspect ratio for better visualization
        if valid_points:
            positions = np.array(valid_points)
            min_coords = positions.min(axis=0)
            max_coords = positions.max(axis=0)
            max_range = (max_coords - min_coords).max() / 2.0
            mid_x = (min_coords[0] + max_coords[0]) * 0.5
            mid_y = (min_coords[1] + max_coords[1]) * 0.5
            mid_z = (min_coords[2] + max_coords[2]) * 0.5

            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Save figure
        output_vis_dir = output_dir / "skeleton_frames"
        output_vis_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_vis_dir / f"skeleton_frame_{frame_idx:04d}.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved skeleton visualization: {output_path}")


def main():
    input_path = Path("/workspace/data/dual_view_pose/fused_smoothed_results")
    output_dir = Path("/workspace/data/dual_view_pose/angle_outputs")

    # Process direct npy files (single-stream reports).
    for person in sorted(input_path.glob("*.npy")):
        person_name = person.stem
        print(f"Processing person: {person_name}")
        person_output_dir = output_dir / "pair_not_turn_comparison" / person_name
        person_output_dir.mkdir(parents=True, exist_ok=True)
        process_person(person, person_output_dir)

    # Process fused/smoothed paired files for turn-by-turn comparison.
    pair_dir = input_path / "person_pairs"
    if pair_dir.exists() and pair_dir.is_dir():
        smoothed_files = sorted(pair_dir.glob("*_smoothed.npy"))
        for smoothed in smoothed_files:
            base_name = smoothed.stem.replace("_smoothed", "")
            fused = pair_dir / f"{base_name}_fused.npy"
            if not fused.exists():
                print(f"Skip pair {base_name}: fused file not found")
                continue

            pair_out = output_dir / "pair_turn_comparison" / base_name
            pair_out.mkdir(parents=True, exist_ok=True)
            print(f"Processing pair: {base_name}")
            process_person_pair(smoothed, fused, pair_out)


def process_person(input_path: Path, output_dir: Path) -> None:
    kpts = np.load(input_path)
    fullframe_dir = output_dir / "fullframe"
    by_turn_dir = output_dir / "by_turn"
    fullframe_dir.mkdir(parents=True, exist_ok=True)
    by_turn_dir.mkdir(parents=True, exist_ok=True)

    up_axis_y_down = np.array([0.0, -1.0, 0.0], dtype=np.float64)  # Y-axis down
    (
        joint_angles,
        body_angles_y_down,
        torso_knee,
        knee_diff,
        elbow_dist,
        heading_deg,
        turns,
    ) = _compute_all_series(kpts, up_axis_y_down)

    # Save joint angles
    joint_csv = fullframe_dir / "angles_joint.csv"
    joint_png = fullframe_dir / "angles_joint.png"
    save_angles_csv(joint_csv, joint_angles)
    plot_angles(joint_png, joint_angles)
    print(f"Joint angles saved to: {joint_csv}")
    print(f"Joint plot saved to: {joint_png}")

    # Save knee angles separately (left and right knee angles)
    knee_angles = {
        "knee_l": joint_angles["knee_l"],
        "knee_r": joint_angles["knee_r"],
    }
    knee_csv = fullframe_dir / "angles_knee.csv"
    knee_png = fullframe_dir / "angles_knee.png"
    save_angles_csv(knee_csv, knee_angles)
    plot_angles(knee_png, knee_angles)
    print(f"Knee angles saved to: {knee_csv}")
    print(f"Knee plot saved to: {knee_png}")

    # Save torso-knee angle
    torso_csv = fullframe_dir / "angles_torso_knee.csv"
    torso_png = fullframe_dir / "angles_torso_knee.png"
    save_angles_csv(torso_csv, torso_knee)
    plot_angles(torso_png, torso_knee)
    print(f"Torso-knee angle saved to: {torso_csv}")
    print(f"Torso-knee plot saved to: {torso_png}")

    # Save knee difference
    knee_diff_csv = fullframe_dir / "angles_knee_diff.csv"
    knee_diff_png = fullframe_dir / "angles_knee_diff.png"
    save_angles_csv(knee_diff_csv, knee_diff)
    plot_angles(knee_diff_png, knee_diff)
    print(f"Knee difference saved to: {knee_diff_csv}")
    print(f"Knee difference plot saved to: {knee_diff_png}")

    # Save elbow distances
    elbow_csv = fullframe_dir / "distance_elbow_midline.csv"
    elbow_png = fullframe_dir / "distance_elbow_midline.png"
    save_angles_csv(elbow_csv, elbow_dist)
    plot_angles(elbow_png, elbow_dist)
    print(f"Elbow distances saved to: {elbow_csv}")
    print(f"Elbow distance plot saved to: {elbow_png}")

    # Save body angles (Y-axis down)
    body_csv_y_down = fullframe_dir / "angles_body_y_down.csv"
    body_png_y_down = fullframe_dir / "angles_body_y_down.png"
    save_angles_csv(body_csv_y_down, body_angles_y_down)
    plot_angles(body_png_y_down, body_angles_y_down)
    print(f"Body angles (Y-down) saved to: {body_csv_y_down}")
    print(f"Body plot (Y-down) saved to: {body_png_y_down}")

    # Base per-frame series.
    base_series = {
        **joint_angles,
        **torso_knee,
        **knee_diff,
        **elbow_dist,
        **body_angles_y_down,
    }

    # Save full-frame changes before turn splitting.
    change_series = save_fullframe_change_reports(fullframe_dir, base_series)

    # Turn-level reports also include change metrics.
    report_series = {**base_series, **change_series}
    save_turn_reports(by_turn_dir, turns, heading_deg, report_series)
    print(f"Turn summary saved to: {by_turn_dir / 'turn_summary.csv'}")
    print(f"Turn metrics saved to: {by_turn_dir / 'turn_metrics.csv'}")

    # Visualize elbow position relative to body
    visualize_elbow_position(kpts, ID_TO_INDEX, fullframe_dir)

    # Visualize 3D keypoints
    visualization_dir = fullframe_dir / "skeleton_visualization"
    visualize_3d_keypoints(kpts, ID_TO_INDEX, visualization_dir, num_frames_to_save=5)
    print(f"3D skeleton visualization saved to: {visualization_dir}")


def process_person_pair(
    smoothed_path: Path,
    fused_path: Path,
    output_dir: Path,
) -> None:
    """Generate turn reports for before/after and save pairwise comparison."""
    kpts_before = np.load(smoothed_path)
    kpts_after = np.load(fused_path)
    up_axis_y_down = np.array([0.0, -1.0, 0.0], dtype=np.float64)

    (
        joint_before,
        body_before,
        torso_before,
        knee_before,
        elbow_before,
        heading_before,
        turns_before,
    ) = _compute_all_series(kpts_before, up_axis_y_down)

    (
        joint_after,
        body_after,
        torso_after,
        knee_after,
        elbow_after,
        heading_after,
        turns_after,
    ) = _compute_all_series(kpts_after, up_axis_y_down)

    before_base_series = {
        **joint_before,
        **torso_before,
        **knee_before,
        **elbow_before,
        **body_before,
    }
    after_base_series = {
        **joint_after,
        **torso_after,
        **knee_after,
        **elbow_after,
        **body_after,
    }

    before_dir = output_dir / "before_smoothed"
    after_dir = output_dir / "after_fused"
    before_fullframe_dir = before_dir / "fullframe"
    before_by_turn_dir = before_dir / "by_turn"
    after_fullframe_dir = after_dir / "fullframe"
    after_by_turn_dir = after_dir / "by_turn"

    before_fullframe_dir.mkdir(parents=True, exist_ok=True)
    before_by_turn_dir.mkdir(parents=True, exist_ok=True)
    after_fullframe_dir.mkdir(parents=True, exist_ok=True)
    after_by_turn_dir.mkdir(parents=True, exist_ok=True)

    before_change_series = save_fullframe_change_reports(
        before_fullframe_dir,
        before_base_series,
    )
    after_change_series = save_fullframe_change_reports(
        after_fullframe_dir,
        after_base_series,
    )

    before_series = {**before_base_series, **before_change_series}
    after_series = {**after_base_series, **after_change_series}

    save_turn_reports(before_by_turn_dir, turns_before, heading_before, before_series)
    save_turn_reports(after_by_turn_dir, turns_after, heading_after, after_series)

    compare_csv = output_dir / "turn_compare_fused_vs_smoothed.csv"
    save_turn_comparison_report(
        compare_csv,
        turns_before,
        turns_after,
        before_series,
        after_series,
    )
    print(f"Pair turn comparison saved to: {compare_csv}")


if __name__ == "__main__":
    main()
