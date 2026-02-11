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

import csv
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
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

    rows = len(angle_names)
    fig, axes = plt.subplots(rows, 1, figsize=(10, max(3, rows * 2)), sharex=True)
    if rows == 1:
        axes = [axes]

    for ax, name in zip(axes, angle_names):
        ax.plot(angles[name], label=name)
        ax.set_ylabel("deg")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.3)

    axes[-1].set_xlabel("frame")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    input_path = Path("/workspace/data/fused_smoothed_results")
    output_dir = Path("/workspace/data/angle_outputs")

    for person in input_path.iterdir():
        person_name = person.stem
        print(f"Processing person: {person_name}")

        person_output_dir = output_dir / person_name
        person_output_dir.mkdir(parents=True, exist_ok=True)

        process_person(person, person_output_dir)


def process_person(input_path: Path, output_dir: Path) -> None:
    kpts = np.load(input_path)
    joint_angles = compute_angles(kpts, ANGLE_DEFS, ID_TO_INDEX)

    # Compute body tilt angles for both Y-axis directions
    up_axis_y_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)  # Y-axis up
    up_axis_y_down = np.array([0.0, -1.0, 0.0], dtype=np.float64)  # Y-axis down
    
    body_angles_y_up = compute_tilt_angles(kpts, ID_TO_INDEX, up_axis_y_up)
    body_angles_y_down = compute_tilt_angles(kpts, ID_TO_INDEX, up_axis_y_down)

    # Save joint angles
    joint_csv = output_dir / "angles_joint.csv"
    joint_png = output_dir / "angles_joint.png"
    save_angles_csv(joint_csv, joint_angles)
    plot_angles(joint_png, joint_angles)
    print(f"Joint angles saved to: {joint_csv}")
    print(f"Joint plot saved to: {joint_png}")

    # Save body angles (Y-axis up)
    # body_csv_y_up = output_dir / "angles_body_y_up.csv"
    # body_png_y_up = output_dir / "angles_body_y_up.png"
    # save_angles_csv(body_csv_y_up, body_angles_y_up)
    # plot_angles(body_png_y_up, body_angles_y_up)
    # print(f"Body angles (Y-up) saved to: {body_csv_y_up}")
    # print(f"Body plot (Y-up) saved to: {body_png_y_up}")

    # Save body angles (Y-axis down)
    body_csv_y_down = output_dir / "angles_body_y_down.csv"
    body_png_y_down = output_dir / "angles_body_y_down.png"
    save_angles_csv(body_csv_y_down, body_angles_y_down)
    plot_angles(body_png_y_down, body_angles_y_down)
    print(f"Body angles (Y-down) saved to: {body_csv_y_down}")
    print(f"Body plot (Y-down) saved to: {body_png_y_down}")


if __name__ == "__main__":
    main()
