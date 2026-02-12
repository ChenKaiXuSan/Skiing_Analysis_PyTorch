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
    (69, 5),    # neck -> shoulder_l
    (5, 7),     # shoulder_l -> elbow_l
    (7, 62),    # elbow_l -> hand_l
    # Right arm
    (69, 6),    # neck -> shoulder_r
    (6, 8),     # shoulder_r -> elbow_r
    (8, 41),    # elbow_r -> hand_r
    # Spine
    (69, 9),    # neck -> hip_l
    (69, 10),   # neck -> hip_r
    # Left leg
    (9, 11),    # hip_l -> knee_l
    (11, 13),   # knee_l -> foot_l
    # Right leg
    (10, 12),   # hip_r -> knee_r
    (12, 14),   # knee_r -> foot_r
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
    ax1.scatter(pelvis_pos[:, 0], pelvis_pos[:, 2], c='black', s=20, label='Pelvis', alpha=0.5)
    
    if elbow_l_pos is not None:
        ax1.scatter(elbow_l_pos[:, 0], elbow_l_pos[:, 2], c='blue', s=20, label='Left Elbow', alpha=0.5)
    
    if elbow_r_pos is not None:
        ax1.scatter(elbow_r_pos[:, 0], elbow_r_pos[:, 2], c='red', s=20, label='Right Elbow', alpha=0.5)
    
    ax1.set_xlabel('X (Left-Right)')
    ax1.set_ylabel('Z (Front-Back)')
    ax1.set_title('Elbow Position Relative to Body - Top View')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
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
            dist_l = np.sqrt((elbow_l[0] - pelvis[0])**2 + (elbow_l[2] - pelvis[2])**2)
            elbow_l_dist.append(dist_l)
        
        # Right elbow distance
        elbow_r = frame[id_to_index[8]]
        if np.all(np.isfinite(elbow_r)):
            dist_r = np.sqrt((elbow_r[0] - pelvis[0])**2 + (elbow_r[2] - pelvis[2])**2)
            elbow_r_dist.append(dist_r)
    
    ax2.plot(elbow_l_dist, label='Left Elbow', color='blue', alpha=0.7)
    ax2.plot(elbow_r_dist, label='Right Elbow', color='red', alpha=0.7)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Horizontal Distance (m)')
    ax2.set_title('Elbow Distance from Body Midline Over Time')
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
        ax = fig.add_subplot(111, projection='3d')
        
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
                    'b-', linewidth=2, alpha=0.7
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
                ax.scatter(*pos, c='r', s=50, alpha=0.8)
        
        # Set axis labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'3D Skeleton - Frame {frame_idx}')
        
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
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved skeleton visualization: {output_path}")


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
    up_axis_y_down = np.array([0.0, -1.0, 0.0], dtype=np.float64)  # Y-axis down

    body_angles_y_down = compute_tilt_angles(kpts, ID_TO_INDEX, up_axis_y_down)

    # Compute torso-knee angle
    torso_knee = compute_torso_knee_angle(kpts, ID_TO_INDEX)

    # Compute knee difference
    knee_diff = compute_knee_difference(kpts, ID_TO_INDEX)

    # Compute elbow distance from body midline
    elbow_dist = compute_elbow_distance_from_midline(kpts, ID_TO_INDEX)

    # Save joint angles
    joint_csv = output_dir / "angles_joint.csv"
    joint_png = output_dir / "angles_joint.png"
    save_angles_csv(joint_csv, joint_angles)
    plot_angles(joint_png, joint_angles)
    print(f"Joint angles saved to: {joint_csv}")
    print(f"Joint plot saved to: {joint_png}")

    # Save torso-knee angle
    torso_csv = output_dir / "angles_torso_knee.csv"
    torso_png = output_dir / "angles_torso_knee.png"
    save_angles_csv(torso_csv, torso_knee)
    plot_angles(torso_png, torso_knee)
    print(f"Torso-knee angle saved to: {torso_csv}")
    print(f"Torso-knee plot saved to: {torso_png}")

    # Save knee difference
    knee_diff_csv = output_dir / "angles_knee_diff.csv"
    knee_diff_png = output_dir / "angles_knee_diff.png"
    save_angles_csv(knee_diff_csv, knee_diff)
    plot_angles(knee_diff_png, knee_diff)
    print(f"Knee difference saved to: {knee_diff_csv}")
    print(f"Knee difference plot saved to: {knee_diff_png}")

    # Save elbow distances
    elbow_csv = output_dir / "distance_elbow_midline.csv"
    elbow_png = output_dir / "distance_elbow_midline.png"
    save_angles_csv(elbow_csv, elbow_dist)
    plot_angles(elbow_png, elbow_dist)
    print(f"Elbow distances saved to: {elbow_csv}")
    print(f"Elbow distance plot saved to: {elbow_png}")

    # Save body angles (Y-axis down)
    body_csv_y_down = output_dir / "angles_body_y_down.csv"
    body_png_y_down = output_dir / "angles_body_y_down.png"
    save_angles_csv(body_csv_y_down, body_angles_y_down)
    plot_angles(body_png_y_down, body_angles_y_down)
    print(f"Body angles (Y-down) saved to: {body_csv_y_down}")
    print(f"Body plot (Y-down) saved to: {body_png_y_down}")

    # Visualize elbow position relative to body
    visualize_elbow_position(kpts, ID_TO_INDEX, output_dir)

    # Visualize 3D keypoints
    visualization_dir = output_dir / "skeleton_visualization"
    visualize_3d_keypoints(kpts, ID_TO_INDEX, visualization_dir, num_frames_to_save=5)
    print(f"3D skeleton visualization saved to: {visualization_dir}")

if __name__ == "__main__":
    main()
