#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/vis_3d_kpt/loag.py
Project: /workspace/code/vis_3d_kpt
Created Date: Friday March 13th 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Friday March 13th 2026 5:09:45 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import numpy as np
from pathlib import Path
from typing import Any, Optional

from .main import OnePersonInfo

from dataclasses import dataclass

@dataclass
class OnePersonInfo:
    left_video_path: Path,
    right_video_path: Path,
    left_2d_kpt_path: Path,
    right_2d_kpt_path: Path,
    fused_3d_kpt_path: Path,
    fused_smoothed_3d_kpt_path: Path,



def load_pose_sequence(
    file_path: Path,
) -> np.ndarray:
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    loaded = np.load(file_path, allow_pickle=True)
    data = unwrap_loaded_object(loaded)
    pose = extract_pose_array(data)
    pose = np.asarray(pose, dtype=np.float64)

    if pose.ndim == 2:
        pose = pose[None, ...]
    if pose.ndim != 3 or pose.shape[-1] != 3:
        raise ValueError(f"3D 结果必须是 (T,J,3) 或 (J,3)，当前为 {pose.shape}")
    return pose


def unwrap_loaded_object(loaded: Any) -> Any:
    if isinstance(loaded, np.ndarray) and loaded.dtype != object:
        return loaded

    if isinstance(loaded, np.ndarray) and loaded.shape == ():
        return loaded.item()

    if isinstance(loaded, np.ndarray) and loaded.dtype == object:
        return loaded.tolist()

    return loaded


def load_2d_keypoints(file_path: Path) -> np.ndarray:
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    loaded = np.load(file_path, allow_pickle=True)
    data = unwrap_loaded_object(loaded)

    if isinstance(data, dict) and "pred_keypoints_2d" in data:
        kpts_2d = data["pred_keypoints_2d"]
        return np.asarray(kpts_2d, dtype=np.float64)

    raise ValueError(f"无法从 {file_path} 中提取 2D 关键点，数据结构不符合预期")


def load_video_frames(video_path: Path) -> list:
    if not video_path.exists():
        raise FileNotFoundError(f"文件不存在: {video_path}")

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames


def load_helper(
    person_info: OnePersonInfo,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    left_frames = (
        load_video_frames(person_info.left_video_path)
        if person_info.left_video_path is not None
        else None
    )
    right_frames = (
        load_video_frames(person_info.right_video_path)
        if person_info.right_video_path is not None
        else None
    )
    left_2d_kpt = (
        load_2d_keypoints(person_info.left_2d_kpt_path)
        if person_info.left_2d_kpt_path is not None
        else None
    )
    right_2d_kpt = (
        load_2d_keypoints(person_info.right_2d_kpt_path)
        if person_info.right_2d_kpt_path is not None
        else None
    )
    fused_3d_kpt = (
        load_pose_sequence(person_info.fused_3d_kpt_path)
        if person_info.fused_3d_kpt_path is not None
        else None
    )

    fused_smoothed_3d_kpt = (
        load_pose_sequence(person_info.fused_smoothed_3d_kpt_path)
        if person_info.fused_smoothed_3d_kpt_path is not None
        else None
    )

    return left_frames, right_frames, left_2d_kpt, right_2d_kpt, fused_3d_kpt, fused_smoothed_3d_kpt    
