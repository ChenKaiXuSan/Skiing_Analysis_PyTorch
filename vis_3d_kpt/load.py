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

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class OnePersonInfo:
    person_name: str
    left_video_path: Path = Path("")
    right_video_path: Path = Path("")
    left_2d_kpt_path: Path = Path("")
    right_2d_kpt_path: Path = Path("")
    fused_3d_kpt_path: Path = Path("")
    fused_smoothed_3d_kpt_path: Path = Path("")


def load_fused_3d_kpt(
    file_path: Path,
) -> np.ndarray:
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    loaded = np.load(file_path, allow_pickle=True)

    return loaded


def load_2d_keypoints(file_path: Path) -> np.ndarray:
    res_list = []
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    if file_path.is_file() and file_path.suffix in [".npz"]:
        loaded = np.load(file_path, allow_pickle=True)["outputs"]

        for i in loaded:
            if isinstance(i, dict) and "pred_keypoints_2d" in i:
                kpts_2d = i["pred_keypoints_2d"]
                res_list.append(kpts_2d)

    elif file_path.is_dir():
        npz_files = sorted(list(file_path.rglob("*.npz")))
        if not npz_files:
            raise FileNotFoundError(f"目录 {file_path} 中未找到 .npz 文件")

        for npz_file in npz_files:
            loaded = np.load(npz_file, allow_pickle=True)
            kpts_2d = loaded["outputs"][0]["pred_keypoints_2d"]
            res_list.append(kpts_2d)

    else:
        raise ValueError(f"无法处理的文件路径: {file_path}")

    del loaded
    return np.asarray(res_list, dtype=np.float64)


def load_video_frames(video_path: Path) -> list:
    if not video_path.exists():
        raise FileNotFoundError(f"文件不存在: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()

    return np.asarray(frames, dtype=np.uint8)


def load_helper(
    person_info: OnePersonInfo,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left_frames = load_video_frames(person_info.left_video_path)
    right_frames = load_video_frames(person_info.right_video_path)
    left_2d_kpt = load_2d_keypoints(person_info.left_2d_kpt_path)
    right_2d_kpt = load_2d_keypoints(person_info.right_2d_kpt_path)

    fused_3d_kpt = load_fused_3d_kpt(person_info.fused_3d_kpt_path)
    fused_smoothed_3d_kpt = load_fused_3d_kpt(person_info.fused_smoothed_3d_kpt_path)

    return (
        left_frames,
        right_frames,
        left_2d_kpt,
        right_2d_kpt,
        fused_3d_kpt,
        fused_smoothed_3d_kpt,
    )
