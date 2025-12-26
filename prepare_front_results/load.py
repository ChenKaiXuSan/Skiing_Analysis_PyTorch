#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/front/load.py
Project: /workspace/code/front
Created Date: Friday December 12th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Friday December 12th 2025 4:25:13 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from torchvision.io import read_video

logger = logging.getLogger(__name__)


def _load_npz(file_path: str) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")
    logger.info(f"Loading: {file_path}")
    return dict(np.load(file_path, allow_pickle=True).item())


def load_info(
    pt_file_path: Path,
    video_file_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    从 Detectron2 的 pt_info 中读取关键点与 bbox。
    返回: (keypoints_xy, keypoints_score, bboxes_xyxy, bbox_scores, frames)

    - keypoints_xy: (T, K, 2)  像素坐标 (若可判定为归一化且有 H/W，则已反归一化)
    - keypoints_score: (T, K)  若无则从 kpts[...,2] 回退；再无则全 1
    - bboxes_xyxy: (T, N, 4)   每行为 [x1, y1, x2, y2]（同样会按需反归一化）
    - bbox_scores: (T, N)      若无则全 1
    - frames: (T, H, W, C) 的 torch.uint8，若 return_frames=False 则为 None
    """
    data = _load_npz(pt_file_path)

    # --------- frames / shape ----------
    frames = read_video(video_file_path, pts_unit="sec", output_format="THWC")[0]

    return data
