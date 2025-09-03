#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/triangulation/load.py
Project: /workspace/code/triangulation
Created Date: Wednesday September 3rd 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Wednesday September 3rd 2025 10:02:14 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import os
import numpy as np
import torch
from torchvision.io import read_video


# ---------- 加载关键点 ----------
def load_keypoints_from_pt(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")
    print(f"[INFO] Loading: {file_path}")
    data = torch.load(file_path, map_location="cpu")
    video_path = data.get("video_path", None)
    vframes = (
        read_video(video_path, pts_unit="sec", output_format="THWC")[0]
        if video_path
        else None
    )
    keypoints = np.array(data["keypoint"]["keypoint"]).squeeze(0)
    keypoints_score = np.array(data["keypoint"]["keypoint_score"]).squeeze(0)
    if keypoints.ndim != 3 or keypoints.shape[2] != 2:
        raise ValueError(f"Invalid shape: {keypoints.shape}")
    if vframes is not None:
        keypoints[:, :, 0] *= vframes.shape[2]
        keypoints[:, :, 1] *= vframes.shape[1]
    return keypoints, keypoints_score, vframes
