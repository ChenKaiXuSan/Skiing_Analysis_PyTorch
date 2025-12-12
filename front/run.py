#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/front/run.py
Project: /workspace/code/front
Created Date: Friday December 12th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Friday December 12th 2025 3:22:52 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from pathlib import Path
import torch
import numpy as np
from .load import load_info
from .bev_utils import make_bev


def _convert_xywh_to_xyxy(bboxes_xywh: np.ndarray) -> np.ndarray:
    """
    将 (N,4) 的 xywh 转为 xyxy。
    """
    if bboxes_xywh.ndim != 2 or bboxes_xywh.shape[1] != 4:
        raise ValueError(f"Invalid bbox shape for xywh to xyxy: {bboxes_xywh.shape}")
    x = bboxes_xywh[:, 0]
    y = bboxes_xywh[:, 1]
    w = bboxes_xywh[:, 2]
    h = bboxes_xywh[:, 3]
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    bboxes_xyxy = np.stack([x1, y1, x2, y2], axis=-1)
    return bboxes_xyxy


def process_one_person(subject_data, video_path: Path, output_dir: Path):
    """处理单个被试的数据的主函数占位符

    Args:
        subject_data: SubjectData 实例，包含该被试的所有数据路径
        cfg: 配置对象

    Returns:
        None
    """

    data = load_info(pt_file_path=subject_data, video_file_path=video_path)

    for frame_idx, info in data.items():
        image = info["frame"]  # np.ndarray, (H,W,3)
        bbox_xyxy = info["out_bbox_xyxy"]  # np.ndarray, (4,) xyxy
        process_one_frame(image, bbox_xyxy, output_dir=output_dir)


def process_one_frame(
    image: np.ndarray,
    bbox_xywh: np.ndarray,
    output_dir: Path = None,
):
    """处理单帧图像的主函数占位符

    Args:
        image: np.ndarray, (H,W,3)
        bbox_xyxy: np.ndarray, (4,) xyxy

    Returns:
        None
    """

    img = image
    bbox_xyxy = _convert_xywh_to_xyxy(bbox_xywh)  # np.ndarray

    make_bev(img, bbox_xyxy, out_path=output_dir)
