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
from typing import Tuple

import cv2
import numpy as np

from .bev_utils import make_bev
from .load import load_info


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


def _unnormalize_bbox(bbox: np.ndarray, img_size: Tuple[int, int]) -> np.ndarray:
    """
    将归一化 bbox 还原为像素坐标

    Args:
        bbox: (..., 4) 归一化 bbox
              支持格式: [x1, y1, x2, y2]，范围 [0,1]
        img_size: (H, W)

    Returns:
        bbox_px: (..., 4) 像素坐标 bbox [x1, y1, x2, y2]
    """
    bbox = np.asarray(bbox, dtype=np.float64)
    H, W = img_size

    if bbox.shape[-1] != 4:
        raise ValueError(f"bbox last dim must be 4, got {bbox.shape}")

    bbox_px = bbox.copy()
    bbox_px[..., 0] *= W  # x1
    bbox_px[..., 2] *= W  # x2
    bbox_px[..., 1] *= H  # y1
    bbox_px[..., 3] *= H  # y2

    return bbox_px


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
        bbox_xyxy = info["out_boxes_xywh"]  # np.ndarray, (4,) xyxy
        obj_ids = info.get("out_obj_ids", None)
        probs = info.get("out_probs", None)
        binary_masks = info.get("out_binary_masks")

        process_one_frame(image, bbox_xyxy, output_dir=output_dir, frame_idx=frame_idx)


def process_one_frame(
    image: np.ndarray,
    bbox_xywh: np.ndarray,
    output_dir: Path,
    frame_idx: int
):
    """处理单帧图像的主函数占位符

    Args:
        image: np.ndarray, (H,W,3)
        bbox_xyxy: np.ndarray, (4,) xyxy

    Returns:
        None
    """

    img = image
    H, W = 1080, 1920
    bbox_xyxy = _convert_xywh_to_xyxy(bbox_xywh)  # np.ndarray
    
    bbox_xyxy = _unnormalize_bbox(bbox_xyxy, (H, W))

    # reshap img 
    img = cv2.resize(img, (W, H))   # (W, H)
    make_bev(img=img, bboxes_xyxy=bbox_xyxy, out_dir=output_dir / f"frame{frame_idx}")
