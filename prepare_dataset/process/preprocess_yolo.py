#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/Skiing_Analysis_PyTorch/prepare_dataset/preprocess.py
Project: /workspace/code/Skiing_Analysis_PyTorch/prepare_dataset
Created Date: Wednesday April 23rd 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Wednesday April 23rd 2025 12:56:44 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import logging
from pathlib import Path

import torch

from prepare_dataset.model.depth_estimation import DepthEstimator
from prepare_dataset.model.optical_flow import OpticalFlow
from prepare_dataset.model.yolov11_bbox import YOLOv11Bbox
from prepare_dataset.model.yolov11_pose import YOLOv11Pose
from prepare_dataset.model.yolov11_mask import YOLOv11Mask

from prepare_dataset.model.detectron2 import Detectron2Wrapper

logger = logging.getLogger(__name__)


class PreprocessYOLO:
    def __init__(self, config) -> None:
        super(PreprocessYOLO, self).__init__()

        self.task = config.task
        logger.info(f"Preprocess task: {self.task}")

        if "bbox" in self.task:
            self.yolo_model_bbox = YOLOv11Bbox(config)
        else:
            self.yolo_model_bbox = None

        if "pose" in self.task:
            self.yolo_model_pose = YOLOv11Pose(config)
        else:
            self.yolo_model_pose = None

        if "mask" in self.task:
            self.yolo_model_mask = YOLOv11Mask(config)
        else:
            self.yolo_model_mask = None

    def __call__(self, vframes: torch.Tensor, video_path: Path):

        # * process bbox
        if self.yolo_model_bbox:
            # use MultiPreprocess to process bbox, mask, pose
            bbox, bbox_none_index, bbox_results = self.yolo_model_bbox(
                vframes, video_path
            )
        else:
            bbox_none_index = []
            bbox = torch.empty((0, 4), dtype=torch.float32)

        # * process pose
        if self.yolo_model_pose:
            pose, pose_score, pose_none_index, pose_results = self.yolo_model_pose(
                vframes, video_path
            )
        else:
            pose = torch.empty((0, 17, 3), dtype=torch.float32)
            pose_score = torch.empty((0, 17), dtype=torch.float32)

        # * process mask
        if self.yolo_model_mask:
            mask, mask_none_index, mask_results = self.yolo_model_mask(
                vframes, video_path
            )
        else:
            mask = torch.empty(
                (0, 1, vframes.shape[1], vframes.shape[2]), dtype=torch.float32
            )

        return (
            bbox_none_index,
            bbox,
            mask,
            pose,
            pose_score,
        )
