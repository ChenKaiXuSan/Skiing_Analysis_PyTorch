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

from prepare_dataset.depth_estimation import DepthEstimator
from prepare_dataset.optical_flow import OpticalFlow
from prepare_dataset.yolov11_bbox import YOLOv11Bbox
from prepare_dataset.yolov11_pose import YOLOv11Pose
from prepare_dataset.yolov11_mask import YOLOv11Mask

logger = logging.getLogger(__name__)


class Preprocess:
    def __init__(self, config) -> None:
        super(Preprocess, self).__init__()

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

        if "depth" in self.task:
            self.depth_estimator = DepthEstimator(config)
        else:
            self.depth_estimator = None

        if "optical_flow" in self.task:
            self.of_model = OpticalFlow(config)
        else:
            self.of_model = None

    # def shape_check(self, check: list):
    #     """
    #     shape_check check the given value shape, and assert the shape.

    #     check list include:
    #     # batch, (b, c, t, h, w)
    #     # bbox, (b, t, 4) (cxcywh)
    #     # mask, (b, 1, t, h, w)
    #     # keypoint, (b, t, 17, 2)
    #     # optical_flow, (b, 2, t, h, w)

    #     Args:
    #         check (list): checked value, in list.
    #     """

    #     # first value in list is video, use this as reference.
    #     t, h, w, c = check[0].shape

    #     # frame check, we just need start from 1.
    #     for ck in check[0:]:
    #         if ck is None:
    #             continue
    #         # for label shape
    #         if len(ck.shape) == 1:
    #             assert ck.shape[0] == b
    #         # for bbox shape
    #         elif len(ck.shape) == 3:
    #             assert ck.shape[0] == b and ck.shape[1] == t
    #         # for mask shape and optical flow shape
    #         elif len(ck.shape) == 5:
    #             assert ck.shape[0] == b and (ck.shape[2] == t or ck.shape[2] == t - 1)
    #         # for keypoint shape
    #         elif len(ck.shape) == 4:
    #             assert ck.shape[0] == b and ck.shape[1] == t and ck.shape[2] == 17
    #         else:
    #             raise ValueError("shape not match")

    def __call__(self, vframes: torch.Tensor, video_path: Path):

        # * process depth
        if self.depth_estimator:
            depth = self.depth_estimator(vframes, video_path)
        else:
            depth = torch.empty(
                (0, 1, vframes.shape[1], vframes.shape[2]), dtype=torch.float32
            )

        # * process optical flow
        if self.of_model:
            optical_flow = self.of_model(vframes, video_path)
        else:
            optical_flow = torch.empty(
                (0, 2, vframes.shape[1], vframes.shape[2]), dtype=torch.float32
            )

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

        # * shape check

        return bbox_none_index, optical_flow, bbox, mask, pose, pose_score, depth
