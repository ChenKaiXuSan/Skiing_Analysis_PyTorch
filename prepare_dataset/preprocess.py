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

from prepare_dataset.yolov11 import MultiPreprocess
from prepare_dataset.depth_estimation import DepthEstimator
from prepare_dataset.optical_flow import OpticalFlow

logger = logging.getLogger(__name__)


class Preprocess:
    def __init__(self, config) -> None:
        super(Preprocess, self).__init__()

        self.task = config.task

        if "pose" in self.task or "bbox" in self.task or "mask" in self.task:
            self.yolo_model = MultiPreprocess(config)

        if "depth" in self.task:
            self.depth_estimator = DepthEstimator(config)

        if "optical_flow" in self.task:
            self.of_model = OpticalFlow(config)

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

    def __call__(self, vframes: torch.tensor, video_path: Path):
        """
        forward preprocess method for one batch.

        Args:
            batch (torch.tensor): batch imgs, (b, c, t, h, w)
            batch_idx (int): epoch index.

        Returns:
            list: list for different moddailty, return video, bbox_non_index, labels, bbox, mask, pose
        """
        # TODO：want to use this to control the preprocess flow.
        # * process depth
        if self.depth_estimator:
            depth = self.depth_estimator(vframes, video_path)

        # * process mask, pose, bbox
        if self.yolo_model:
            video, bbox_none_index, bbox, mask, pose, pose_score = self.yolo_model(
                vframes, video_path
            )

        # * process optical flow
        if self.of_model:
            optical_flow = self.of_model(vframes)

        # shape check
        # self.shape_check([video, mask, bbox, pose, optical_flow])

        return video, bbox_none_index, optical_flow, bbox, mask, pose, pose_score, depth
