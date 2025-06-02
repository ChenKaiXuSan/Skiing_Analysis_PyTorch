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

import shutil
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.io import write_png
from torchvision.utils import flow_to_image
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large, raft_small

# TODO: should save the ckpt to path.


logger = logging.getLogger(__name__)


class OpticalFlow(nn.Module):
    def __init__(self, param):
        super().__init__()

        self.weights = Raft_Large_Weights.DEFAULT
        self.transforms = self.weights.transforms()

        self.device = param.device
        # define the network
        self.model = raft_large(weights=self.weights, progress=False).to(self.device)

    def get_Optical_flow(self, frame_batch):
        """
        catch one by one batch optical flow, use RAFT method.

        Args:
            frame_batch (tensor): one batch frame, (c, f, h, w)

        Returns:
            tensor: one batch pred optical flow
        """

        c, f, h, w = frame_batch.shape

        frame_batch = frame_batch.permute(1, 0, 2, 3).to(
            self.device
        )  # c, f, h, w to f, c, h, w

        # prepare the img
        current_frame = frame_batch[:-1, :, :, :]  # 0~-1 frame
        next_frame = frame_batch[1:, :, :, :]  # 1~last frame

        # start predict
        self.model.eval()
        pred_flows = []

        interval = (
            10  # the interval for the OF model predict, because the model is too large.
        )

        with torch.no_grad():
            for i in range(0, f, interval):

                # todo: maybe under scalse the img size.
                # transforms
                current_frame_batch, next_frame_batch = self.transforms(
                    current_frame[i : i + interval], next_frame[i : i + interval]
                )
                temp_pred_flows = self.model(
                    current_frame_batch,
                    next_frame_batch,
                )[-1]
                pred_flows.append(temp_pred_flows)

        # empty cache
        torch.cuda.empty_cache()

        return torch.cat(pred_flows, dim=0)  # f, c, h, w

    def process_batch(self, batch):
        """
        predict one batch optical flow.

        Args:
            batch (nn.Tensor): batches of videos. (b, c, f, h, w)

        Returns:
            nn.Tensor: stacked predict optical flow, (b, 2, f, h, w)
        """

        f, h, w, c = batch.shape

        pred_optical_flow_list = []

        for batch_index in range(b):
            one_batch_pred_flow = self.get_Optical_flow(
                batch[batch_index]
            )  # f, c, h, w
            pred_optical_flow_list.append(one_batch_pred_flow)

        return torch.stack(pred_optical_flow_list).permute(
            0, 2, 1, 3, 4
        )  # b, c, f, h, w
