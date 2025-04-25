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

from prepare_dataset.yolov11 import MultiPreprocess


class OpticalFlow(nn.Module):
    def __init__(self, param):
        super().__init__()

        self.weights = Raft_Large_Weights.DEFAULT
        self.transforms = self.weights.transforms()

        self.device = param.YOLO.device
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

        b, c, f, h, w = batch.shape

        pred_optical_flow_list = []

        for batch_index in range(b):
            one_batch_pred_flow = self.get_Optical_flow(
                batch[batch_index]
            )  # f, c, h, w
            pred_optical_flow_list.append(one_batch_pred_flow)

        return torch.stack(pred_optical_flow_list).permute(
            0, 2, 1, 3, 4
        )  # b, c, f, h, w


class Preprocess(nn.Module):
    def __init__(self, config) -> None:
        super(Preprocess, self).__init__()

        self.yolo_model = MultiPreprocess(config.YOLO)

        # ! notice: the OF method have err, the reason do not know.
        if config.OF:
            self.of_model = OpticalFlow(config)
        else:
            self.of_model = None

    def save_img(self, batch: torch.tensor, flag: str, epoch_idx: int):
        """
        save_img save batch to png, for analysis.

        Args:
            batch (torch.tensor): the batch of imgs (b, c, t, h, w)
            flag (str): flag for filename, ['optical_flow', 'mask']
            batch_idx (int): epoch index.
        """

        pref_Path = Path("/workspace/skeleton/logs/img_test")
        # only rmtree in 1 epoch, and in one flag (mask).
        if pref_Path.exists() and epoch_idx == 0 and flag == "mask":
            shutil.rmtree(pref_Path)
            pref_Path.mkdir(parents=True)

        b, c, t, h, w = batch.shape

        for b_idx in range(b):
            for frame in range(t):
                if flag == "optical_flow":
                    inp = flow_to_image(batch[b_idx, :, frame, ...])
                else:
                    inp = batch[b_idx, :, frame, ...] * 255
                    inp = inp.to(torch.uint8)

                write_png(
                    input=inp.cpu(),
                    filename=str(
                        pref_Path.joinpath(f"{flag}_{epoch_idx}_{b_idx}_{frame}.png")
                    ),
                )

        logging.info("=" * 20)
        logging.info("finish save %s" % flag)

    def shape_check(self, check: list):
        """
        shape_check check the given value shape, and assert the shape.

        check list include:
        # batch, (b, c, t, h, w)
        # bbox, (b, t, 4) (cxcywh)
        # mask, (b, 1, t, h, w)
        # keypoint, (b, t, 17, 2)
        # optical_flow, (b, 2, t, h, w)

        Args:
            check (list): checked value, in list.
        """

        # first value in list is video, use this as reference.
        b, c, t, h, w = check[0].shape

        # frame check, we just need start from 1.
        for ck in check[0:]:
            if ck is None:
                continue
            # for label shape
            if len(ck.shape) == 1:
                assert ck.shape[0] == b
            # for bbox shape
            elif len(ck.shape) == 3:
                assert ck.shape[0] == b and ck.shape[1] == t
            # for mask shape and optical flow shape
            elif len(ck.shape) == 5:
                assert ck.shape[0] == b and (ck.shape[2] == t or ck.shape[2] == t - 1)
            # for keypoint shape
            elif len(ck.shape) == 4:
                assert ck.shape[0] == b and ck.shape[1] == t and ck.shape[2] == 17
            else:
                raise ValueError("shape not match")

    def forward(self, batch: torch.tensor, batch_idx: int):
        """
        forward preprocess method for one batch.

        Args:
            batch (torch.tensor): batch imgs, (b, c, t, h, w)
            batch_idx (int): epoch index.

        Returns:
            list: list for different moddailty, return video, bbox_non_index, labels, bbox, mask, pose
        """

        b, c, t, h, w = batch.shape

        # process mask, pose
        video, bbox_none_index, bbox, mask, pose, pose_score = self.yolo_model(batch)

        # FIXME: OF method have some problem, the reason do not know.
        # when not use OF, return None value.
        if self.of_model is not None:
            optical_flow = self.of_model.process_batch(batch)
        else:
            optical_flow = None

        # shape check
        self.shape_check([video, mask, bbox, pose, optical_flow])

        return video, bbox_none_index, optical_flow, bbox, mask, pose, pose_score
