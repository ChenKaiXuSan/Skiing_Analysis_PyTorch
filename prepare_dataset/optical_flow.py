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
from tqdm import tqdm

import torch
import torch.nn as nn

from torchvision.utils import flow_to_image
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large


from utils.utils import merge_frame_to_video
from torchvision.io import write_png


logger = logging.getLogger(__name__)


class OpticalFlow(nn.Module):
    def __init__(self, param):
        super().__init__()

        self.weights = Raft_Large_Weights.DEFAULT
        self.transforms = self.weights.transforms()

        self.device = param.device
        # define the network
        self.model = raft_large(weights=self.weights, progress=False).to(self.device)

        self.save = param.optical_flow.save
        self.save_path = Path(param.extract_dataset.save_path)

    def get_Optical_flow(self, frame_batch):
        """
        catch one by one batch optical flow, use RAFT method.

        Args:
            frame_batch (tensor): one batch frame, (c, f, h, w)

        Returns:
            tensor: one batch pred optical flow
        """

        f, h, w, c = frame_batch.shape

        frame_batch = frame_batch.permute(0, 3, 1, 2).to(
            self.device
        )  # f, h, w, c to f, c, h, w

        # prepare the img
        current_frame = frame_batch[:-1, :, :, :]  # 0~-1 frame
        next_frame = frame_batch[1:, :, :, :]  # 1~last frame

        # start predict
        self.model.eval()
        pred_flows = []

        interval = (
            2  # the interval for the OF model predict, because the model is too large.
        )

        with torch.no_grad():
            for i in tqdm(range(0, f, interval), desc="Predict optical flow"):

                # transforms
                current_frame_batch, next_frame_batch = self.transforms(
                    current_frame[i : i + interval], next_frame[i : i + interval]
                )

                temp_pred_flows = self.model(
                    current_frame_batch,
                    next_frame_batch,
                )[-1]
                pred_flows.append(temp_pred_flows.cpu())

        # empty cache
        torch.cuda.empty_cache()
        del frame_batch

        return torch.cat(pred_flows, dim=0)  # f, c, h, w

    def save_image(self, flow: torch.Tensor, video_path: Path):
        """
        Save the optical flow image to the specified path.
        """
        person = video_path.parts[-2]
        video_name = video_path.stem

        _save_path = Path(self.save_path) / "vis" / "img" / "optical_flow" / person / video_name
        if not _save_path.exists():
            _save_path.mkdir(parents=True, exist_ok=True)

        f, c, h, w = flow.shape

        for i in tqdm(range(f), desc="Save optical flow", leave=False):

            flow_img = flow_to_image(flow[i])
            write_png(flow_img.cpu(), str(_save_path / f"{i}_flow.png"))

    def __call__(self, frames: torch.Tensor, video_path: Path) -> torch.Tensor:
        """
        predict one batch optical flow.

        Args:
            batch (nn.Tensor): batches of videos. (b, c, f, h, w)

        Returns:
            nn.Tensor: stacked predict optical flow, (b, 2, f, h, w)
        """

        _pred_flow = self.get_Optical_flow(frames)  # f, c, h, w

        if self.save:
            self.save_image(_pred_flow, video_path)

            merge_frame_to_video(
                self.save_path, video_path.parts[-2], video_path.stem, "optical_flow"
            )

        return _pred_flow.permute(0, 2, 3, 1)
