#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/prepare_dataset/depth_estimation.py
Project: /workspace/code/prepare_dataset
Created Date: Monday May 12th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Monday May 12th 2025 6:43:43 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from pathlib import Path
from tqdm import tqdm

from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image

from prepare_dataset.utils import merge_frame_to_video


class DepthEstimator:
    def __init__(self, configs):

        model_name = configs.depth_estimator.model

        self.device = configs.device

        self.processor = DPTImageProcessor.from_pretrained(
            model_name, cache_dir=configs.depth_estimator.dpt_processor_path
        )
        self.model = DPTForDepthEstimation.from_pretrained(
            model_name, cache_dir=configs.depth_estimator.dpt_model_path
        ).to(self.device)

        self.save = configs.depth_estimator.save
        self.save_path = Path(configs.extract_dataset.save_path)

    def estimate_depth(self, frames: np.ndarray):

        t, h, w, c = frames.shape

        # prepare image for the model
        inputs = self.processor(images=frames, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )

        return prediction

    def save_image(self, image: torch.Tensor, i: int, video_path: Path):
        """
        Save the image to the specified path.
        """

        person = video_path.parts[-2]
        video_name = video_path.stem

        _save_path = self.save_path / "vis" / "img" / "depth" / person / video_name
        if not _save_path.exists():
            _save_path.mkdir(parents=True, exist_ok=True)

        # visualize the prediction

        output = image.squeeze().cpu().numpy()
        formatted = (output * 255 / np.max(output)).astype("uint8")
        depth = Image.fromarray(formatted)

        depth.save(_save_path / f"{i}_depth.png")

    def __call__(self, vframes: torch.Tensor, video_path: Path):

        t, h, w, c = vframes.shape

        res_depth = []

        for i in tqdm(range(t), desc="Depth Estimation", leave=False):

            vframes_numpy = vframes[i].unsqueeze(0).numpy()

            depths = self.estimate_depth(vframes_numpy)

            if self.save:
                self.save_image(depths, i, video_path)

            res_depth.append(depths.cpu())

        if self.save:
            merge_frame_to_video(
                self.save_path, video_path.parts[-2], video_path.stem, "depth"
            )

        return torch.cat(res_depth, dim=0)
