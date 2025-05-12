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

from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image


class DepthEstimator:
    def __init__(self, configs):

        model_name = configs.model
        self.device = configs.device

        self.processor = DPTImageProcessor.from_pretrained(model_name)
        self.model = DPTForDepthEstimation.from_pretrained(model_name).to(configs.device)

    def estimate_depth(self, frame_batch: np.ndarray):
        
        t, h, w, c = frame_batch.shape

        one_batch_depth = [] 

        for frame in frame_batch:
            
            # prepare image for the model
            inputs = self.processor(images=frame, return_tensors="pt").to(self.device)

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

            one_batch_depth.append(prediction) # t, d, h, w

            # visualize the prediction
            output = prediction.squeeze().cpu().numpy()
            formatted = (output * 255 / np.max(output)).astype("uint8")
            depth = Image.fromarray(formatted)

        return depth

    def process_batch(self, batch: torch.Tensor):

        # batch is a tensor of shape (b, c, t, h, w)
        b, c, t, h, w = batch.shape

        for batch_idx in range(b):
            
            # c, t, h, w > t, h, w, c
            one_batch_numpy = (
                batch[batch_idx, [2, 1, 0], ...]
                .permute(1, 2, 3, 0)
                .to(torch.uint8)
                .numpy()
            )
            
            assert one_batch_numpy.shape == (t, h, w, c)
            assert one_batch_numpy.dtype == np.uint8

            # estimate depth for each image
            depths = self.estimate_depth(one_batch_numpy)

        return depths
    
    def __call__(self, batch: torch.Tensor):

        b, c, t, h, w = batch.shape

        depth = self.process_batch(batch)

        return depth
