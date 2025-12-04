#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/sam3d_body/sam3d.py
Project: /workspace/code/sam3d_body
Created Date: Thursday December 4th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday December 4th 2025 4:24:51 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import os
from glob import glob
from typing import Optional, List

import cv2
import numpy as np
import torch
from tqdm import tqdm

from omegaconf.omegaconf import DictConfig
import os
from glob import glob
from typing import Optional, List, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from sam3d_body.tools.vis_utils import visualize_sample_together


class SAM3DBodyPipeline:
    """
    一个简单的 SAM-3D-Body 推理封装类：
    - 初始化时加载 sam-3d-body 主模型 + 可选 detector / segmentor / fov
    - 提供 process_image / process_folder 接口
    """

    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        """
        Args:
            checkpoint_path: sam-3d-body 主模型 ckpt 路径
            detector_name: 检测器名称（为空字符串则不启用）
            segmentor_name: 分割器名称（为空字符串则不启用）
            fov_name: FOV 估计器名称（为空字符串则不启用）
            mhr_path, detector_path, segmentor_path, fov_path:
                对应模块的本地权重路径；若为空，则会从环境变量
                SAM3D_MHR_PATH / SAM3D_DETECTOR_PATH / SAM3D_SEGMENTOR_PATH / SAM3D_FOV_PATH 中读取
            device: torch.device；None 时自动选择 cuda / cpu
        """

        self.device = cfg.infer.gpu

        # 如果参数为空，则从环境变量中读取
        self.mhr_path = cfg.model.get("mhr_path", "") or os.environ.get(
            "SAM3D_MHR_PATH", ""
        )
        self.detector_name = cfg.model.get("detector_name", "vitdet")
        self.segmentor_name = cfg.model.get("segmentor_name", "")
        self.fov_name = cfg.model.get("fov_name", "moge2")
        self.detector_path = cfg.model.get("detector_path", "") or os.environ.get(
            "SAM3D_DETECTOR_PATH", ""
        )
        self.segmentor_path = cfg.model.get("segmentor_path", "") or os.environ.get(
            "SAM3D_SEGMENTOR_PATH", ""
        )
        self.fov_path = cfg.model.get("fov_path", "") or os.environ.get(
            "SAM3D_FOV_PATH", ""
        )
        # -------------------- 初始化主模型 -------------------- #
        sam3d_model, model_cfg = load_sam_3d_body(
            cfg.model.checkpoint_path,
            device=self.device,
            mhr_path=self.mhr_path,
        )

        # -------------------- 可选模块：detector / segmentor / fov -------------------- #
        human_detector = None
        human_segmentor = None
        fov_estimator = None

        if self.detector_name:
            from sam3d_body.tools.build_detector import HumanDetector

            human_detector = HumanDetector(
                name=self.detector_name,
                device=self.device,
                path=self.detector_path,
            )

        if self.segmentor_name:
            from sam3d_body.tools.build_sam import HumanSegmentor

            human_segmentor = HumanSegmentor(
                name=self.segmentor_name,
                device=self.device,
                path=self.segmentor_path,
            )

        if self.fov_name:
            from sam3d_body.tools.build_fov_estimator import FOVEstimator

            fov_estimator = FOVEstimator(
                name=self.fov_name,
                device=self.device,
                path=self.fov_path,
            )

        # 挂到成员变量上
        self.estimator = SAM3DBodyEstimator(
            sam_3d_body_model=sam3d_model,
            model_cfg=model_cfg,
            human_detector=human_detector,
            human_segmentor=human_segmentor,
            fov_estimator=fov_estimator,
        )

    # ------------------------------------------------------------------ #
    # 基础接口
    # ------------------------------------------------------------------ #
    def process_image(
        self,
        image: np.ndarray,
        *,
        bbox_thresh: float = 0.8,
        use_mask: bool = False,
        return_outputs: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, dict]:
        """
        对单张图片跑 sam-3d-body 推理并返回可视化结果。

        Args:
            image_path: 图片路径
            bbox_thresh: 检测阈值
            use_mask: 是否使用 mask-conditioned 预测
            return_outputs: 是否同时返回原始 outputs 字典

        Returns:
            vis_img 或 (vis_img, outputs)
        """
        outputs = self.estimator.process_one_image(
            image,
            bbox_thr=bbox_thresh,
            use_mask=use_mask,
        )

        img = image
        vis_img = visualize_sample_together(img, outputs, self.estimator.faces)

        if return_outputs:
            return vis_img, outputs
        return vis_img

