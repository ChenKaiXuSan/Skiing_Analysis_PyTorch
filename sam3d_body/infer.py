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

import logging
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch
from omegaconf.omegaconf import DictConfig
from tqdm import tqdm

from .load import load_info
from .sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
from .save import save_mesh_results
from .vis import (
    display_results_grid,
    setup_visualizer,
    visualize_2d_results,
    visualize_3d_mesh,
)

logger = logging.getLogger(__name__)


def setup_sam_3d_body(
    cfg: DictConfig,
):
    # 如果参数为空，则从环境变量中读取
    mhr_path = cfg.model.get("mhr_path", "") or os.environ.get("SAM3D_MHR_PATH", "")
    # helper params
    detector_name = cfg.model.get("detector_name", "vitdet")
    segmentor_name = cfg.model.get("segmentor_name", "")
    fov_name = cfg.model.get("fov_name", "moge2")
    detector_path = cfg.model.get("detector_path", "") or os.environ.get(
        "SAM3D_DETECTOR_PATH", ""
    )
    segmentor_path = cfg.model.get("segmentor_path", "") or os.environ.get(
        "SAM3D_SEGMENTOR_PATH", ""
    )
    fov_path = cfg.model.get("fov_path", "") or os.environ.get("SAM3D_FOV_PATH", "")

    # -------------------- 初始化主模型 -------------------- #
    sam3d_model, model_cfg = load_sam_3d_body(
        cfg.model.checkpoint_path,
        device=cfg.infer.gpu,
        mhr_path=mhr_path,
    )

    # -------------------- 可选模块：detector / segmentor / fov -------------------- #
    human_detector = None
    human_segmentor = None
    fov_estimator = None

    if detector_name:
        from .tools.build_detector import HumanDetector

        human_detector = HumanDetector(
            name=detector_name,
            device=cfg.infer.gpu,
            path=detector_path,
        )

    if segmentor_name:
        from .tools.build_sam import HumanSegmentor

        human_segmentor = HumanSegmentor(
            name=segmentor_name,
            device=cfg.infer.gpu,
            path=segmentor_path,
        )

    if fov_name:
        from .tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(
            name=fov_name,
            device=cfg.infer.gpu,
            path=fov_path,
        )

    # 挂到成员变量上
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=sam3d_model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )

    logger.info("==== SAM 3D Body Estimator Setup ====")
    logger.info(f"  Model checkpoint: {cfg.model.checkpoint_path}")
    logger.info(f"  MHR model path: {mhr_path if mhr_path else 'Default'}")
    logger.info(
        f"  Human detector: {'✓' if human_detector else '✗ (will use full image or manual bbox)'}"
    )
    logger.info(
        f"  Human segmentor: {'✓' if human_segmentor else '✗ (mask inference disabled)'}"
    )
    logger.info(
        f"  FOV estimator: {'✓' if fov_estimator else '✗ (will use default FOV)'}"
    )
    return estimator


# ------------------------------------------------------------------ #
# 高级接口（处理文件夹等）
# ------------------------------------------------------------------ #
def process_one_video(
    video_path: Path,
    pt_path: Path,
    out_dir: Path,
    cfg: DictConfig,
):
    """处理单个视频文件的镜头编辑。"""

    out_dir.mkdir(parents=True, exist_ok=True)

    _, _, bbox_xyxy, bbox_scores, frames = load_info(
        pt_file_path=pt_path, video_file_path=video_path
    )

    # 初始化模型与可视化器
    estimator = setup_sam_3d_body(cfg)
    visualizer = setup_visualizer()

    for idx in tqdm(range(0, frames.shape[0]), desc="Processing frames"):
        # if idx > 1:
        #     break
        outputs = estimator.process_one_image(
            image=frames[idx],
            bboxes=bbox_xyxy[idx],
        )

        # 2D 结果可视化
        # vis_results = visualize_2d_results(frames[idx], outputs, visualizer)

        # # Display results using grid function
        # titles = [f"Person {i} - 2D Keypoints & BBox" for i in range(len(vis_results))]
        # display_results_grid(vis_results, titles, figsize_per_image=(6, 6))

        # 3D 网格可视化
        mesh_results = visualize_3d_mesh(frames[idx], outputs, estimator.faces)

        # Display results
        for i, combined_img in enumerate(mesh_results):
            combined_rgb = cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)

            plt.figure(figsize=(20, 5))
            plt.imshow(combined_rgb)
            plt.title(f"Person {i}: Original | Mesh Overlay | Front View | Side View")
            plt.axis("off")

        plt.savefig(out_dir / f"frame_{idx:04d}_3d_mesh_visualization.png")

        # save mesh results to files
        save_mesh_results(
            img_cv2=frames[idx],
            outputs=outputs,
            faces=estimator.faces,
            save_dir=str(out_dir / f"frame_{idx:04d}_meshes"),
            image_name=f"frame_{idx:04d}",
        )

    # final
    torch.cuda.empty_cache()
    del estimator
    del visualizer

    return out_dir
