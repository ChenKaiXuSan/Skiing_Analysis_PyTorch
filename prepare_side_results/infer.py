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
import numpy as np
import torch
from omegaconf.omegaconf import DictConfig
from tqdm import tqdm

from .load import load_info
from .sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
from .sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info
from .sam_3d_body.visualization.renderer import Renderer
from .sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
from .save import save_mesh_results, save_results
from .tools.vis_utils import visualize_sample, visualize_sample_together
from .vis import (
    display_results_grid,
    visualize_2d_results,
    visualize_3d_mesh,
)

logger = logging.getLogger(__name__)


def select_closest_person(outputs, previous_person=None):
    """Select the closest person using camera depth, orientation, and frame continuity.
    
    Args:
        outputs: List of detected persons
        previous_person: Person info from previous frame (for continuity check)
    """
    if not outputs:
        return outputs

    # Strategy 1: Use camera depth to find closest person
    cam_candidates = []
    for i, out in enumerate(outputs):
        cam_t = out.get("pred_cam_t")
        if cam_t is None:
            continue
        cam_t = np.asarray(cam_t).reshape(-1)
        if cam_t.size >= 3 and np.isfinite(cam_t[2]):
            cam_candidates.append((float(cam_t[2]), i))

    if cam_candidates:
        # Get closest person by camera depth
        cam_candidates.sort(key=lambda x: x[0])
        closest_depth, closest_idx = cam_candidates[0]
        closest_out = outputs[closest_idx]
        
        # If we have a previous frame, try to find continuous person
        if previous_person is not None:
            prev_cam_t = previous_person.get("pred_cam_t")
            prev_rot = previous_person.get("pred_global_rots")
            
            if prev_cam_t is not None and prev_rot is not None:
                prev_cam_t = np.asarray(prev_cam_t).reshape(-1)
                prev_rot = np.asarray(prev_rot)
                
                # Handle different rotation matrix shapes
                # If shape is (J, 3, 3), take the root (first) joint
                # If shape is (3, 3), use directly
                if prev_rot.ndim == 3 and prev_rot.shape[0] > 1:
                    prev_rot = prev_rot[0]  # Take root joint rotation
                elif prev_rot.ndim > 2:
                    prev_rot = prev_rot.reshape(3, 3)
                
                if prev_rot.shape != (3, 3):
                    # Skip continuity check if rotation shape is invalid
                    return [outputs[closest_idx]]
                
                prev_forward = prev_rot[:, 2]
                
                # Check all candidates for continuity
                best_continuity_idx = -1
                best_continuity_score = -1.0
                
                for depth, idx in cam_candidates:
                    curr_out = outputs[idx]
                    curr_rot = curr_out.get("pred_global_rots")
                    
                    if curr_rot is None:
                        continue
                    
                    curr_rot = np.asarray(curr_rot)
                    
                    # Handle different rotation matrix shapes
                    if curr_rot.ndim == 3 and curr_rot.shape[0] > 1:
                        curr_rot = curr_rot[0]  # Take root joint rotation
                    elif curr_rot.ndim > 2:
                        try:
                            curr_rot = curr_rot.reshape(3, 3)
                        except ValueError:
                            continue  # Skip if reshape fails
                    
                    if curr_rot.shape != (3, 3):
                        continue
                    
                    curr_forward = curr_rot[:, 2]
                    
                    # Compute continuity score: 
                    # - Camera depth change (prefer small change)
                    # - Orientation similarity (prefer high similarity)
                    depth_ratio = depth / (float(prev_cam_t[2]) + 1e-6)
                    depth_change = abs(depth_ratio - 1.0)  # 0-1, lower is better
                    
                    orientation_sim = np.dot(prev_forward, curr_forward)  # -1-1, higher is better
                    
                    # Combined score (weighted average)
                    # If depth changes < 20%, it's very likely the same person
                    # If orientation changes < 30 degrees (sim > 0.866), it's continuous
                    continuity_score = (1.0 - min(depth_change, 1.0) * 0.5) * 0.5 + orientation_sim * 0.5
                    
                    if continuity_score > best_continuity_score:
                        best_continuity_score = continuity_score
                        best_continuity_idx = idx
                
                # If we found a good match (score > 0.6), use it
                if best_continuity_score > 0.6:
                    return [outputs[best_continuity_idx]]
        
        # Otherwise, just return the closest person
        return [outputs[closest_idx]]

    # Strategy 2: Fallback to bbox area if no camera depth available
    bbox_candidates = []
    for i, out in enumerate(outputs):
        bbox = out.get("bbox")
        if bbox is None:
            continue
        bbox = np.asarray(bbox).reshape(-1)
        if bbox.size >= 4:
            area = max(0.0, float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))
            bbox_candidates.append((area, i))

    if bbox_candidates:
        _, best_idx = max(bbox_candidates, key=lambda x: x[0])
        return [outputs[best_idx]]

    return [outputs[0]]


def setup_visualizer():
    """Set up skeleton visualizer with MHR70 pose info"""
    visualizer = SkeletonVisualizer(line_width=2, radius=5)
    visualizer.set_pose_meta(mhr70_pose_info)
    return visualizer


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
    inference_output_path: Path,
    cfg: DictConfig,
):
    """处理单个视频文件的镜头编辑。"""

    out_dir.mkdir(parents=True, exist_ok=True)
    inference_output_path.mkdir(parents=True, exist_ok=True)
    
    # 统一保存路径
    vis_dir = out_dir / "visualizations"  # 2D/3D可视化
    mesh_dir = out_dir / "meshes"  # 网格结果
    vis_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)

    # 读取 pt 信息
    _, _, bbox_xyxy, bbox_scores, frames = load_info(
        pt_file_path=pt_path, video_file_path=video_path
    )

    # 初始化模型与可视化器
    estimator = setup_sam_3d_body(cfg)
    visualizer = setup_visualizer()

    previous_person = None  # Track previous frame's selected person

    for idx in tqdm(range(0, frames.shape[0]), desc="Processing frames"):
        # if idx > 1:
        #     break

        if bbox_xyxy is not None:
            outputs = estimator.process_one_image(
                img=frames[idx],
                bboxes=bbox_xyxy[idx],
            )
        else:
            outputs = estimator.process_one_image(
                img=frames[idx],
                bboxes=None,
            )
        
        # 选取离摄像头最近的人作为运动员，同时考虑与上一帧的连续性
        outputs = select_closest_person(outputs, previous_person)
        
        frame_name = f"frame_{idx:04d}"
        
        # 2D 结果可视化
        vis_results = visualize_2d_results(frames[idx], outputs, visualizer)
        cv2.imwrite(str(vis_dir / f"{frame_name}_2d.png"), vis_results[0])

        # 3D 网格可视化
        mesh_results = visualize_3d_mesh(frames[idx], outputs, estimator.faces)
        for i, combined_img in enumerate(mesh_results):
            cv2.imwrite(
                str(vis_dir / f"{frame_name}_3d_mesh_{i}.png"), 
                combined_img
            )

        # 保存网格结果到文件
        save_mesh_results(
            img_cv2=frames[idx],
            outputs=outputs,
            faces=estimator.faces,
            save_dir=str(mesh_dir / frame_name),
            image_name=frame_name,
        )

        # 处理输出并保存
        selected_output = outputs[0]
        selected_output["frame"] = frames[idx]
        
        # 保存用于下一帧比较
        previous_person = selected_output
        
        # 按帧保存结果
        save_results(
            outputs=[selected_output],
            save_dir=inference_output_path / frame_name,
        )

    # 清理资源
    torch.cuda.empty_cache()
    del estimator
    del visualizer

    logger.info(f"✓ Video processing completed")
    logger.info(f"  Visualizations: {vis_dir}")
    logger.info(f"  Mesh results: {mesh_dir}")
    logger.info(f"  Inference outputs: {inference_output_path}")

    return out_dir
