#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/kpt_generation/camera_position.py
Project: /workspace/code/kpt_generation
Created Date: Tuesday August 5th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Tuesday August 5th 2025 10:11:30 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""
import numpy as np
import cv2
import torch


# ---------- 姿态估计 ----------
def estimate_camera_pose_from_kpt(pts1, pts2, K):
    E, mask = cv2.findEssentialMat(
        pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    if E is None:
        return None, None, None
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
    return R, t, mask_pose


def to_gray_cv_image(tensor_img):
    """
    将 torch 格式的 HWC 图像（0~1 或 0~255）转换为 OpenCV 灰度图像（uint8）

    参数:
        tensor_img: torch.Tensor, shape=(H, W, C), dtype=torch.float32 或 torch.uint8

    返回:
        img_gray: np.ndarray, shape=(H, W), dtype=uint8，适用于 OpenCV
    """
    if tensor_img.dim() != 3 or tensor_img.shape[2] != 3:
        raise ValueError("输入必须是 (H, W, 3) 的 RGB 图像")

    # 如果是 float，转换到 [0, 255]
    if tensor_img.dtype == torch.float32:
        img_np = tensor_img.clamp(0, 1).mul(255).byte().cpu().numpy()
    elif tensor_img.dtype == torch.uint8:
        img_np = tensor_img.cpu().numpy()
    else:
        raise TypeError("只支持 float32 或 uint8 类型的输入")

    # 转灰度（OpenCV 格式）
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    return img_gray


def estimate_camera_pose_from_sift_imgs(img1, img2, K, ratio_thresh=0.75):
    """
    使用 SIFT 特征从两幅图像中估计相机相对姿态（R, T）

    参数:
        img1: np.ndarray, 图像1（灰度图）
        img2: np.ndarray, 图像2（灰度图）
        K: np.ndarray, 相机内参矩阵 (3x3)
        ratio_thresh: float, Lowe's ratio 阈值（默认0.75）

    返回:
        R: np.ndarray, 旋转矩阵 (3x3)
        T: np.ndarray, 平移向量 (3x1)
        pts1: np.ndarray, 匹配点（图像1） Nx2
        pts2: np.ndarray, 匹配点（图像2） Nx2
        mask_pose: np.ndarray, 有效匹配掩码
    """

    if img1 is None or img2 is None or len(img1.shape) != 2 or len(img2.shape) != 2:
        raise ValueError("输入图像必须是灰度图，并且不能为空")

    # Step 1: 提取 SIFT 特征
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Step 2: 匹配特征
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Step 3: Lowe’s ratio test
    pts1, pts2 = [], []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    if len(pts1) < 8:
        raise ValueError(f"匹配点太少（仅有{len(pts1)}个），无法估计姿态。")

    pts1 = np.array(pts1, dtype=np.float32)
    pts2 = np.array(pts2, dtype=np.float32)

    # Step 4: 估计本质矩阵与姿态
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, threshold=1.0)
    if E is None:
        raise ValueError("Essential matrix 计算失败。")

    _, R, T, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

    return R, T, pts1, pts2, mask_pose
