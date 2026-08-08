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


def recover_pose_from_multiE(E, pts1, pts2, K):
    # 规范化输入
    pts1 = np.asarray(pts1, np.float64).reshape(-1, 2)
    pts2 = np.asarray(pts2, np.float64).reshape(-1, 2)
    assert len(pts1) >= 5 and len(pts1) == len(pts2)
    assert K.shape == (3, 3)

    # 把 (3N,3) 或 (N,9) 变成 (N,3,3)
    E = np.asarray(E)
    if E.shape == (3, 3):
        E_list = [E]
    elif E.ndim == 2 and E.shape[1] == 3 and E.shape[0] % 3 == 0:
        E_list = [E[i : i + 3, :] for i in range(0, E.shape[0], 3)]
    elif E.ndim == 2 and E.shape[1] == 9:
        E_list = [E[i].reshape(3, 3) for i in range(E.shape[0])]
    else:
        raise ValueError(f"Unexpected E shape: {E.shape}")

    best = dict(inliers=-1, R=None, t=None, mask=None, E=None)

    for Ei in E_list:
        if not np.isfinite(Ei).all():  # 排除 NaN/Inf
            continue
        # 使用 recoverPose 评估该候选
        inliers, R, t, mask = cv2.recoverPose(Ei, pts1, pts2, K)
        if inliers > best["inliers"]:
            best = dict(inliers=inliers, R=R, t=t, mask=mask, E=Ei)

    if best["inliers"] <= 0 or best["R"] is None:
        raise RuntimeError("No valid essential matrix candidate passed recoverPose")

    return best["R"], best["t"], best["mask"], best["E"]


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


# ---------- 姿态估计 ----------
def estimate_camera_pose_from_kpt(pts1, pts2, K, baseline_m):
    """从关键点估计相机姿态（R, t）
    这里的pts1是左边的视角，pts2是右边的视角

    Args:
        pts1 (np.ndarray): 第一帧关键点 (N, 2)
        pts2 (np.ndarray): 第二帧关键点 (N, 2)
        K (np.ndarray): 相机内参矩阵 (3, 3)

    Returns:
        R (np.ndarray): 旋转矩阵 (3, 3)
        t (np.ndarray): 平移向量 (3, 1)
        mask_pose (np.ndarray): 有效匹配掩码
    """

    E, mask = cv2.findEssentialMat(
        pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    if E is None:
        return None, None, None
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
    T = (t / np.linalg.norm(t)) * float(baseline_m)

    # check
    C2 = -R.T @ T
    assert np.isclose(
        np.linalg.norm(C2), baseline_m, rtol=1e-6, atol=1e-9
    ), "baseline check failed"

    return R, T.reshape(3, 1), mask_pose


def estimate_camera_pose_from_SIFT(img1, img2, K, baseline_m, ratio_thresh=0.75):
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

    img1 = to_gray_cv_image(img1)
    img2 = to_gray_cv_image(img2)

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

    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

    T = (t / np.linalg.norm(t)) * float(baseline_m)

    # check
    C2 = -R.T @ T
    assert np.isclose(
        np.linalg.norm(C2), baseline_m, rtol=1e-6, atol=1e-9
    ), "baseline check failed"

    return R, T.reshape(3, 1), pts1, pts2, mask_pose


def estimate_camera_pose_from_ORB(img1, img2, K, baseline_m, max_matches=1000):
    """
    估计相邻帧之间的相机相对位姿（R, t方向, 欧拉角）

    参数:
        img1, img2 : ndarray
            两帧灰度图像 (H, W)。
            两帧灰度图像 (H, W)。
        K : ndarray (3x3)
            相机内参矩阵。
        max_matches : int
            用于估计的最大匹配数量，默认 1000。

    返回:
        R : ndarray (3x3)
            从 img1 到 img2 的旋转矩阵。
        t_dir : ndarray (3,)
            单位化的平移方向向量（无尺度）。
        euler : tuple (yaw, pitch, roll) [degrees]
            从 R 转换得到的欧拉角（ZYX顺序）。
    """
    img1 = to_gray_cv_image(img1)
    img2 = to_gray_cv_image(img2)
    # --- Step1: 特征检测和描述 ---
    orb = cv2.ORB_create(3000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # --- Step2: 暴力匹配（汉明距离，交叉验证） ---
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:max_matches]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # --- Step3: 本质矩阵 + R,t ---
    E, inliers = cv2.findEssentialMat(
        pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    inlier_mask = inliers.ravel() == 1
    pts1, pts2 = pts1[inlier_mask], pts2[inlier_mask]

    _, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, K)

    # --- Step4: 计算欧拉角 (ZYX顺序: yaw,pitch,roll) ---
    # yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    # pitch = np.degrees(np.arcsin(-R[2, 0]))
    # roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))

    T = (t / np.linalg.norm(t)) * float(baseline_m)

    # check
    C2 = -R.T @ T
    assert np.isclose(
        np.linalg.norm(C2), baseline_m, rtol=1e-6, atol=1e-9
    ), "baseline check failed"

    return R, T.reshape(3, 1), pose_mask


def estimate_pose_from_bbox_region(imgL, imgR, bboxL, bboxR, K, baseline_m):
    """
    从左右图像的 bbox 区域中估计相机姿态
    bbox: [x1, y1, x2, y2]
    """

    # 1. 裁剪区域
    patchL = imgL[int(bboxL[1]) : int(bboxL[3]), int(bboxL[0]) : int(bboxL[2])]
    patchR = imgR[int(bboxR[1]) : int(bboxR[3]), int(bboxR[0]) : int(bboxR[2])]

    patchL = to_gray_cv_image(patchL)
    patchR = to_gray_cv_image(patchR)

    # 2. SIFT 特征提取
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(patchL, None)
    kp2, des2 = sift.detectAndCompute(patchR, None)

    if des1 is None or des2 is None:
        return np.eye(3), np.zeros((3, 1)), None # 无特征，返回默认值

    # 3. 匹配
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)
    good = [
        m[0] for m in matches if len(m) == 2 and m[0].distance < 0.75 * m[1].distance
    ]

    if len(good) < 5:
        return np.eye(3), np.zeros((3, 1)), None # 匹配点太少，返回默认值

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    # 4. 坐标偏移修正（恢复到整幅图坐标）
    pts1[:, 0] += bboxL[0]
    pts1[:, 1] += bboxL[1]
    pts2[:, 0] += bboxR[0]
    pts2[:, 1] += bboxR[1]

    # 5. 估计姿态
    E, mask = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, 0.999, 1.0)
    if E is None:
        return None, None, None

    R, t, mask_pose, _ = recover_pose_from_multiE(E, pts1, pts2, K)

    T = (t / np.linalg.norm(t)) * float(baseline_m)

    # check
    C2 = -R.T @ T
    assert np.isclose(
        np.linalg.norm(C2), baseline_m, rtol=1e-6, atol=1e-9
    ), "baseline check failed"

    return R, T.reshape(3, 1), mask_pose
