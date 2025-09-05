#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/code/triangulation/camera_position/SIFT_kpt.py
Project: /workspace/code/triangulation/camera_position
Created Date: Friday September 5th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Friday September 5th 2025 5:02:54 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''
#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import annotations
import numpy as np
import cv2
from typing import Optional, Tuple

# ---------- 工具 ----------

def _to_gray(img):
    # 支持 numpy/torch，RGB/灰度；输出 OpenCV 灰度 uint8
    import torch
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    img = np.ascontiguousarray(img)
    if img.ndim == 3 and img.shape[2] == 3:
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if img.ndim == 2:
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        return img
    raise ValueError("img must be HxW or HxWx3")

def _normalize(K: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """像素 -> 归一化平面 (x,y)，K 为 3x3。"""
    uv1 = np.hstack([uv, np.ones((uv.shape[0], 1), dtype=uv.dtype)])
    x = (np.linalg.inv(K) @ uv1.T).T
    x = x[:, :2] / x[:, 2:3]
    return x

def _sift_matches(img1_gray: np.ndarray, img2_gray: np.ndarray, ratio: float = 0.75):
    sift = cv2.SIFT_create()
    k1, d1 = sift.detectAndCompute(img1_gray, None)
    k2, d2 = sift.detectAndCompute(img2_gray, None)
    if d1 is None or d2 is None:
        return np.zeros((0,2), np.float64), np.zeros((0,2), np.float64)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw = bf.knnMatch(d1, d2, k=2)
    good = [m for m, n in raw if m.distance < ratio * n.distance]
    pts1 = np.float64([k1[m.queryIdx].pt for m in good])
    pts2 = np.float64([k2[m.trainIdx].pt for m in good])
    return pts1, pts2

def _stack_kpt(pts: Optional[np.ndarray], scores: Optional[np.ndarray], th: float):
    """筛选 kpt (N,2) by score (N,)；允许 None。返回 (M,2)。"""
    if pts is None:
        return np.zeros((0,2), dtype=np.float64)
    pts = np.asarray(pts, dtype=np.float64)
    if scores is not None:
        scores = np.asarray(scores).reshape(-1)
        m = np.isfinite(pts).all(1) & np.isfinite(scores) & (scores >= th)
        return pts[m]
    else:
        m = np.isfinite(pts).all(1)
        return pts[m]

# ---------- 主函数 ----------

def estimate_camera_pose_hybrid(
    img1, img2, K: np.ndarray,
    *,
    # 人体关键点（像素坐标）
    kpt1: Optional[np.ndarray] = None,   # (J,2) 或 None
    kpt2: Optional[np.ndarray] = None,
    kpt_score1: Optional[np.ndarray] = None,  # (J,) 0~1
    kpt_score2: Optional[np.ndarray] = None,
    score_thresh: float = 0.30,          # kpt 置信度阈值
    sift_ratio: float = 0.75,            # Lowe ratio
    magsac_thresh: float = 1e-3,         # 归一化平面上的 MAGSAC 阈值
    kpt_boost: int = 2,                  # 提高 kpt 在 RANSAC 采样中的存在感（重复次数）
    refine: bool = True,                 # 是否做非线性微调
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    返回:
      R: (3,3) 旋转矩阵（单位范数）
      t: (3,)  平移方向（单位范数，尺度不定）
      info: 统计信息/中间结果
    """
    # 1) SIFT 背景匹配
    g1 = _to_gray(img1); g2 = _to_gray(img2)
    sift1, sift2 = _sift_matches(g1, g2, ratio=sift_ratio)

    # 2) 汇入人体 kpt（取两路共同有效 & 置信度过滤）
    if kpt1 is not None and kpt2 is not None:
        k1 = _stack_kpt(kpt1, kpt_score1, score_thresh)
        k2 = _stack_kpt(kpt2, kpt_score2, score_thresh)
        # 对齐长度（如果你传入的是同一顺序的关键点，直接逐元素对齐；否则请先对齐）
        M = min(len(k1), len(k2))
        k1, k2 = k1[:M], k2[:M]
    else:
        k1 = np.zeros((0,2), np.float64); k2 = np.zeros((0,2), np.float64)

    # 3) 合并对应（可以通过重复 kpt 提升其被 RANSAC 采样概率）
    if kpt_boost > 1 and len(k1) > 0:
        k1_rep = np.repeat(k1, kpt_boost, axis=0)
        k2_rep = np.repeat(k2, kpt_boost, axis=0)
    else:
        k1_rep, k2_rep = k1, k2

    uv1 = np.vstack([sift1, k1_rep]).astype(np.float64)
    uv2 = np.vstack([sift2, k2_rep]).astype(np.float64)

    if len(uv1) < 8:
        raise ValueError(f"有效对应太少：{len(uv1)}")

    # 4) 归一化到单位焦距坐标（更稳的 E 估计）
    x1 = _normalize(K, uv1)
    x2 = _normalize(K, uv2)

    # 5) USAC_MAGSAC 估 Essential
    E, inl = cv2.findEssentialMat(
        x1, x2, focal=1.0, pp=(0,0),
        method=cv2.USAC_MAGSAC, threshold=magsac_thresh, prob=0.9999, maxIters=20000
    )
    if E is None:
        raise RuntimeError("findEssentialMat 失败")
    inlier_mask = (inl.ravel() > 0)

    # 6) recoverPose
    _, R, t, inl2 = cv2.recoverPose(E, x1, x2, focal=1.0, pp=(0,0), mask=inl)
    t = t.reshape(3)
    # 统一方向（让大多数点在两相机前方，可选）
    # —— 简化起见此处略过，recoverPose 已做可视深度约束

    info = dict(
        n_sift=len(sift1),
        n_kpt=len(k1),
        n_all=len(uv1),
        n_inlier=int(inlier_mask.sum()),
        inlier_mask=inlier_mask,
        used_pts1=uv1,
        used_pts2=uv2,
    )

    # 7) 可选：用 Sampson 残差 + Huber 微调 (R,t)
    if refine:
        try:
            from scipy.optimize import least_squares
        except Exception:
            return R, t, info  # 没有 SciPy 就直接返回

        def rodrigues(r):
            Rm, _ = cv2.Rodrigues(r.reshape(3,1))
            return Rm

        def residuals(p):
            r = p[:3]; tt = p[3:6]
            tt = tt / (np.linalg.norm(tt) + 1e-12)
            Rm = rodrigues(r)
            # Essential E = [t]_x R
            tx = np.array([[0, -tt[2], tt[1]],
                           [tt[2], 0, -tt[0]],
                           [-tt[1], tt[0], 0]], dtype=np.float64)
            E_ = tx @ Rm
            # Sampson 距离
            x1h = np.hstack([x1, np.ones((x1.shape[0],1))])
            x2h = np.hstack([x2, np.ones((x2.shape[0],1))])
            Ex1 = (E_ @ x1h.T).T     # N x 3
            Etx2= (E_.T @ x2h.T).T   # N x 3
            x2tEx1 = np.sum(x2h * (E_ @ x1h.T).T, axis=1)  # N
            denom = Ex1[:,0]**2 + Ex1[:,1]**2 + Etx2[:,0]**2 + Etx2[:,1]**2 + 1e-12
            s = x2tEx1 / np.sqrt(denom)

            # kpt 提高权重（未重复时可在此乘权重；这里我们已通过重复采样增强了影响）
            # 但仍可对原始 kpt 索引区间给更高 w：
            N_sift = len(sift1)
            N_rep  = len(k1_rep)
            w = np.ones_like(s)
            if N_rep > 0:
                w[N_sift:] = 1.5  # 让 kpt 残差稍微更受重视

            # Huber（f_scale=1.0）
            delta = 1.0
            a = np.abs(s)
            huber = np.where(a <= delta, s, np.sign(s)*(delta + (a - delta)*0.1))
            return w * huber

        # 参数化：rvec(3)+t(3)
        r0, _ = cv2.Rodrigues(R)
        p0 = np.hstack([r0.ravel(), t])
        sol = least_squares(residuals, p0, method="trf", loss="huber", f_scale=1.0,
                            max_nfev=100, xtol=1e-8, ftol=1e-8, gtol=1e-8, verbose=0)
        r_opt = sol.x[:3]; t_opt = sol.x[3:6]
        R_opt, _ = cv2.Rodrigues(r_opt.reshape(3,1))
        t_opt = t_opt / (np.linalg.norm(t_opt) + 1e-12)
        info["refine_cost"] = float(sol.cost)
        return R_opt, t_opt, info

    return R, t, info
