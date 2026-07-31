#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/triangulation/camera_position/camera_position_kpt_bbox.py
Project: /workspace/code/triangulation/camera_position
Created Date: Wednesday October 8th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Wednesday October 8th 2025 9:15:13 pm
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
from typing import List, Tuple, Optional, Dict
from triangulation.camera_position.camera_position import to_gray_cv_image

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
        E_list = [E[i:i+3, :] for i in range(0, E.shape[0], 3)]
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

def _sift_match_patch_pair(imgL, imgR, bboxL, bboxR, ratio=0.75, max_kp=1000):
    """
    在给定的一对 bbox patch 内做 SIFT + FLANN 匹配，返回整幅图坐标的对应点 (M,2),(M,2) 及原始匹配距离。
    """
    x1L, y1L, x2L, y2L = map(int, bboxL)
    x1R, y1R, x2R, y2R = map(int, bboxR)
    patchL = to_gray_cv_image(imgL[y1L:y2L, x1L:x2L])
    patchR = to_gray_cv_image(imgR[y1R:y2R, x1R:x2R])

    if patchL.size == 0 or patchR.size == 0:
        return (
            np.empty((0, 2), np.float32),
            np.empty((0, 2), np.float32),
            np.empty((0,), np.float32),
        )

    sift = cv2.SIFT_create(nfeatures=max_kp)
    k1, d1 = sift.detectAndCompute(patchL, None)
    k2, d2 = sift.detectAndCompute(patchR, None)
    if d1 is None or d2 is None or len(k1) == 0 or len(k2) == 0:
        return (
            np.empty((0, 2), np.float32),
            np.empty((0, 2), np.float32),
            np.empty((0,), np.float32),
        )

    # FLANN（KDTree for SIFT）
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=64)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    knn = flann.knnMatch(d1, d2, k=2)

    good = []
    for m in knn:
        if len(m) == 2 and m[0].distance < ratio * m[1].distance:
            good.append(m[0])
    if len(good) < 5:
        return (
            np.empty((0, 2), np.float32),
            np.empty((0, 2), np.float32),
            np.empty((0,), np.float32),
        )

    ptsL = np.float32([k1[m.queryIdx].pt for m in good])
    ptsR = np.float32([k2[m.trainIdx].pt for m in good])
    # 回到整幅图坐标
    ptsL[:, 0] += x1L
    ptsL[:, 1] += y1L
    ptsR[:, 0] += x1R
    ptsR[:, 1] += y1R
    dists = np.float32([m.distance for m in good])  # 越小越好

    return ptsL, ptsR, dists


def _stack_and_weight(
    pts1_kpt: Optional[np.ndarray],
    pts2_kpt: Optional[np.ndarray],
    kpt_scores: Optional[np.ndarray],
    pts1_pix: Optional[np.ndarray],
    pts2_pix: Optional[np.ndarray],
    pix_dists: Optional[np.ndarray],
    kpt_weight: float = 1.0,
    pix_weight: float = 1.0,
    top_pix: Optional[int] = 500,
):
    """
    合并两路对应并生成权重。OpenCV 的 findEssentialMat 不直接接受权重，
    我们用“重复采样”近似：按权重把强匹配重复若干次（等价于加权内点投票）。
    """
    P1_list, P2_list = [], []

    def _repeat_by_w(P1, P2, w_arr, base=1.0):
        # 把权重归一化到 [1, ceil(3*base)] 的整数重复次数（稳妥做法，避免爆内存）
        if P1 is None or len(P1) == 0:
            return [], []
        w = np.asarray(w_arr, np.float32)
        w = w / (w.max() + 1e-8) * (3.0 * base)
        reps = np.clip(np.rint(w), 1, max(1, int(3 * base))).astype(int)
        out1, out2 = [], []
        for p1, p2, r in zip(P1, P2, reps):
            out1 += [p1] * int(r)
            out2 += [p2] * int(r)
        return out1, out2

    # kpt 权重（用 score * kpt_weight）
    if pts1_kpt is not None and pts2_kpt is not None and len(pts1_kpt) > 0:
        ks = (
            kpt_scores
            if (kpt_scores is not None and kpt_scores.size == len(pts1_kpt))
            else np.ones((len(pts1_kpt),), np.float32)
        )
        o1, o2 = _repeat_by_w(pts1_kpt, pts2_kpt, ks, base=kpt_weight)
        P1_list += o1
        P2_list += o2

    # 像素匹配权重（把距离转为分数，score = exp(-dist/σ)）
    if pts1_pix is not None and pts2_pix is not None and len(pts1_pix) > 0:
        if top_pix is not None and len(pts1_pix) > top_pix:
            # 先保留最好的 top_pix（距离最小）
            idx = np.argsort(pix_dists)[:top_pix]
            pts1_pix = pts1_pix[idx]
            pts2_pix = pts2_pix[idx]
            pix_dists = pix_dists[idx]
        sigma = np.median(pix_dists) + 1e-6
        pix_scores = np.exp(-pix_dists / sigma)
        o1, o2 = _repeat_by_w(pts1_pix, pts2_pix, pix_scores, base=pix_weight)
        P1_list += o1
        P2_list += o2

    if len(P1_list) < 5:
        return None, None

    P1 = np.asarray(P1_list, np.float32)
    P2 = np.asarray(P2_list, np.float32)
    return P1, P2


def estimate_pose_from_bbox_and_kpt(
    imgL: np.ndarray,
    imgR: np.ndarray,
    bboxesL: Tuple[float, float, float, float],
    bboxesR: Tuple[float, float, float, float],
    K: np.ndarray,
    baseline_m: float,
    *,
    kptsL: Optional[np.ndarray] = None,  # (K,2)
    kptsR: Optional[np.ndarray] = None,  # (K,2)
    kpt_scores: Optional[np.ndarray] = None,  # (K,)
    ratio: float = 0.75,  # SIFT ratio test
    max_kp_per_patch: int = 1000,
    kpt_weight: float = 1.5,  # kpt 的相对权重
    pix_weight: float = 1.0,  # 像素匹配的相对权重
    top_pix: Optional[int] = 800,  # 限制像素匹配数量
    ransac_prob: float = 0.999,
    ransac_thresh: float = 1.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Dict]:
    """
    融合 bbox 内像素匹配 + kpt 对应的相机姿态估计。
    bboxesL/bboxesR 需一一对应（同一目标）。
    返回: R(3,3), T(3,1), mask_pose(M,1), diagnostics(字典)
    """
    assert len(bboxesL) == len(bboxesR), "bboxesL / bboxesR 数量必须一致"

    # 1) 在每对 bbox 中做像素级特征匹配
    p1, p2, d = _sift_match_patch_pair(
        imgL, imgR, bboxesL, bboxesR, ratio=ratio, max_kp=max_kp_per_patch
    )

    # 2) 合并 kpt 对应（可选）
    if kptsL is not None and kptsR is not None:
        assert kptsL.shape == kptsR.shape and kptsL.shape[1] == 2
        pts1_kpt = kptsL.astype(np.float32)
        pts2_kpt = kptsR.astype(np.float32)
        ks = (
            kpt_scores.astype(np.float32)
            if (kpt_scores is not None and kpt_scores.shape[0] == kptsL.shape[0])
            else None
        )
    else:
        pts1_kpt = pts2_kpt = ks = None

    # 3) 统一加权合并（重复采样近似权重）
    P1, P2 = _stack_and_weight(
        pts1_kpt,
        pts2_kpt,
        ks,
        p1,
        p2,
        d,
        kpt_weight=kpt_weight,
        pix_weight=pix_weight,
        top_pix=top_pix,
    )
    if P1 is None:
        return None, None, None, dict(reason="insufficient_correspondences")

    # 4) 估计本质矩阵 + 恢复姿态
    E, mask = cv2.findEssentialMat(
        P1, P2, K, method=cv2.RANSAC, prob=ransac_prob, threshold=ransac_thresh
    )
    if E is None:
        return None, None, None, dict(reason="findEssentialMat_failed")

    inlier_ratio = float(mask.sum()) / float(len(mask) + 1e-12)
    # _, R, t, mask_pose = cv2.recoverPose(E, P1, P2, K)
    # 你已有的 E, pts1, pts2, K
    R, t, mask_pose, E_best = recover_pose_from_multiE(E, P1, P2, K)

    # 5) 基线定标 + 校验
    T = (t / (np.linalg.norm(t) + 1e-12)) * float(baseline_m)
    C2 = -R.T @ T
    baseline_ok = bool(np.isclose(np.linalg.norm(C2), baseline_m, rtol=1e-6, atol=1e-9))

    diagnostics = dict(
        n_pix_matches=int(len(p1)),
        n_kpt_matches=int(0 if pts1_kpt is None else len(pts1_kpt)),
        n_used=int(len(P1)),
        inlier_ratio=inlier_ratio,
        baseline_ok=baseline_ok,
    )

    return R, T.reshape(3, 1), mask_pose, diagnostics
