#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/VideoPose3D/fuse/fuse_check.py
Project: /workspace/code/VideoPose3D/fuse
Created Date: Friday November 7th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Friday November 7th 2025 8:16:46 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import numpy as np


def estimate_rigid_umeyama(target, source, allow_scale=False):
    """
    使用 Umeyama 方法估计刚体/相似变换，使 s*R*Y + t ≈ X
    Args:
        X, Y: (N,3) 对应点集，须按一一对应顺序排列
        allow_scale: 是否允许相似变换的尺度 s
    Returns:
        R(3,3), t(3,), s(float), info(dict)
    """
    target = np.asarray(target, dtype=float)
    source = np.asarray(source, dtype=float)
    assert target.shape == source.shape and target.shape[1] == 3, "X,Y 必须是 (N,3)"
    mask = np.all(np.isfinite(target), axis=1) & np.all(np.isfinite(source), axis=1)
    target, source = target[mask], source[mask]

    N = target.shape[0]
    if N < 3:
        raise ValueError("至少需要 3 个非共线对应点")

    target_mean = target.mean(0)
    source_mean = source.mean(0)

    target_mm = target - target_mean
    source_mm = source - source_mean

    H = (source_mm.T @ target_mm) / N

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # 反射修正：det(R) 应当为 +1
    reflect_fix = False
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
        reflect_fix = True

    if allow_scale:
        varY = (source_mm**2).sum() / N
        s = (S.sum()) / (varY + 1e-12)
    else:
        s = 1.0

    t = target_mean - s * (R @ source_mean)

    info = {
        "num_points": N,
        "singular_values": S,
        "reflect_fixed": reflect_fix,
        "H_rank": int(np.linalg.matrix_rank(H)),
        "cond_H": (S[0] / S[-1]) if S[-1] > 0 else np.inf,
    }
    return R, t, s, info


def _pairwise_distances(A):
    """(N,3) → 上三角成对距离向量（不含对角）"""
    N = A.shape[0]
    dists = []
    for i in range(N):
        v = A[i + 1 :] - A[i]
        d = np.linalg.norm(v, axis=1)
        dists.append(d)
    return np.concatenate(dists, axis=0) if dists else np.array([])


def check_rigid_validity(X, Y, R, t, allow_scale=False, tol=1e-6):
    """
    对“刚体/相似变换”进行全面体检。
    Args:
        X, Y: (N,3) 对应点集
        R, t: 估计的旋转与平移（若是相似变换，可先外部给 s=1 的 R,t，再单独传 s）
        allow_scale: 若为 True，会额外评估距离比例（但刚体应为 False）
        tol: 正交性/距离保持等判断阈值
    Returns:
        report(dict): 各类布尔检查与数值指标
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    assert X.shape == Y.shape and X.shape[1] == 3, "X,Y 必须是 (N,3)"
    mask = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(Y), axis=1)
    X, Y = X[mask], Y[mask]

    N = X.shape[0]
    report = {"ok": True, "notes": []}

    # -------- 基础数据检查 --------
    report["num_points"] = N
    report["enough_points"] = N >= 3
    if not report["enough_points"]:
        report["ok"] = False
        report["notes"].append("点数不足（<3）")

    # 退化/共线/共面粗查（通过中心化后协方差矩阵秩）
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    rank_X = np.linalg.matrix_rank(Xc)
    rank_Y = np.linalg.matrix_rank(Yc)
    report["rank_X"] = int(rank_X)
    report["rank_Y"] = int(rank_Y)
    report["degenerate_X"] = rank_X < 2
    report["degenerate_Y"] = rank_Y < 2
    if report["degenerate_X"] or report["degenerate_Y"]:
        report["ok"] = False
        report["notes"].append("X 或 Y 退化（秩 < 2）")

    # 共面提示：秩 == 2 可能导致旋转绕法歧义
    report["coplanar_X_hint"] = rank_X == 2
    report["coplanar_Y_hint"] = rank_Y == 2

    # -------- 刚体矩阵性质 --------
    RtR = R.T @ R
    I3 = np.eye(3)
    ortho_err = np.linalg.norm(RtR - I3, ord="fro")
    detR = np.linalg.det(R)

    report["orthogonality_error_Fro"] = float(ortho_err)
    report["det_R"] = float(detR)
    report["R_is_orthogonal"] = ortho_err < 1e-6
    report["R_det_is_pos_one"] = abs(detR - 1.0) < 1e-6

    if not report["R_is_orthogonal"]:
        report["ok"] = False
        report["notes"].append("R 非正交")
    if not report["R_det_is_pos_one"]:
        report["ok"] = False
        report["notes"].append("det(R) ≠ 1（可能出现反射/缩放问题）")

    # -------- 距离保持（刚体核心性质）--------
    # 理论上刚体不允许尺度变化；allow_scale=True 时，仅报告比例
    dX = _pairwise_distances(X)
    Yt = (R @ Y.T).T + t  # 刚体假设 s=1
    dYt = _pairwise_distances(Yt)
    common = min(len(dX), len(dYt))
    if common >= 1:
        dist_diff = np.abs(dX[:common] - dYt[:common])
        report["pairwise_dist_mean_abs_err"] = float(np.mean(dist_diff))
        report["pairwise_dist_max_abs_err"] = float(np.max(dist_diff))
        report["distance_preserved"] = report["pairwise_dist_mean_abs_err"] < tol
        if not report["distance_preserved"] and not allow_scale:
            report["ok"] = False
            report["notes"].append("点间距离未保持（非刚体）")
    else:
        report["pairwise_dist_mean_abs_err"] = np.nan
        report["pairwise_dist_max_abs_err"] = np.nan
        report["distance_preserved"] = False
        report["notes"].append("成对距离不足，无法检查距离保持")

    # -------- 配准残差 --------
    resid = X - Yt
    per_point_err = np.linalg.norm(resid, axis=1)
    report["rmse"] = float(np.sqrt(np.mean(per_point_err**2)))
    report["median_err"] = float(np.median(per_point_err))
    report["p95_err"] = float(np.percentile(per_point_err, 95))

    # -------- 方向/尺度提示（可选）--------
    if allow_scale:
        # 粗略估一个比例（基于平均成对距离）
        ratio = (
            (np.mean(dX) / (np.mean(_pairwise_distances(Y)) + 1e-12))
            if len(dX) > 0
            else np.nan
        )
        report["distance_ratio_X_over_Y"] = float(ratio)
        # 注意：这只是量纲提示，不代表 s
    else:
        report["distance_ratio_X_over_Y"] = None

    return report
