#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
fuse_eval.py (fixed)
评估左右融合后的 3D 姿态（H36M-17）：
- 左右一致性改善
- 骨段长度稳定性
- 左右肢体对称
- 时序平滑性（可选）
- 空间镜像对称性
"""
import numpy as np

# -------- H36M(17) 关节语义（常见定义）--------
# 0:Pelvis, 1:RHip, 2:RKnee, 3:RAnkle,
# 4:LHip, 5:LKnee, 6:LAnkle,
# 7:Spine, 8:Thorax, 9:Neck/Nose, 10:Head,
# 11:RShoulder, 12:RElbow, 13:RWrist,
# 14:LShoulder, 15:LElbow, 16:LWrist

H36M_EDGES = np.array([
    [0,1],[1,2],[2,3],            # 右腿
    [0,4],[4,5],[5,6],            # 左腿
    [0,7],[7,8],[8,9],[9,10],     # 躯干头部
    [8,11],[11,12],[12,13],       # 右臂
    [8,14],[14,15],[15,16]        # 左臂
], dtype=int)

# 左右肢体“成对关节”映射（用于对称度等）
# (左, 右)
LR_PAIRS = [
    (4, 1),  # LHip - RHip
    (5, 2),  # LKnee - RKnee
    (6, 3),  # LAnkle - RAnkle
    (14, 11),# LShoulder - RShoulder
    (15, 12),# LElbow - RElbow
    (16, 13) # LWrist - RWrist
]

# 左/右侧“骨段”索引（用于左右长度对比；按上面的 EDGES 选）
LEFT_BONES  = np.array([[0,4],[4,5],[5,6],[8,14],[14,15],[15,16]], dtype=int)
RIGHT_BONES = np.array([[0,1],[1,2],[2,3],[8,11],[11,12],[12,13]], dtype=int)


def _nanmean(a, axis=None):
    """对 NaN 安全的均值"""
    return np.nanmean(a, axis=axis)


def _valid_mask(*arrays):
    """公共有效掩码：所有数组都 finite 的位置"""
    mask = np.ones(arrays[0].shape[:-1], dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr).all(axis=-1)
    return mask


def bone_lengths(X, edges=H36M_EDGES):
    """
    (T,J,3) or (J,3) -> (T, len(edges)) 骨段长度。NaN 安全。
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        X = X[None, ...]  # (1,J,3)
    A, B = X[:, edges[:, 0]], X[:, edges[:, 1]]  # (T,E,3)
    diff = B - A
    L = np.linalg.norm(diff, axis=-1)  # (T,E)
    # 如果任一端点无效 -> 长度置为 NaN
    inv = ~(_valid_mask(A, B))
    L[inv] = np.nan
    return L


def side_bone_length_mean(X, which="left"):
    """计算左/右侧骨段平均长度 (T,) 或标量（取均值）"""
    edges = LEFT_BONES if which == "left" else RIGHT_BONES
    L = bone_lengths(X, edges)  # (T,E)
    return _nanmean(L)  # 标量（也可以返回 per-frame：np.nanmean(L, axis=1)）


def symmetry_score_mirror(X_last):
    """
    空间镜像对称性（单帧）：将 X 过 X轴镜像（x->-x），比较左右配对关节距离
    return: 标量，越小越对称
    """
    X = np.asarray(X_last, dtype=float)
    Xm = X.copy()
    Xm[:, 0] *= -1.0
    dists = []
    for l, r in LR_PAIRS:
        if np.all(np.isfinite(X[l])) and np.all(np.isfinite(X[r])):
            d = np.linalg.norm(X[l] - Xm[r])
            dists.append(d)
    return float(np.nan) if len(dists) == 0 else float(np.mean(dists))


def mean_pairwise_distance(A, B):
    """
    对齐度量：‖A - B‖ 的均值（在有效关节上）
    支持 (T,J,3) 或 (J,3)
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}")
    if A.ndim == 2:
        A = A[None, ...]
        B = B[None, ...]
    mask = _valid_mask(A, B)  # (T,J)
    D = np.linalg.norm(A - B, axis=-1)  # (T,J)
    D[~mask] = np.nan
    return _nanmean(D)


def temporal_stats(X):
    """
    时序平滑性：速度/加速度 P95，NaN 安全
    X: (T,J,3)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 3 or X.shape[0] < 3:
        return {}
    # 用线性插值简单补 NaN（仅用于统计，不改原数据）
    Xf = X.copy()
    T, J, C = X.shape
    t = np.arange(T)
    for j in range(J):
        for c in range(C):
            y = Xf[:, j, c]
            m = np.isfinite(y)
            if m.sum() >= 2:
                Xf[:, j, c] = np.interp(t, t[m], y[m])
    v = np.linalg.norm(np.diff(Xf, axis=0), axis=-1)       # (T-1,J)
    a = np.linalg.norm(np.diff(Xf, n=2, axis=0), axis=-1)  # (T-2,J)
    return {
        "Speed P95": float(np.percentile(v, 95)),
        "Accel P95": float(np.percentile(a, 95)),
    }


def eval_fused_pose(left_3d, right_3d, fused_3d):
    """
    评估融合质量（H36M-17）
    Args:
        left_3d, right_3d, fused_3d: (T,J,3) 或 (J,3)
    Returns:
        metrics: dict
    """
    metrics = {}

    # --- 1) 融合对齐度：左右/融合的互相距离 ---
    # 两侧之间原始差异
    metrics["L-R MeanDist (Before)"] = mean_pairwise_distance(left_3d, right_3d)
    # 融合与各侧的一致性
    metrics["Fused-Left MeanDist"]   = mean_pairwise_distance(fused_3d, left_3d)
    metrics["Fused-Right MeanDist"]  = mean_pairwise_distance(fused_3d, right_3d)
    # 两侧差异是否下降（越大越好）
    lr_after = mean_pairwise_distance(left_3d, fused_3d) + mean_pairwise_distance(right_3d, fused_3d)
    metrics["L/R→Fused Gain (approx)"] = metrics["L-R MeanDist (Before)"] - 0.5 * lr_after

    # --- 2) 骨段长度稳定性（CV） ---
    bl = bone_lengths(fused_3d)  # (T,E)
    metrics["Bone Length CV"] = float(np.nanstd(bl) / (np.nanmean(bl) + 1e-9))

    # --- 3) 左右肢体长度对称 ---
    L_mean = side_bone_length_mean(fused_3d, which="left")
    R_mean = side_bone_length_mean(fused_3d, which="right")
    metrics["LR Length Symmetry"] = float(abs(L_mean - R_mean) / (0.5 * (L_mean + R_mean) + 1e-9))

    # --- 4) 时序平滑性（可选，若有时间维） ---
    if np.asarray(fused_3d).ndim == 3 and fused_3d.shape[0] >= 3:
        metrics.update(temporal_stats(fused_3d))

    # --- 5) 空间镜像对称（取最后一帧做快照） ---
    Xlast = fused_3d[-1] if np.asarray(fused_3d).ndim == 3 else fused_3d
    metrics["Symmetry Score (mirror)"] = symmetry_score_mirror(Xlast)

    return metrics
