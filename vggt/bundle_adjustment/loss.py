#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/vggt/bundle_adjustment/loss.py
Project: /workspace/code/vggt/bundle_adjustment
Created Date: Tuesday November 25th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday November 20th 2025 5:47:43 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import torch

def project_points(X3d, R, t, K):
    """
    X3d: (T, J, 3)          # 每帧的 3D 关节点（世界坐标系）
    R:   (T, C, 3, 3)       # 每帧、每相机的旋转（world -> camera）
    t:   (T, C, 3)          # 每帧、每相机的平移（world -> camera）
    K:   (C, 3, 3)          # 每个相机的内参
    return:
        x2d_pred: (T, C, J, 2)  # 投影到各相机平面上的 2D 坐标（像素）
    """
    T, J, _ = X3d.shape
    C = R.shape[1]

    # ---- 1) 将 3D 点复制到每个相机 ----
    # X: (T, C, J, 3)
    X = X3d[:, None, :, :].expand(-1, C, -1, -1)

    # ---- 2) world -> camera: X_cam = R * X + t ----
    # 调整维度方便做 batch matmul
    # R_exp: (T, C, 1, 3, 3)
    R_exp = R.unsqueeze(2)
    # X_vec: (T, C, J, 3, 1)
    X_vec = X.unsqueeze(-1)
    # (T,C,1,3,3) @ (T,C,J,3,1) -> (T,C,J,3,1)
    X_cam = (R_exp @ X_vec).squeeze(-1)        # (T, C, J, 3)
    # 加平移 t: (T,C,1,3) -> (T,C,J,3)
    X_cam = X_cam + t.unsqueeze(2)

    # ---- 3) 归一化坐标 ----
    Z = X_cam[..., 2:3].clamp(min=1e-6)        # 防止除零
    x_norm = X_cam[..., 0:1] / Z
    y_norm = X_cam[..., 1:2] / Z
    # 齐次坐标: (x,y,1)
    xy1 = torch.cat([x_norm, y_norm, torch.ones_like(x_norm)], dim=-1)  # (T,C,J,3)

    # ---- 4) 乘相机内参 K ----
    # K: (C,3,3) -> (1,C,1,3,3), 让 T 和 J 自动广播
    K_exp = K[None, :, None, :, :]             # (1,C,1,3,3)
    # xy1_vec: (T,C,J,3,1)
    xy1_vec = xy1.unsqueeze(-1)
    # proj_h: (T,C,J,3,1)
    proj_h = K_exp @ xy1_vec
    proj = proj_h.squeeze(-1)[..., :2]         # (T,C,J,2)

    return proj


def reprojection_loss(X3d, R, t, K, x2d, conf2d, w=1.0):
    x2d_pred = project_points(X3d, R, t, K)  # (T,C,J,2)
    diff = (x2d_pred - x2d) ** 2  # (T,C,J,2)
    diff = diff.sum(dim=-1)  # (T,C,J)
    loss = (conf2d * diff).sum() / (conf2d.sum() + 1e-6)
    return w * loss


def camera_center_from_Rt(R, t):
    # C = -R^T t
    RT = R.transpose(-1, -2)  # (T,C,3,3)
    C = -(RT @ t[..., None]).squeeze(-1)  # (T,C,3)
    return C


def camera_smooth_loss(R, t, w=1e-2):
    C = camera_center_from_Rt(R, t)  # (T,C,3)
    diff = C[1:] - C[:-1]  # (T-1,C,3)
    return w * (diff**2).mean()


def baseline_reg_loss(R, t, w=1e-2):
    C = camera_center_from_Rt(R, t)  # (T,C,3)
    # 假设C=2（左、右相机）
    baseline = torch.norm(C[:, 0] - C[:, 1], dim=-1)  # (T,)
    b0 = baseline.mean().detach()
    return w * ((baseline - b0) ** 2).mean()


BONES = [
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (11, 12),
    (5, 11),
    (6, 12),
]


def bone_length_loss(X3d, ref_bone_len=None, w=1e-2):
    """
    X3d: (T,J,3)
    ref_bone_len: (len(BONES),) or None
    """
    T, J, _ = X3d.shape
    lens = []
    for i, j in BONES:
        if i >= J or j >= J:
            continue
        seg = X3d[..., i, :] - X3d[..., j, :]  # (T,3)
        lens.append(torch.norm(seg, dim=-1))  # (T,)
    if not lens:
        return torch.tensor(0.0, device=X3d.device)
    L = torch.stack(lens, dim=-1)  # (T, B)

    if ref_bone_len is None:
        # 用平均长度当作参考（不固定绝对数值，只约束同一条骨前后一致）
        ref = L.mean(dim=0, keepdim=True).detach()  # (1,B)
    else:
        ref = ref_bone_len[None, :]  # (1,B)

    loss = ((L - ref) ** 2).mean()
    return w * loss


def pose_temporal_loss(X3d, w=1e-2):
    diff = X3d[1:] - X3d[:-1]  # (T-1,J,3)
    return w * (diff**2).mean()
