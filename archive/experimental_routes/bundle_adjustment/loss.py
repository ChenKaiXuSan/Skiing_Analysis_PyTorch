#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Differentiable losses for multi-view bundle adjustment
Author: Kaixu Chen
"""

from typing import Optional
import torch


# ----------------------------------------------------------
#  Differentiable projection: WORLD 3D → pixel 2D
# ----------------------------------------------------------
import torch

def project_points(
    X3d: torch.Tensor,  # (T,J,3) or (J,3)
    R: torch.Tensor,    # (T,C,3,3) or (C,3,3)
    t: torch.Tensor,    # (T,C,3)   or (C,3)
    K: torch.Tensor,    # (C,3,3)   or (T,C,3,3)
) -> torch.Tensor:
    """
    Vectorised projection for all frames / cameras / joints.
    Return: (T,C,J,2)
    """
    device = X3d.device
    dtype = X3d.dtype

    K = K.to(device=device, dtype=dtype)
    R = R.to(device=device, dtype=dtype)
    t = t.to(device=device, dtype=dtype)

    # ---- ensure time dim ----
    if X3d.dim() == 2:        # (J,3) -> (1,J,3)
        X3d = X3d.unsqueeze(0)
    T, J, _ = X3d.shape

    # ---- handle R,t shapes ----
    if R.dim() == 3:          # (C,3,3) -> (T,C,3,3)
        C = R.shape[0]
        R = R.unsqueeze(0).expand(T, -1, -1, -1)
        t = t.unsqueeze(0).expand(T, -1, -1)
    elif R.dim() == 4:        # (T,C,3,3)
        T_R, C, _, _ = R.shape
        assert T_R == T
        if t.dim() == 2:      # (C,3) -> (T,C,3)
            t = t.unsqueeze(0).expand(T, -1, -1)
        else:
            assert t.shape[:2] == (T, C)
    else:
        raise ValueError(f"Unsupported R shape: {R.shape}")

    # ---- world -> camera ----
    # X: (T,C,J,3)
    X = X3d.unsqueeze(1).expand(-1, C, -1, -1)

    # 显式给 R 多一个 J 维度，避免广播错位
    # R_exp: (T,C,1,3,3) -> 自动广播到 (T,C,J,3,3)
    R_exp = R.unsqueeze(2)                 # (T,C,1,3,3)
    X_vec = X.unsqueeze(-1)                # (T,C,J,3,1)
    X_cam = torch.matmul(R_exp, X_vec).squeeze(-1)  # (T,C,J,3)

    X_cam = X_cam + t.unsqueeze(2)         # (T,C,J,3)

    # ---- normalised coords ----
    Z = X_cam[..., 2:3].clamp(min=1e-6)
    xy_norm = X_cam[..., 0:2] / Z          # (T,C,J,2)

    ones = torch.ones_like(Z)
    xy1 = torch.cat([xy_norm, ones], dim=-1)  # (T,C,J,3)

    # ---- intrinsics ----
    if K.dim() == 3:                       # (C,3,3)
        K_expand = K.unsqueeze(0).unsqueeze(2)   # (1,C,1,3,3)
    elif K.dim() == 4:                     # (T,C,3,3)
        K_expand = K.unsqueeze(2)               # (T,C,1,3,3)
    else:
        raise ValueError(f"Unsupported K shape: {K.shape}")

    proj_h = torch.matmul(K_expand, xy1.unsqueeze(-1))  # (T,C,J,3,1)
    proj = proj_h.squeeze(-1)[..., :2]                  # (T,C,J,2)

    return proj


# ----------------------------------------------------------
#  Losses
# ----------------------------------------------------------
def reprojection_loss(X3d, R, t, K, x2d, conf2d, w=1.0) -> torch.Tensor:
    pred = project_points(X3d, R, t, K)  # (T,C,J,2)
    diff = (pred - x2d) ** 2  # (T,C,J,2)
    diff = diff.sum(-1)  # (T,C,J)
    return w * (conf2d * diff).sum() / (conf2d.sum() + 1e-6)


def camera_center_from_Rt(R, t):
    RT = R.transpose(-1, -2)
    C = -(RT @ t[..., None]).squeeze(-1)
    return C  # (T,C,3)


def camera_smooth_loss(R, t, w=1e-2):
    C = camera_center_from_Rt(R, t)
    diff = C[1:] - C[:-1]
    return w * (diff**2).mean()


def baseline_reg_loss(R, t, w=1e-2):
    C = camera_center_from_Rt(R, t)
    if C.shape[1] < 2:  # single-camera case
        return torch.tensor(0.0, device=C.device)
    baseline = torch.norm(C[:, 0] - C[:, 1], dim=-1)
    return w * ((baseline - baseline.mean().detach()) ** 2).mean()


# Skeletal topology
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
    T, J, _ = X3d.shape
    lens = []
    for i, j in BONES:
        if i >= J or j >= J:
            continue
        seg = X3d[..., i, :] - X3d[..., j, :]
        lens.append(torch.norm(seg, dim=-1))
    if not lens:
        return torch.tensor(0.0, device=X3d.device)
    L = torch.stack(lens, dim=-1)  # (T,B)
    ref = (
        L.mean(0, keepdim=True).detach()
        if ref_bone_len is None
        else ref_bone_len[None, :].to(X3d.device)
    )
    return w * ((L - ref) ** 2).mean()


def pose_temporal_loss(X3d, w=1e-2):
    diff = X3d[1:] - X3d[:-1]
    return w * (diff**2).mean()
