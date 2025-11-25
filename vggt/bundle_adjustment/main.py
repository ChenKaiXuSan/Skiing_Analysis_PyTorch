#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/vggt/bundle_adjustment/main.py
Project: /workspace/code/vggt/bundle_adjustment
Created Date: Tuesday November 25th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Tuesday November 25th 2025 3:36:04 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import torch
import torch.nn as nn

from .loss import (
    reprojection_loss,
    camera_smooth_loss,
    baseline_reg_loss,
    bone_length_loss,
    pose_temporal_loss,
)


def run_local_ba(
    K_torch,  # (C,3,3)
    R_init_torch,  # (T,C,3,3)  固定
    t_init_torch,  # (T,C,3)    可优化
    X3d_init_torch,  # (T,J,3)    可优化
    x2d_torch,  # (T,C,J,2)
    conf2d_torch,  # (T,C,J)
    num_iters=200,
    lr=1e-3,
    device="cuda",
):
    K = K_torch.to(device)
    R = R_init_torch.to(device)

    t_param = nn.Parameter(t_init_torch.to(device))  # 优化 t
    X_param = nn.Parameter(X3d_init_torch.to(device))  # 优化 3D pose

    optimizer = torch.optim.Adam([t_param, X_param], lr=lr)

    x2d = x2d_torch.to(device)
    conf2d = conf2d_torch.to(device)

    # 可选：先用初始化算一个骨长参考
    with torch.no_grad():
        ref_bone = None
        # 如果你想固定骨长绝对值，可以用这一行：
        # ref_bone = compute_ref_bone_lengths_np(X3d_init) -> 再转torch

    for it in range(num_iters):
        optimizer.zero_grad()

        # 1) 重投影误差
        L_reproj = reprojection_loss(X_param, R, t_param, K, x2d, conf2d, w=1.0)

        # 2) camera 平滑 + baseline 正则
        L_cam_smooth = camera_smooth_loss(R, t_param, w=1e-2)
        L_baseline = baseline_reg_loss(R, t_param, w=1e-2)

        # 3) 3D pose 几何 + 时序
        L_bone = bone_length_loss(X_param, ref_bone_len=None, w=1e-2)
        L_pose = pose_temporal_loss(X_param, w=1e-2)

        loss = L_reproj + L_cam_smooth + L_baseline + L_bone + L_pose
        loss.backward()
        optimizer.step()

        if (it + 1) % 20 == 0:
            print(
                f"[BA] iter={it + 1} "
                f"reproj={L_reproj.item():.4f} "
                f"cam_sm={L_cam_smooth.item():.4f} "
                f"base={L_baseline.item():.4f} "
                f"bone={L_bone.item():.4f} "
                f"pose={L_pose.item():.4f}"
            )

    X_opt = X_param.detach().cpu().numpy()  # (T,J,3)
    t_opt = t_param.detach().cpu().numpy()  # (T,C,3)

    return X_opt, t_opt
