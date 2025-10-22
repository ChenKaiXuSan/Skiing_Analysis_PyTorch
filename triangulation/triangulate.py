#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/triangulation/triangulate.py
Project: /workspace/code/triangulation
Created Date: Wednesday October 22nd 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Wednesday October 22nd 2025 10:51:04 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""
import os
import numpy as np
import cv2
import logging


logger = logging.getLogger(__name__)

from triangulation.reproject import reproject_and_visualize

# vis
from triangulation.vis.pose_visualization import (
    visualize_3d_joints,
)

# K_dist = np.array([0.17697328, -0.45675065, -0.0026601, -0.00330938, 0.35538705])
K_dist = np.array(
    [
        -1.1940477842823853,
        -15.440461757486913,
        0.00013163161053023783,
        0.00019082529328353381,
        98.843073622415901,
        -1.3588290520381034,
        -14.555841222727574,
        96.219667412855202,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
)


# ---------- 三角测量 ----------
def triangulate_joints(keypoints1, keypoints2, K, R, T):
    if keypoints1.shape != keypoints2.shape or keypoints1.shape[1] != 2:
        raise ValueError(
            f"Keypoints shape mismatch: {keypoints1.shape} vs {keypoints2.shape}"
        )
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, T.reshape(3, 1)))
    pts_4d = cv2.triangulatePoints(P1, P2, keypoints1.T, keypoints2.T)
    return (pts_4d[:3, :] / pts_4d[3, :]).T


def process_triangulate(
    left_kpts, right_kpts, left_vframes, right_vframes, K, R, T, output_path
):

    joints_3d_all = []

    for l_kpt, r_kpt, l_frame, r_frame, r, t, i in zip(
        left_kpts, right_kpts, left_vframes, right_vframes, R, T, range(len(left_kpts))
    ):
        W, H = l_frame.shape[1], l_frame.shape[0]

        # * 这里是没有过滤的3d pose
        joints_3d = triangulate_joints(l_kpt, r_kpt, K, r, t)

        # * 可视化3d关节
        visualize_3d_joints(
            joints_3d=joints_3d,
            R=r,
            T=t,
            K=K,
            image_size=(W, H),
            save_path=os.path.join(output_path, f"frame_{i:04d}.png"),
            title=f"Frame {i} - 3D Joints",
            y_up=True,
        )

        res = reproject_and_visualize(
            img1=l_frame,
            img2=r_frame,
            X3=joints_3d,
            kptL=l_kpt,  # (J,2)
            kptR=r_kpt,  # (J,2)
            K1=K,
            dist1=K_dist,
            K2=K,
            dist2=K_dist,
            R=r,
            T=t,  # 注意：T要用“有尺度”的（基线已缩放）
            joint_names=None,  # 或者传 COCO 的关节名列表
            out_path=os.path.join(output_path, "reproj", f"{i:04d}.jpg"),
        )
        logger.info(f"Saved to: {res['out_path']}")
        logger.info(
            f"Reprojection error - Frame {i}: Left {res['mean_err_L']:.2f}px, Right {res['mean_err_R']:.2f}px"
        )

        joints_3d_all.append(joints_3d)

    return joints_3d_all
