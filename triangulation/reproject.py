#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/triangulation/reproject.py
Project: /workspace/code/triangulation
Created Date: Tuesday September 2nd 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Tuesday September 2nd 2025 8:34:51 pm
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
import matplotlib.pyplot as plt
from pathlib import Path


def reproject_and_visualize(
    img1: np.ndarray,
    img2: np.ndarray,
    X3: np.ndarray,  # (J,3) 3D points in Cam1/world (Cam1) coordinates
    kptL: np.ndarray,  # (J,2) observed pixels in left/cam1
    kptR: np.ndarray,  # (J,2) observed pixels in right/cam2
    K1: np.ndarray,
    dist1: np.ndarray,
    K2: np.ndarray,
    dist2: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    joint_names=None,
    circle_r: int = 5,
    thickness: int = 2,
    out_path: str = "/mnt/data/reprojection_compare.jpg",
):
    """
    - Projects 3D points X3 to both images using cv2.projectPoints (with distortion).
    - Draws observed keypoints (green), reprojected (red), and error vectors.
    - Saves a side-by-side visualization and returns per-joint errors.
    """

    J = X3.shape[0]
    # Prepare 3D points
    X3 = X3.reshape(-1, 1, 3).astype(np.float32)

    # Cam1: rvec=[0,0,0], tvec=[0,0,0]
    rvec1 = np.zeros((3, 1), np.float32)
    tvec1 = np.zeros((3, 1), np.float32)
    # Cam2: from R, T
    rvec2, _ = cv2.Rodrigues(R.astype(np.float32))
    tvec2 = T.reshape(3, 1).astype(np.float32)

    # Project with distortion model so it matches raw pixels
    proj1, _ = cv2.projectPoints(
        X3,
        rvec1,
        tvec1,
        K1.astype(np.float32),
        None if dist1 is None else dist1.astype(np.float32),
    )
    proj2, _ = cv2.projectPoints(
        X3,
        rvec2,
        tvec2,
        K2.astype(np.float32),
        None if dist2 is None else dist2.astype(np.float32),
    )
    proj1 = proj1.reshape(-1, 2)
    proj2 = proj2.reshape(-1, 2)

    # Compute per-joint errors (euclidean, pixels)
    errL = np.linalg.norm(proj1 - kptL, axis=1)
    errR = np.linalg.norm(proj2 - kptR, axis=1)

    vis1 = draw(
        img1,
        kptL,
        proj1,
        "L",
        joint_names=joint_names,
        circle_r=circle_r,
        thickness=thickness,
    )
    vis2 = draw(
        img2,
        kptR,
        proj2,
        "R",
        joint_names=joint_names,
        circle_r=circle_r,
        thickness=thickness,
    )

    # Stack side-by-side
    h = max(vis1.shape[0], vis2.shape[0])
    w = vis1.shape[1] + vis2.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[: vis1.shape[0], : vis1.shape[1]] = vis1
    canvas[: vis2.shape[0], vis1.shape[1] : vis1.shape[1] + vis2.shape[1]] = vis2
    # titles
    cv2.putText(
        canvas,
        "Left/Cam1 (Green=Observed, Red=Reprojected)",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Right/Cam2",
        (vis1.shape[1] + 20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Save
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_path = str(Path(out_path))
    cv2.imwrite(out_path, canvas)
    return {
        "proj_L": proj1,
        "proj_R": proj2,
        "err_L": errL,
        "err_R": errR,
        "mean_err_L": float(np.nanmean(errL)),
        "mean_err_R": float(np.nanmean(errR)),
        "median_err_L": float(np.nanmedian(errL)),
        "median_err_R": float(np.nanmedian(errR)),
        "out_path": out_path,
    }


# Draw helper
def draw(img, obs, rep, name="L", circle_r=3, thickness=1, joint_names=None):
    # 确保输入是 numpy 格式
    if hasattr(img, "cpu"):  # torch.Tensor
        vis = img.detach().cpu().numpy().copy()
    elif hasattr(img, "numpy"):  # numpy.ndarray
        vis = img.numpy().copy()
    else:  # 已经是 numpy
        vis = np.array(img).copy()

    for j, (o, r) in enumerate(zip(obs, rep)):
        if not np.all(np.isfinite(o)) or not np.all(np.isfinite(r)):
            continue

        # 转成 int tuple
        o = [int(round(o[0])), int(round(o[1]))]
        r = [int(round(r[0])), int(round(r[1]))]

        # * 重投影的位置超过图像的话，就归零
        if r[0] > vis.shape[1] or r[0] < 0:
            r[0] = 0

        if r[1] > vis.shape[0] or r[1] < 0:
            r[1] = 0

        # 画 observed (绿色)
        cv2.circle(vis, o, circle_r, (0, 255, 0), thickness)
        # 画 reprojected (红色)
        cv2.circle(vis, r, circle_r, (0, 0, 255), thickness)
        # 画误差线 (青色)
        cv2.line(vis, o, r, (255, 255, 0), 1)

        # 标签
        label = str(joint_names[j]) if joint_names is not None else str(j)
        cv2.putText(
            vis,
            label,
            (o[0] + 6, o[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return vis
