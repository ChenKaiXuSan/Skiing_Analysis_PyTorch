#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/code/triangulation/postprocess.py
Project: /workspace/code/triangulation
Created Date: Tuesday September 2nd 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Tuesday September 2nd 2025 7:06:59 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''
import numpy as np, cv2
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# ---------- 工具函数 ----------
def build_P(K, R=np.eye(3), t=np.zeros(3)):
    return K @ np.hstack([R, t.reshape(3,1)])

def project(P, X3):  # X3:(N,3)
    Xh = np.hstack([X3, np.ones((X3.shape[0],1))])
    Y  = (Xh @ P.T)
    return Y[:, :2] / (Y[:, 2:3] + 1e-12)

def reproj_errors(P1, P2, X3, x1_pix, x2_pix):
    p1 = project(P1, X3); p2 = project(P2, X3)
    e1 = np.linalg.norm(p1 - x1_pix, axis=1)
    e2 = np.linalg.norm(p2 - x2_pix, axis=1)
    return e1, e2, 0.5*(e1+e2)

def positive_depth_mask(R, T, X3):
    # 左相机系Z>0 & 右相机系Z>0
    z1 = X3[:,2]
    Xr = (X3 @ R.T) + T.reshape(1,3)
    z2 = Xr[:,2]
    return (z1 > 0) & (z2 > 0)

def smooth_skeleton(X, win=9, poly=2):
    """X: [T,J,3] -> 平滑后的同形状"""
    Xs = X.copy()
    T, J, C = X.shape
    win = min(win if win%2==1 else win+1, max(1 if T%2==1 else T-1, 3))
    for j in range(J):
        for c in range(C):
            vec = X[:, j, c]
            mask = np.isfinite(vec)
            if mask.sum() >= win:
                v = vec.copy()
                v[mask] = savgol_filter(vec[mask], window_length=win, polyorder=poly)
                Xs[:, j, c] = v
    return Xs

# ---------- 主函数：单帧过滤 ----------
def post_triage_single(
    X3_frame,                   # (J,3) 三角结果
    kptL_frame, kptR_frame,     # (J,2) 像素
    K1, K2, R, T,               # 相机参数
    dist1=None, dist2=None,     # 可为 None（若三角前已去畸变）
    confL=None, confR=None, conf_thr=0.3,
    err_thresh_px=2.0,
    return_masks=False
):
    J = X3_frame.shape[0]

    # 可选：把像素先去畸变（如果三角时用的是像素+K）
    def undist(x, K, d):
        if d is None: return x
        u = cv2.undistortPoints(x.reshape(-1,1,2), K, d, P=K).reshape(-1,2)
        return u

    x1 = undist(kptL_frame, K1, dist1)
    x2 = undist(kptR_frame, K2, dist2)

    P1 = build_P(K1, np.eye(3), np.zeros(3))
    P2 = build_P(K2, R, T)

    e1, e2, em = reproj_errors(P1, P2, X3_frame, x1, x2)
    pos_mask = positive_depth_mask(R, T, X3_frame)

    conf_mask = np.ones(J, dtype=bool)
    if confL is not None and confR is not None:
        conf_mask = (confL >= conf_thr) & (confR >= conf_thr)

    err_mask  = np.isfinite(em) & (em <= err_thresh_px)
    keep = pos_mask & err_mask & conf_mask

    X3_clean = X3_frame.copy()
    X3_clean[~keep] = np.nan

    report = {
        "rmse_px": float(np.sqrt(np.nanmean(em**2))),
        "median_err_px": float(np.nanmedian(em)),
        "pos_depth_ratio": float(np.mean(pos_mask)),
        "kept_ratio": float(np.mean(keep)),
        "kept_count": int(keep.sum()),
    }
    return (X3_clean, report, keep) if return_masks else (X3_clean, report)

# ---------- 可视化（单帧3D骨架） ----------
def plot_skeleton_3d(X3_frame, bones=None, title="3D Joints", elev=20, azim=-60):
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111, projection='3d')
    pts = X3_frame

    # 点
    mask = np.isfinite(pts).all(1)
    ax.scatter(pts[mask,0], pts[mask,1], pts[mask,2], s=30, c='b', alpha=0.9)
    for i, m in enumerate(mask):
        if m:
            ax.text(pts[i,0], pts[i,1], pts[i,2], f"{i}", fontsize=8)

    # 骨架
    if bones is not None:
        for (i,j) in bones:
            if mask[i] and mask[j]:
                ax.plot([pts[i,0], pts[j,0]],
                        [pts[i,1], pts[j,1]],
                        [pts[i,2], pts[j,2]], linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout(); plt.show()

# ---------- 批量（整段） ----------
def post_triage_sequence(
    X3_seq, kptL_seq, kptR_seq, K1, K2, R, T,
    dist1=None, dist2=None, confL=None, confR=None,
    conf_thr=0.3, err_thresh_px=2.0, smooth=False, sg_win=9, sg_poly=2
):
    Tn, J, _ = X3_seq.shape
    X_clean = np.full_like(X3_seq, np.nan, dtype=np.float32)
    stats = []

    for t in range(Tn):
        cl, rep = post_triage_single(
            X3_seq[t], kptL_seq[t], kptR_seq[t],
            K1, K2, R, T, dist1, dist2,
            confL=None if confL is None else confL[t],
            confR=None if confR is None else confR[t],
            conf_thr=conf_thr, err_thresh_px=err_thresh_px
        )
        X_clean[t] = cl
        stats.append(rep)

    if smooth:
        X_clean = smooth_skeleton(X_clean, win=sg_win, poly=sg_poly)

    return X_clean, stats

