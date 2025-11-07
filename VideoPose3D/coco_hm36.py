#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/code/VideoPose3D/coco_hm36.py
Project: /workspace/code/VideoPose3D
Created Date: Friday November 7th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Friday November 7th 2025 4:23:07 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''
import numpy as np

# ---------------------------
# 索引定义
# ---------------------------
# COCO-17:
# 0 nose, 1 l_eye, 2 r_eye, 3 l_ear, 4 r_ear,
# 5 l_sho, 6 r_sho, 7 l_elb, 8 r_elb, 9 l_wri, 10 r_wri,
# 11 l_hip, 12 r_hip, 13 l_knee, 14 r_knee, 15 l_ank, 16 r_ank
COCO = dict(
    NOSE=0, L_EYE=1, R_EYE=2, L_EAR=3, R_EAR=4,
    L_SHO=5, R_SHO=6, L_ELB=7, R_ELB=8, L_WRI=9, R_WRI=10,
    L_HIP=11, R_HIP=12, L_KNE=13, R_KNE=14, L_ANK=15, R_ANK=16
)

# H36M-17 (VideoPose3D 常用):
# 0 Pelvis, 1 RHip, 2 RKnee, 3 RAnkle,
# 4 LHip, 5 LKnee, 6 LAnkle,
# 7 Spine, 8 Thorax, 9 Neck/Nose, 10 Head,
# 11 LShoulder, 12 LElbow, 13 LWrist,
# 14 RShoulder, 15 RElbow, 16 RWrist
H36M = dict(
    PEL=0, R_HIP=1, R_KNE=2, R_ANK=3, L_HIP=4, L_KNE=5, L_ANK=6,
    SPINE=7, THORAX=8, NECK=9, HEAD=10,
    L_SHO=11, L_ELB=12, L_WRI=13, R_SHO=14, R_ELB=15, R_WRI=16
)

# ---------------------------
# 工具：形状展开/还原
# ---------------------------
def _to_TJC(X):
    """接受 (J,C) 或 (T,J,C) → 返回 (T,J,C), T=1 时同时给出还原 lambda"""
    X = np.asarray(X)
    if X.ndim == 2:
        X_TJC = X[None, ...]
        restore = lambda Y: Y[0]
    elif X.ndim == 3:
        X_TJC = X
        restore = lambda Y: Y
    else:
        raise ValueError("Input must be (J,C) or (T,J,C).")
    return X_TJC, restore

def _mid(a, b):
    return (a + b) * 0.5

def _safe_get(X, idx):
    return X[:, idx, :]

# ---------------------------
# COCO → H36M
# ---------------------------
def coco_to_h36m(X_coco, synthesize_head=True):
    """
    X_coco: (J,C) or (T,J,C), J=17, C=2/3
    返回 H36M-17 (T,17,C) 或 (17,C)
    近似：
      - Pelvis = (L_HIP, R_HIP) 中点
      - Thorax = (L_SHO, R_SHO) 中点
      - Spine  = Pelvis 与 Thorax 的中点
      - Neck   = COCO Nose（更稳可用 Thorax 与 Nose 的中点，按需修改）
      - Head   = 若 synthesize_head=True，基于 Nose 与双眼估计；否则=Neck
    """
    X, restore = _to_TJC(X_coco)
    T, J, C = X.shape
    out = np.full((T, 17, C), np.nan, dtype=X.dtype)

    # 方便访问
    nose = _safe_get(X, COCO["NOSE"])
    l_eye = _safe_get(X, COCO["L_EYE"])
    r_eye = _safe_get(X, COCO["R_EYE"])
    l_ear = _safe_get(X, COCO["L_EAR"])
    r_ear = _safe_get(X, COCO["R_EAR"])
    l_sho = _safe_get(X, COCO["L_SHO"])
    r_sho = _safe_get(X, COCO["R_SHO"])
    l_elb = _safe_get(X, COCO["L_ELB"])
    r_elb = _safe_get(X, COCO["R_ELB"])
    l_wri = _safe_get(X, COCO["L_WRI"])
    r_wri = _safe_get(X, COCO["R_WRI"])
    l_hip = _safe_get(X, COCO["L_HIP"])
    r_hip = _safe_get(X, COCO["R_HIP"])
    l_kne = _safe_get(X, COCO["L_KNE"])
    r_kne = _safe_get(X, COCO["R_KNE"])
    l_ank = _safe_get(X, COCO["L_ANK"])
    r_ank = _safe_get(X, COCO["R_ANK"])

    pelvis = _mid(l_hip, r_hip)
    thorax = _mid(l_sho, r_sho)
    spine  = _mid(pelvis, thorax)

    # 粗略估计 head：用鼻子与双眼方向外推；若眼缺失则退化为 nose
    if synthesize_head:
        eyes_mid = _mid(l_eye, r_eye)
        # 估计头顶为沿着 (nose -> eyes_mid) 反向延伸一定比例
        vec = nose - eyes_mid
        head = nose + 0.5 * vec
    else:
        head = nose.copy()

    neck = nose  # 简化处理；如需可改为 _mid(thorax, nose)

    # 填充 H36M
    out[:, H36M["PEL"], :] = pelvis
    out[:, H36M["R_HIP"], :] = r_hip
    out[:, H36M["R_KNE"], :] = r_kne
    out[:, H36M["R_ANK"], :] = r_ank
    out[:, H36M["L_HIP"], :] = l_hip
    out[:, H36M["L_KNE"], :] = l_kne
    out[:, H36M["L_ANK"], :] = l_ank
    out[:, H36M["SPINE"], :] = spine
    out[:, H36M["THORAX"], :] = thorax
    out[:, H36M["NECK"], :] = neck
    out[:, H36M["HEAD"], :] = head
    out[:, H36M["L_SHO"], :] = l_sho
    out[:, H36M["L_ELB"], :] = l_elb
    out[:, H36M["L_WRI"], :] = l_wri
    out[:, H36M["R_SHO"], :] = r_sho
    out[:, H36M["R_ELB"], :] = r_elb
    out[:, H36M["R_WRI"], :] = r_wri

    return restore(out)

# ---------------------------
# H36M → COCO
# ---------------------------
def h36m_to_coco(X_h36m, synthesize_face=False, face_mode="nan"):
    """
    X_h36m: (J,C) or (T,J,C), J=17, C=2/3
    返回 COCO-17 (T,17,C) 或 (17,C)
    注意：H36M 没有眼/耳，默认填 NaN；若 synthesize_face=True，则基于 Head/Neck/Shoulder 估计眼耳。
      - Nose: 用 H36M 的 Neck(9) 近似（或可用 Head/Thorax 推断）
      - Eyes/Ears: 若 synthesize_face=True，用几何启发式从 Head/Neck + 肩宽构造
      - Pelvis: COCO 无 pelvis，COCO 的髋点直接来自 H36M
    参数：
      synthesize_face: 是否合成眼/耳
      face_mode: "nan" | "approx" （兼容性标志，"approx" 等价于 synthesize_face=True）
    """
    if face_mode == "approx":
        synthesize_face = True

    X, restore = _to_TJC(X_h36m)
    T, J, C = X.shape
    out = np.full((T, 17, C), np.nan, dtype=X.dtype)

    pel = _safe_get(X, H36M["PEL"])
    r_hip = _safe_get(X, H36M["R_HIP"]); r_kne = _safe_get(X, H36M["R_KNE"]); r_ank = _safe_get(X, H36M["R_ANK"])
    l_hip = _safe_get(X, H36M["L_HIP"]); l_kne = _safe_get(X, H36M["L_KNE"]); l_ank = _safe_get(X, H36M["L_ANK"])
    spine = _safe_get(X, H36M["SPINE"])
    thor  = _safe_get(X, H36M["THORAX"])
    neck  = _safe_get(X, H36M["NECK"])
    head  = _safe_get(X, H36M["HEAD"])
    l_sho = _safe_get(X, H36M["L_SHO"]); l_elb = _safe_get(X, H36M["L_ELB"]); l_wri = _safe_get(X, H36M["L_WRI"])
    r_sho = _safe_get(X, H36M["R_SHO"]); r_elb = _safe_get(X, H36M["R_ELB"]); r_wri = _safe_get(X, H36M["R_WRI"])

    # Nose：直接用 H36M 的 NECK（很多实现把 9 当鼻根/颈部近似）
    nose = neck.copy()

    # 眼耳合成（可选）：用肩宽确定横向尺度，用 head->neck 确定纵向方向
    if synthesize_face:
        shoulder_mid = _mid(l_sho, r_sho)
        shoulder_vec = l_sho - r_sho
        if np.linalg.norm(shoulder_vec, axis=1).mean() < 1e-8:
            # 肩重合，退化
            shoulder_vec = np.tile(np.array([1, 0, 0])[:C], (T, 1))
        side_dir = shoulder_vec / (np.linalg.norm(shoulder_vec, axis=1, keepdims=True) + 1e-9)
        up_dir = (head - neck)
        up_dir = up_dir / (np.linalg.norm(up_dir, axis=1, keepdims=True) + 1e-9)
        # 经验系数（你可以按自己数据调）
        eye_offset_side = 0.25 * np.linalg.norm(shoulder_vec, axis=1, keepdims=True)
        eye_offset_up   = 0.15 * np.linalg.norm(shoulder_vec, axis=1, keepdims=True)
        ear_offset_side = 0.35 * np.linalg.norm(shoulder_vec, axis=1, keepdims=True)
        ear_offset_up   = 0.10 * np.linalg.norm(shoulder_vec, axis=1, keepdims=True)

        l_eye = head + (+eye_offset_side) * side_dir - eye_offset_up * up_dir
        r_eye = head + (-eye_offset_side) * side_dir - eye_offset_up * up_dir
        l_ear = head + (+ear_offset_side) * side_dir - ear_offset_up * up_dir
        r_ear = head + (-ear_offset_side) * side_dir - ear_offset_up * up_dir
    else:
        l_eye = r_eye = l_ear = r_ear = np.full_like(nose, np.nan)

    # 填充 COCO
    out[:, COCO["NOSE"], :] = nose
    out[:, COCO["L_EYE"], :] = l_eye
    out[:, COCO["R_EYE"], :] = r_eye
    out[:, COCO["L_EAR"], :] = l_ear
    out[:, COCO["R_EAR"], :] = r_ear
    out[:, COCO["L_SHO"], :] = l_sho
    out[:, COCO["R_SHO"], :] = r_sho
    out[:, COCO["L_ELB"], :] = l_elb
    out[:, COCO["R_ELB"], :] = r_elb
    out[:, COCO["L_WRI"], :] = l_wri
    out[:, COCO["R_WRI"], :] = r_wri
    out[:, COCO["L_HIP"], :] = l_hip
    out[:, COCO["R_HIP"], :] = r_hip
    out[:, COCO["L_KNE"], :] = l_kne
    out[:, COCO["R_KNE"], :] = r_kne
    out[:, COCO["L_ANK"], :] = l_ank
    out[:, COCO["R_ANK"], :] = r_ank

    return restore(out)

# ---------------------------
# 简单自测（随机）
# ---------------------------
if __name__ == "__main__":
    # 假设 3D，单帧
    coco_xyz = np.random.randn(17, 3)
    h36m_xyz = coco_to_h36m(coco_xyz)           # (17,3)
    coco_back = h36m_to_coco(h36m_xyz)          # (17,3)
    print("COCO->H36M shape:", h36m_xyz.shape, "H36M->COCO shape:", coco_back.shape)

    # 多帧
    coco_seq = np.random.randn(5, 17, 3)
    h36m_seq = coco_to_h36m(coco_seq)
    coco_seq_back = h36m_to_coco(h36m_seq, synthesize_face=True)
    print("Seq OK:", h36m_seq.shape, coco_seq_back.shape)
