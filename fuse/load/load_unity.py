#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/project/load.py
Project: /workspace/code/project
Created Date: Friday January 30th 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Friday January 30th 2026 5:01:44 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""
import json
import os

import numpy as np

# --- 設定 ---
UNITY_MHR70_MAPPING = {
    1: "Bone_Eye_L",
    2: "Bone_Eye_R",
    5: "Upperarm_L",
    6: "Upperarm_R",
    7: "lowerarm_l",
    8: "lowerarm_r",
    9: "Thigh_L",
    10: "Thigh_R",
    11: "calf_l",
    12: "calf_r",
    13: "Foot_L",
    14: "Foot_R",
    41: "Hand_R",
    62: "Hand_L",
    69: "neck_01",
}
TARGET_IDS = list(UNITY_MHR70_MAPPING.keys())


def harmonize_to_pixel_coords(
    sam3d_raw,
    unity_raw,
    mapping_dict,
    target_ids,
    width=1920,
    height=1080,
    scale_x=1.0,
    scale_y=1.0,
):
    """
    Sam3D(ピクセル座標)とUnity(負のピクセル座標)を共通のピクセル座標系に統一する
    2d版本
    """
    unified_sam3d = {}
    unified_unity = {}

    # --- 1. Sam3Dデータの統一 (型を float に整理) ---
    sam3d_coords = np.atleast_2d(sam3d_raw.squeeze())
    for i, m_id in enumerate(target_ids):
        if i < len(sam3d_coords):
            # np.float32 を標準の float にキャストしてタプル化
            unified_sam3d[m_id] = (float(sam3d_coords[i][0]), float(sam3d_coords[i][1]))

    # --- 2. Unityデータの統一 (V軸の負の値を補正) ---
    name_to_id = {name: m_id for m_id, name in mapping_dict.items()}
    joints_list = (
        unity_raw.get("joints2d", unity_raw)
        if isinstance(unity_raw, dict)
        else unity_raw
    )

    for item in joints_list:
        m_id = name_to_id.get(item["name"])
        if m_id in target_ids:
            # Unityデータがピクセル値で、かつVが負の値(-461.9など)の場合
            # 画像の上端を0とするには絶対値(abs)を取るのが一般的です
            u_px = float(item["u"]) * scale_x
            v_px = height - (float(item["v"]) * scale_y)

            unified_unity[m_id] = (u_px, v_px)

    return unified_sam3d, unified_unity


def convert_unity_to_sam3d_coords(unity_kpts_3d):
    """
    将unity的3D坐标转换为Sam3D的3D坐标系
    """
    sam3d_coords = {}

    for i, (x, y, z) in unity_kpts_3d.items():
        name = UNITY_MHR70_MAPPING[i]

        # Unity座標系からSam3D座標系への変換
        # x=[z],
        # y=[-y],
        # z=[x],
        x_sam = -z
        y_sam = -y
        z_sam = x

        sam3d_coords[i] = np.array([x_sam, y_sam, z_sam])

    return sam3d_coords


def get_unity_gt_dicts(gt_2d_raw, gt_3d_raw, height=1080):
    """Unityの1フレーム分のGTデータを辞書形式に整理"""
    name_to_id = {v: k for k, v in UNITY_MHR70_MAPPING.items()}

    # 2D GT (Pixel coords)
    gt_2d = {
        name_to_id[item["name"]]: np.array(
            [float(item["u"]), height - float(item["v"])]
        )
        for item in gt_2d_raw.get("joints2d", [])
        if item["name"] in name_to_id
    }

    # 3D GT (Unity to Sam3D axis conversion)
    gt_3d = {
        name_to_id[item["name"]]: np.array(
            [-float(item["z"]), -float(item["y"]), float(item["x"])]
        )
        for item in gt_3d_raw.get("joints3d", [])
        if item["name"] in name_to_id
    }

    return gt_2d, gt_3d


def get_sam_pred_dicts(sam_frame):
    """Sam3Dの1フレーム分の予測データを辞書形式に整理"""
    # 2D Prediction
    pred_2d = sam_frame["pred_keypoints_2d"][TARGET_IDS]

    # 3D Prediction

    pred_3d = sam_frame["pred_keypoints_3d"][TARGET_IDS]

    return pred_2d, pred_3d


def load(paths: dict):
    # データ読み込み
    try:
        sam_l = np.load(paths["sam_l"], allow_pickle=True)["arr_0"]
        sam_r = np.load(paths["sam_r"], allow_pickle=True)["arr_0"]
    except:
        sam_l = np.load(paths["sam_l"], allow_pickle=True)['outputs']
        sam_r = np.load(paths["sam_r"], allow_pickle=True)['outputs']

    gt_2d_l = None
    gt_2d_r = None
    gt_3d = None

    if paths.get("gt_2d_l") and os.path.exists(paths["gt_2d_l"]):
        gt_2d_l = [
            json.loads(line)
            for line in open(paths["gt_2d_l"], "r", encoding="utf-8-sig")
        ]
    if paths.get("gt_2d_r") and os.path.exists(paths["gt_2d_r"]):
        gt_2d_r = [
            json.loads(line)
            for line in open(paths["gt_2d_r"], "r", encoding="utf-8-sig")
        ]
    if paths.get("gt_3d") and os.path.exists(paths["gt_3d"]):
        gt_3d = [
            json.loads(line)
            for line in open(paths["gt_3d"], "r", encoding="utf-8-sig")
        ]

    lengths = [len(sam_l), len(sam_r)]
    if gt_2d_l is not None:
        lengths.append(len(gt_2d_l))
    if gt_2d_r is not None:
        lengths.append(len(gt_2d_r))
    if gt_3d is not None:
        lengths.append(len(gt_3d))

    num_frames = min(lengths)

    all_frame_results = {}
    # 通过左右视角融合单帧
    for i in range(num_frames):
        # GT整理
        g2d_l = None
        g2d_r = None
        g3d_l = None
        g3d_r = None

        if gt_2d_l is not None and gt_3d is not None:
            g2d_l, g3d_l = get_unity_gt_dicts(gt_2d_l[i], gt_3d[i])
        if gt_2d_r is not None and gt_3d is not None:
            g2d_r, g3d_r = get_unity_gt_dicts(gt_2d_r[i], gt_3d[i])

        # Pred整理
        p2d_l, p3d_l = get_sam_pred_dicts(sam_l[i])
        p2d_r, p3d_r = get_sam_pred_dicts(sam_r[i])

        # 2d 坐标转换
        if gt_2d_l is not None:
            p2d_l, g2d_l = harmonize_to_pixel_coords(
                sam3d_raw=p2d_l,
                unity_raw=gt_2d_l[i],
                mapping_dict=UNITY_MHR70_MAPPING,
                target_ids=TARGET_IDS,
                width=1920,
                height=1080,
                scale_x=1.0,
                scale_y=1.0,
            )

        if gt_2d_r is not None:
            p2d_r, g2d_r = harmonize_to_pixel_coords(
                sam3d_raw=p2d_r,
                unity_raw=gt_2d_r[i],
                mapping_dict=UNITY_MHR70_MAPPING,
                target_ids=TARGET_IDS,
                width=1920,
                height=1080,
                scale_x=1.0,
                scale_y=1.0,
            )

        # 3d 坐标转换
        if g3d_l is not None:
            p3d_l = convert_unity_to_sam3d_coords(g3d_l)

        if g3d_r is not None:
            p3d_r = convert_unity_to_sam3d_coords(g3d_r)

        # 按照顺序整理
        p2d_l = {k: v for k, v in sorted(p2d_l.items())}
        p2d_r = {k: v for k, v in sorted(p2d_r.items())}
        if g2d_l is not None:
            g2d_l = {k: v for k, v in sorted(g2d_l.items())}
        if g2d_r is not None:
            g2d_r = {k: v for k, v in sorted(g2d_r.items())}
        p3d_l = {k: v for k, v in sorted(p3d_l.items())}
        p3d_r = {k: v for k, v in sorted(p3d_r.items())}
        if g3d_l is not None:
            g3d_l = {k: v for k, v in sorted(g3d_l.items())}
        if g3d_r is not None:
            g3d_r = {k: v for k, v in sorted(g3d_r.items())}

        all_frame_results[i] = {
            "L_2D": {"pred": p2d_l, "gt": g2d_l},
            "R_2D": {"pred": p2d_r, "gt": g2d_r},
            "L_3D": {"pred": p3d_l, "gt": g3d_l},
            "R_3D": {"pred": p3d_r, "gt": g3d_r},
        }

    return all_frame_results