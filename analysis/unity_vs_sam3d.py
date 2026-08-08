#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/analysis/unity_vs_sam3d.py
Project: /workspace/code/analysis
Created Date: Friday January 16th 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Friday January 16th 2026 11:13:21 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

# %%
pose_3d = "/workspace/data/unity_data/RecordingsPose/male_pose3d_trimmed.jsonl"
pose_2d_left = "/workspace/data/unity_data/RecordingsPose/cam_left camera/male_kpt2d_left camera_trimmed.jsonl"
pose_2d_right = "/workspace/data/unity_data/RecordingsPose/cam_right camera/male_kpt2d_right camera_trimmed.jsonl"

left_video = "/workspace/data/unity_data/Recordings/male/left.mp4"
right_video = "/workspace/data/unity_data/Recordings/male/right.mp4"

sam3d_left_results = (
    "/workspace/data/sam3d_body_results/unity/male/left_sam_3d_body_outputs.npz"
)
sam3d_right_results = (
    "/workspace/data/sam3d_body_results/unity/male/right_sam_3d_body_outputs.npz"
)

# 对应 MHR70 标准中的 ID 索引
target_mhr_ids = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 41, 62, 69]

# MHR70 标准 ID 与你的 JSON 骨骼名称映射
# 格式: { MHR70_ID: "Your_JSON_Bone_Name" }
unity_mhr70_mapping = {
    0: "head",  # nose -> 用 head 代替
    1: "Bone_Eye_L",  # left_eye
    2: "Bone_Eye_R",  # right_eye
    5: "Upperarm_L",  # left_shoulder
    6: "Upperarm_R",  # right_shoulder
    7: "lowerarm_l",  # left_elbow
    8: "lowerarm_r",  # right_elbow
    9: "Thigh_L",  # left_hip
    10: "Thigh_R",  # right_hip
    11: "calf_l",  # left_knee
    12: "calf_r",  # right_knee
    13: "Foot_L",  # left_ankle
    14: "Foot_R",  # right_ankle
    15: "ball_l",  # left_big_toe -> 对应脚掌
    18: "ball_r",  # right_big_toe
    41: "Hand_R",  # right_wrist
    62: "Hand_L",  # left_wrist
    69: "neck_01",  # neck
}

# 基于图片连线逻辑的 MHR70 索引对
# 格式: (起点ID, 终点ID, 描述)
mhr70_skeleton_links = [
    # 躯干与头部
    (69, 0, "脖子 -> 头部"),
    (69, 5, "脖子 -> 左肩"),
    (69, 6, "脖子 -> 右肩"),
    (5, 6, "左肩 -> 右肩"),  # 肩膀横向连线
    (5, 1, "左肩 -> 左眼"),  # 左眼连线
    (6, 2, "右肩 -> 右眼"),  # 右眼连线
    (1, 2, "左眼 -> 右眼"),  # 眼睛横向连线
    # 左半身 (左手)
    (5, 7, "左肩 -> 左肘"),
    (7, 62, "左肘 -> 左手腕"),
    # 右半身 (右手)
    (6, 8, "右肩 -> 右肘"),
    (8, 41, "右肘 -> 右手腕"),
    # 下半身核心 (胯部)
    (9, 10, "左胯 -> 右胯"),  # 胯部横向连线
    (5, 9, "左肩膀 -> 左胯"),
    (6, 10, "右肩膀 -> 右胯"),
    # 左腿
    (9, 11, "左胯 -> 左膝"),
    (11, 13, "左膝 -> 左脚踝"),
    (13, 15, "左脚踝 -> 左脚掌"),
    # 右腿
    (10, 12, "右胯 -> 右膝"),
    (12, 14, "右膝 -> 右脚踝"),
    (14, 18, "右脚踝 -> 右脚掌"),
]

# %%
import numpy as np

left_outputs = np.load(sam3d_left_results, allow_pickle=True)["arr_0"]
right_outputs = np.load(sam3d_right_results, allow_pickle=True)["arr_0"]

# %%
for k, v in left_outputs[0].items():
    if isinstance(v, np.ndarray):
        print(f"{k}: ndarray with shape {v.shape}")
    else:
        print(f"{k}: {type(v)}")

# %%
import json
import os


def load_all_GT_data(file_path):
    """
    JSONL形式（または1行1JSON形式）のファイルを安全にロードする
    """
    all_frames_data = []

    # ファイルが存在するかチェック
    if not os.path.exists(file_path):
        print(f"Error: ファイルが見つかりません -> {file_path}")
        return []

    try:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue  # 空行をスキップ

                try:
                    # 1行ずつJSONとしてパース
                    frame_data = json.loads(line)
                    all_frames_data.append(frame_data)
                except json.JSONDecodeError as e:
                    print(f"Warning: {line_number}行目の解析に失敗しました: {e}")

        print(f"Successfully loaded {len(all_frames_data)} frames.")
    except Exception as e:
        print(f"Fatal Error: ファイルの読み込み中にエラーが発生しました: {e}")

    return all_frames_data


# --- 実行 ---
# pose_2d_left はファイルパス文字列であることを前提としています
pose_2d_left_data = load_all_GT_data(pose_2d_left)
pose_2d_right_data = load_all_GT_data(pose_2d_right)
pose_3d_data = load_all_GT_data(pose_3d)


# 最初のフレームのデータ構造を確認
if pose_2d_left_data:
    print("Keys in first frame:", pose_2d_left_data[0].keys())

# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np


def visualize_combined_pose(
    frame, raw_data, mapping_dict, skeleton_links, title="Pose Visualization"
):
    """
    Sam3D(NumPy)とUnity(Dict)の両方の形式に対応した可視化関数
    """
    v_h, v_w = frame.shape[:2]
    id_to_pos = {}

    # 1. データ形式の判定と座標変換
    if isinstance(raw_data, (np.ndarray, list)) and len(raw_data) > 0:
        # 形式A: 辞書のリストの場合 (Unity GTデータ)
        if isinstance(raw_data[0], dict):
            # mapping_dict の 逆引き (Name -> ID) を作成してIDベースに統一
            name_to_id = {v: k for k, v in mapping_dict.items()}
            for item in raw_data:
                m_id = name_to_id.get(item["name"])
                if m_id is not None:
                    # 正規化座標(0-1)ならスケーリング、ピクセル座標ならそのまま
                    u = item["u"] * v_w if item["u"] <= 1.0 else item["u"]
                    v = item["v"] * v_h if item["v"] <= 1.0 else item["v"]
                    id_to_pos[m_id] = (u, v)

        # 形式B: NumPy配列の場合 (Sam3D予測データ)
        else:
            coords = np.atleast_2d(np.array(raw_data).squeeze())
            # target_mhr_ids の順番通りに格納されている前提
            for i, mhr_id in enumerate(target_mhr_ids):
                if i < len(coords):
                    u = coords[i, 0] * v_w if coords[i, 0] <= 1.0 else coords[i, 0]
                    v = coords[i, 1] * v_h if coords[i, 1] <= 1.0 else coords[i, 1]
                    id_to_pos[mhr_id] = (u, v)

    # 2. 描画処理
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 骨格（線）
    for start_id, end_id, *info in skeleton_links:
        if start_id in id_to_pos and end_id in id_to_pos:
            p1, p2 = id_to_pos[start_id], id_to_pos[end_id]
            plt.plot(
                [p1[0], p2[0]], [p1[1], p2[1]], color="lime", linewidth=2, alpha=0.6
            )

    # 関節（点）
    for mhr_id, (u, v) in id_to_pos.items():
        name = mapping_dict.get(mhr_id, str(mhr_id)).split("_")[-1]
        plt.scatter(u, v, c="red", s=30, edgecolors="white", zorder=5)
        plt.text(
            u + 2,
            v - 2,
            name,
            color="yellow",
            fontsize=8,
            fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.5, lw=0),
        )

    plt.title(title)
    plt.axis("off")
    plt.show()


# --- 実行部分 ---

frame = left_outputs[0]["frame"]

# 1. Sam3Dデータの準備 (NumPy)
# shapeが (1, 70, 2) の場合は [0] で抜き出し、target_mhr_ids でフィルタリング
sam3d_input = left_outputs[0]["pred_keypoints_2d"][target_mhr_ids]

# 2. Unityデータの準備 (JSONからロードした辞書リスト)
# pose_2d_left_data[0] が [{'name': 'head', 'u':...}, ...] の形式であることを想定
unity_input = (
    pose_2d_left_data[0]["joints2d"]
    if "joints2d" in pose_2d_left_data[0]
    else pose_2d_left_data[0]
)

# unity的数据需要转换坐标
# 3. フィルタリングと座標変換 (建立 骨骼名 -> 像素坐标 的映射)
v_h, v_w = frame.shape[:2]
# 注意：如果 JSON 中没有宽向高度信息，请确保默认值正确
d_w, d_h = (
    unity_input[0].get("image_width", v_w),
    unity_input[0].get("image_height", v_h),
)

scale_x, scale_y = v_w / d_w, v_h / d_h

unity_input_normalized = {}
for j in unity_input:
    u = j["u"] * scale_x
    # 注意：如果你的 2D 数据 v=0 是顶部，则不需要 v_h - ...
    # 如果可视化发现倒置了，请保留或移除这一行
    v = v_h - (j["v"] * scale_y)
    unity_input_normalized[j["name"]] = (u, v)

# filter unity data
unity_input_filtered = []
for k, v in unity_input_normalized.items():
    for i in target_mhr_ids:
        if k in unity_mhr70_mapping[i]:
            unity_input_filtered.append((k, v))

# 3. それぞれ実行
visualize_combined_pose(
    frame,
    sam3d_input,
    unity_mhr70_mapping,
    mhr70_skeleton_links,
    title="Sam3D Prediction",
)
visualize_combined_pose(
    frame,
    unity_input_normalized,
    unity_mhr70_mapping,
    mhr70_skeleton_links,
    title="Unity GT",
)
