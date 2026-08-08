#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/front_side/side/metadata/mhr70 copy.py
Project: /workspace/code/front_side/side/metadata
Created Date: Friday January 16th 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Friday January 16th 2026 10:18:57 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

mhr_names = [
    "nose",
    "left-eye",
    "right-eye",
    "left-ear",
    "right-ear",
    "left-shoulder",
    "right-shoulder",
    "left-elbow",
    "right-elbow",
    "left-hip",
    "right-hip",
    "left-knee",
    "right-knee",
    "left-ankle",
    "right-ankle",
    "left-big-toe-tip",
    "left-small-toe-tip",
    "left-heel",
    "right-big-toe-tip",
    "right-small-toe-tip",
    "right-heel",
    "right-thumb-tip",
    "right-thumb-first-joint",
    "right-thumb-second-joint",
    "right-thumb-third-joint",
    "right-index-tip",
    "right-index-first-joint",
    "right-index-second-joint",
    "right-index-third-joint",
    "right-middle-tip",
    "right-middle-first-joint",
    "right-middle-second-joint",
    "right-middle-third-joint",
    "right-ring-tip",
    "right-ring-first-joint",
    "right-ring-second-joint",
    "right-ring-third-joint",
    "right-pinky-tip",
    "right-pinky-first-joint",
    "right-pinky-second-joint",
    "right-pinky-third-joint",
    "right-wrist",
    "left-thumb-tip",
    "left-thumb-first-joint",
    "left-thumb-second-joint",
    "left-thumb-third-joint",
    "left-index-tip",
    "left-index-first-joint",
    "left-index-second-joint",
    "left-index-third-joint",
    "left-middle-tip",
    "left-middle-first-joint",
    "left-middle-second-joint",
    "left-middle-third-joint",
    "left-ring-tip",
    "left-ring-first-joint",
    "left-ring-second-joint",
    "left-ring-third-joint",
    "left-pinky-tip",
    "left-pinky-first-joint",
    "left-pinky-second-joint",
    "left-pinky-third-joint",
    "left-wrist",
    "left-olecranon",
    "right-olecranon",
    "left-cubital-fossa",
    "right-cubital-fossa",
    "left-acromion",
    "right-acromion",
    "neck",
]

# Unityボーン構造に合わせたMHR70ベースの定義
unity_mhr70_pose_info = dict(
    pose_format="unity_mhr70_reduced",
    min_visible_keypoints=8,
    image_height=4096,
    image_width=2668,
    # 使用する18個のキーポイント定義
    keypoint_info={
        0: dict(name="head", id=0, color=[51, 153, 255], swap=""),
        1: dict(name="Bone_Eye_L", id=1, color=[51, 153, 255], swap="Bone_Eye_R"),
        2: dict(name="Bone_Eye_R", id=2, color=[51, 153, 255], swap="Bone_Eye_L"),
        5: dict(name="Upperarm_L", id=5, color=[0, 255, 0], swap="Upperarm_R"),
        6: dict(name="Upperarm_R", id=6, color=[255, 128, 0], swap="Upperarm_L"),
        7: dict(name="lowerarm_l", id=7, color=[0, 255, 0], swap="lowerarm_r"),
        8: dict(name="lowerarm_r", id=8, color=[255, 128, 0], swap="lowerarm_l"),
        9: dict(name="Thigh_L", id=9, color=[0, 255, 0], swap="Thigh_R"),
        10: dict(name="Thigh_R", id=10, color=[255, 128, 0], swap="Thigh_L"),
        11: dict(name="calf_l", id=11, color=[0, 255, 0], swap="calf_r"),
        12: dict(name="calf_r", id=12, color=[255, 128, 0], swap="calf_l"),
        13: dict(name="Foot_L", id=13, color=[0, 255, 0], swap="Foot_R"),
        14: dict(name="Foot_R", id=14, color=[255, 128, 0], swap="Foot_L"),
        15: dict(name="ball_l", id=15, color=[0, 255, 0], swap="ball_r"),
        18: dict(name="ball_r", id=18, color=[255, 128, 0], swap="ball_l"),
        41: dict(name="Hand_R", id=41, color=[255, 128, 0], swap="Hand_L"),
        62: dict(name="Hand_L", id=62, color=[0, 255, 0], swap="Hand_R"),
        69: dict(name="neck_01", id=69, color=[51, 153, 255], swap=""),
    },
    # Unityでの可視化に必要なスケルトン接続（リンク）定義
    skeleton_info={
        0: dict(link=("neck_01", "head"), id=0, color=[51, 153, 255]),
        1: dict(link=("Upperarm_L", "lowerarm_l"), id=1, color=[0, 255, 0]),
        2: dict(link=("lowerarm_l", "Hand_L"), id=2, color=[0, 255, 0]),
        3: dict(link=("Upperarm_R", "lowerarm_r"), id=3, color=[255, 128, 0]),
        4: dict(link=("lowerarm_r", "Hand_R"), id=4, color=[255, 128, 0]),
        5: dict(link=("Thigh_L", "calf_l"), id=5, color=[0, 255, 0]),
        6: dict(link=("calf_l", "Foot_L"), id=6, color=[0, 255, 0]),
        7: dict(link=("Foot_L", "ball_l"), id=7, color=[0, 255, 0]),
        8: dict(link=("Thigh_R", "calf_r"), id=8, color=[255, 128, 0]),
        9: dict(link=("calf_r", "Foot_R"), id=9, color=[255, 128, 0]),
        10: dict(link=("Foot_R", "ball_r"), id=10, color=[255, 128, 0]),
    },
    joint_weights=[1.0] * 70,  # インデックスの整合性を保つため元の長さを維持
    sigmas=[],
)
