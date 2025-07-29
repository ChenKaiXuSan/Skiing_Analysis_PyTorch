#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/project/prepare_data_2d_custom.py
Project: /workspace/code/project
Created Date: Thursday June 12th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday June 12th 2025 8:32:31 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch

import hydra
from pathlib import Path

from VideoPose3D.data.data_utils import suggest_metadata

import numpy as np

COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def interpolate_keypoints(kp):
    """
    kp: ndarray of shape (T, 17, 2) — keypoints over time
    其中 (0, 0) 被认为是缺失值，将其替换为 np.nan，再用线性插值
    """
    kp = kp.copy()
    mask = (kp[:, :, 0] == 0) & (kp[:, :, 1] == 0)
    kp[mask] = np.nan

    print(f"none keypoints: {np.sum(mask)}")
    res_valid = 0

    T = kp.shape[0]
    indices = np.arange(T)
    for j in range(17):  # 17个关键点
        for d in range(2):  # x 和 y 坐标
            valid = ~np.isnan(kp[:, j, d])
            if np.sum(valid) < 2:
                kp[:, j, d] = 0  # 如果有效点少于2个，直接置为0
                continue  # 不能插值
            kp[:, j, d] = np.interp(indices, indices[valid], kp[valid, j, d])
            res_valid += np.sum(valid)

    print("{} total frames processed".format(len(kp)))
    print("{} frames were interpolated".format(res_valid))
    print("----------")
    return kp


def interpolate_bboxes(bboxes):
    """
    对 (T, 4) 的 bbox 数组中含 NaN 的位置做线性插值
    """
    bboxes = bboxes.copy()
    T, D = bboxes.shape
    indices = np.arange(T)
    for i in range(D):  # 对每个 bbox 分量（x1, y1, x2, y2）分别处理
        valid = ~np.isnan(bboxes[:, i])
        if np.sum(valid) < 2:
            continue  # 无法插值
        bboxes[:, i] = np.interp(indices, indices[valid], bboxes[valid, i])

    print("{} total frames processed".format(len(bboxes)))
    print("{} frames were interpolated".format(np.sum(~valid)))
    print("----------")

    return bboxes


COCO_ADJACENT = {
    1: [0, 3],
    3: [1, 5],
    7: [5, 9],
    9: [7],
    11: [5, 13],
    13: [11, 15],
    # 继续添加其他你感兴趣的点
}


def estimate_by_neighbors(kpts):
    """
    使用邻居估算关键点。记录被补的位置。

    参数:
        kpts: (T, N, 2) 的 numpy 数组，关键点数据（0 表示缺失）

    返回:
        kpts: 补全后的关键点数组
        filled_indices: 列表，元素为 (t, i)，表示第 t 帧的第 i 个点被补全
    """
    T, N, _ = kpts.shape
    kpts = kpts.copy()
    filled_indices = []

    for t in range(T):
        for i in range(N):
            if np.all(kpts[t, i] == 0):
                neighbors = COCO_ADJACENT.get(i, [])
                valid = [kpts[t, j] for j in neighbors if not np.all(kpts[t, j] == 0)]
                if valid:
                    kpts[t, i] = np.mean(valid, axis=0)
                    filled_indices.append((t, i))  # 记录这个点被补了
    return kpts, filled_indices


def decode(filename):
    # Latin1 encoding because Detectron runs on Python 2.7
    print("Processing {}".format(filename))
    data = torch.load(filename, map_location="cpu")

    bb = (
        data["bbox"].numpy() if isinstance(data["bbox"], torch.Tensor) else data["bbox"]
    )
    kp = (
        data["keypoint"]["keypoint"].numpy()
        if isinstance(data["keypoint"]["keypoint"], torch.Tensor)
        else data["keypoints"]
    )
    kp_score = (
        data["keypoint"]["keypoint_score"].numpy()
        if isinstance(data["keypoint"]["keypoint"], torch.Tensor)
        else data["keypoint"]["keypoint_score"]
    )

    metadata = {
        "video_name": data["video_name"],
        "img_shape": data["img_shape"],
        "frame_count": data["frame_count"],
        "video_path": str(data["video_path"]),
        "none_index": (
            data["none_index"].numpy()
            if isinstance(data["none_index"], torch.Tensor)
            else data["none_index"]
        ),
    }

    bb = np.array(bb, dtype=np.float32)
    kp = np.array(kp, dtype=np.float32)

    # TODO: 这里应该是有问题的，因为bbox和kpt没有怼起来
    # Fix missing bboxes/keypoints by linear interpolation

    bb = interpolate_bboxes(bb)
    kp = interpolate_keypoints(kp)

    # * if interpolate not work, we use nebighborhood method to fill the missing values
    # kp = estimate_by_neighbors(kp)
    kp, filled_points = estimate_by_neighbors(kp)
    print("共补全了 {} 个关键点:".format(len(filled_points)))
    for t, i in filled_points:
        print(f"第 {t} 帧，第 {i} 个关键点（{COCO_KEYPOINT_NAMES[i]}）被补全")

    return [
        {
            "start_frame": 0,  # Inclusive
            "end_frame": len(kp),  # Exclusive
            "bounding_boxes": bb,
            "keypoints": kp,
        }
    ], metadata


@hydra.main(config_path="../../configs", config_name="convert_custom_data")
def process(args):

    if not args.input:
        print("Please specify the input directory")
        exit(0)
    INPUT = Path(args.input)

    if not args.output:
        print("Please specify an output suffix (e.g. detectron_pt_coco)")
        exit(0)

    print("Parsing 2D detections from", args.input)

    metadata = suggest_metadata("coco")
    metadata["video_metadata"] = {}

    output = {}
    # file_list = glob(args.input + '/*.npz')
    file_list = INPUT.glob("**/*.pt")
    # for _p in INPUT.dirglob('*.pt'):

    for f in file_list:
        canonical_name = f"{f.parts[-2]}_{f.stem}"
        data, video_metadata = decode(f)
        output[canonical_name] = {}
        output[canonical_name]["custom"] = [data[0]["keypoints"].astype("float32")]
        metadata["video_metadata"][canonical_name] = video_metadata

    print("Saving...")
    np.savez_compressed(args.output, positions_2d=output, metadata=metadata)
    print("Done.")


if __name__ == "__main__":

    process()
