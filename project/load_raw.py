#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/project/load_raw.py
Project: /workspace/code/project
Created Date: Tuesday February 11th 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Tuesday February 11th 2026
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

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


def _array_to_dict_keep_nan(arr: np.ndarray, target_ids):
	"""Keep all target IDs, including NaN values, to preserve index alignment."""
	out = {}
	arr = np.asarray(arr)
	for idx, jid in enumerate(target_ids):
		if idx < arr.shape[0]:
			out[jid] = np.asarray(arr[idx], dtype=np.float64)
	return out


def get_sam_pred_dicts(sam_frame):
	"""Sam3Dの1フレーム分の予測データを辞書形式に整理"""
	pred_2d = sam_frame["pred_keypoints_2d"][TARGET_IDS]
	pred_3d = sam_frame["pred_keypoints_3d"][TARGET_IDS]
	return pred_2d, pred_3d


def load_raw(paths: dict):
	# データ読み込み
	try:
		sam_l = np.load(paths["sam_l"], allow_pickle=True)["arr_0"]
		sam_r = np.load(paths["sam_r"], allow_pickle=True)["arr_0"]
	except Exception:
		sam_l = np.load(paths["sam_l"], allow_pickle=True)["outputs"]
		sam_r = np.load(paths["sam_r"], allow_pickle=True)["outputs"]

	num_frames = min(len(sam_l), len(sam_r))
	all_frame_results = {}

	for i in range(num_frames):
		p2d_l, p3d_l = get_sam_pred_dicts(sam_l[i])
		p2d_r, p3d_r = get_sam_pred_dicts(sam_r[i])

		p2d_l = _array_to_dict_keep_nan(p2d_l, TARGET_IDS)
		p2d_r = _array_to_dict_keep_nan(p2d_r, TARGET_IDS)
		p3d_l = _array_to_dict_keep_nan(p3d_l, TARGET_IDS)
		p3d_r = _array_to_dict_keep_nan(p3d_r, TARGET_IDS)

		p2d_l = {k: v for k, v in sorted(p2d_l.items())}
		p2d_r = {k: v for k, v in sorted(p2d_r.items())}
		p3d_l = {k: v for k, v in sorted(p3d_l.items())}
		p3d_r = {k: v for k, v in sorted(p3d_r.items())}

		all_frame_results[i] = {
			"L_2D": {"pred": p2d_l, "gt": None},
			"R_2D": {"pred": p2d_r, "gt": None},
			"L_3D": {"pred": p3d_l, "gt": None},
			"R_3D": {"pred": p3d_r, "gt": None},
		}

	return all_frame_results
