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
from pathlib import Path
import glob

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


def load_sam_data(path):
	"""加载 SAM 数据，支持单个文件（列表）或按帧保存的多个文件。
	
	Args:
		path: npz 文件路径（str 或 Path）
		
	Returns:
		list: 包含所有帧数据的列表
	"""
	path = Path(path)
	
	# 情况1: 如果路径是一个 .npz 文件
	if path.is_file() and path.suffix == '.npz':
		data = np.load(str(path), allow_pickle=True)
		# 尝试不同的键名
		if "arr_0" in data:
			outputs = data["arr_0"]
		elif "outputs" in data:
			outputs = data["outputs"]
		else:
			raise KeyError(f"Cannot find 'arr_0' or 'outputs' in {path}")
		
		# 检查是否为列表，如果是单个元素则转换为列表
		if not isinstance(outputs, (list, np.ndarray)):
			outputs = [outputs]
		elif isinstance(outputs, np.ndarray) and outputs.ndim == 0:
			# 处理 0-d array 的情况
			outputs = [outputs.item()]
			
		return list(outputs)
	
	# 情况2: 如果路径指向一个目录或包含通配符的模式
	# 尝试找到所有按帧保存的文件
	if path.is_dir():
		# 在目录中查找所有 frame_*_sam_3d_body_outputs.npz 文件
		pattern = str(path / "frame_*_sam_3d_body_outputs.npz")
	else:
		# 路径可能是一个模式，去掉文件名，在父目录中查找
		parent_dir = path.parent
		pattern = str(parent_dir / "frame_*_sam_3d_body_outputs.npz")
	
	frame_files = sorted(glob.glob(pattern))
	
	if frame_files:
		# 按帧保存的情况：加载所有帧
		all_frames = []
		for frame_file in frame_files:
			data = np.load(frame_file, allow_pickle=True)
			if "outputs" in data:
				frame_output = data["outputs"]
				# outputs 应该是一个只包含一个元素的列表
				if isinstance(frame_output, (list, np.ndarray)) and len(frame_output) > 0:
					all_frames.append(frame_output[0])
				else:
					all_frames.append(frame_output)
			elif "arr_0" in data:
				frame_output = data["arr_0"]
				if isinstance(frame_output, (list, np.ndarray)) and len(frame_output) > 0:
					all_frames.append(frame_output[0])
				else:
					all_frames.append(frame_output)
		return all_frames
	
	# 如果都不是，抛出错误
	raise FileNotFoundError(f"Cannot load SAM data from {path}")


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
	"""加载并处理 SAM 数据。
	
	Args:
		paths: 包含 'sam_l' 和 'sam_r' 键的字典，值为文件路径或目录路径
		
	Returns:
		dict: 包含所有帧结果的字典
	"""
	# データ読み込み - 支持单个文件或按帧保存的多个文件
	sam_l = load_sam_data(paths["sam_l"])
	sam_r = load_sam_data(paths["sam_r"])

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
