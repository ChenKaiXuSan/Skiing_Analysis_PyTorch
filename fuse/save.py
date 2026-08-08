#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/code/project/save.py
Project: /workspace/code/project
Created Date: Wednesday February 11th 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Wednesday February 11th 2026 11:22:17 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''

import os
from typing import Dict, Iterable, List

import numpy as np


def _seq_dicts_to_array(
	seq_dicts: List[Dict[int, Iterable[float]]],
	target_ids: List[int],
) -> np.ndarray:
	"""
	Convert a list of {joint_id: (x,y,z)} dicts to a (T, J, 3) array.
	Missing or non-finite joints are filled with NaN.
	"""
	t_len = len(seq_dicts)
	j_len = len(target_ids)
	out = np.full((t_len, j_len, 3), np.nan, dtype=np.float64)

	for t, frame in enumerate(seq_dicts):
		for j, jid in enumerate(target_ids):
			if jid in frame:
				xyz = np.asarray(frame[jid], dtype=np.float64)
				if xyz.shape == (3,) and np.all(np.isfinite(xyz)):
					out[t, j] = xyz
				else:
					out[t, j] = xyz.reshape(-1)[:3]

	return out


def save_smoothed_results(
	smooth_seq: List[Dict[int, Iterable[float]]],
	target_ids: List[int],
	out_path: str,
) -> str:
	"""
	Save only the smoothed 3D keypoints sequence.
	Returns the output path.
	"""
	if not smooth_seq:
		raise ValueError("smooth_seq is empty; nothing to save")

	os.makedirs(os.path.dirname(out_path), exist_ok=True)

	seq_array = _seq_dicts_to_array(smooth_seq, target_ids)
	np.save(out_path, seq_array)
	return out_path
