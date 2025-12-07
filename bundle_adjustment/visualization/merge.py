#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/bundle_adjustment/visualization/merge.py
Project: /workspace/code/bundle_adjustment/visualization
Created Date: Sunday December 7th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Sunday December 7th 2025 4:32:17 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import logging
from pathlib import Path

import cv2
from tqdm import tqdm

logger = logging.getLogger(__name__)


def merge_frame_to_video(save_path: Path, flag: str) -> None:
    _save_path = save_path / flag
    _out_path = save_path / "video"

    frames = sorted(list(_save_path.iterdir()), key=lambda x: int(x.stem.split("_")[0]))

    if not _out_path.exists():
        _out_path.mkdir(parents=True, exist_ok=True)

    first_frame = cv2.imread(str(frames[0]))
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        str(_out_path / (flag + ".mp4")), fourcc, 30.0, (width, height)
    )

    for f in tqdm(frames, desc=f"Save {flag}", total=len(frames)):
        img = cv2.imread(str(f))
        out.write(img)

    out.release()

    logger.info(f"Video saved to {_out_path / (flag + '.mp4')}")
