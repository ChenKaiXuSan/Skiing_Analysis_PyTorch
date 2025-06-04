#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/Skiing_Analysis_PyTorch/preprocess/main.py
Project: /workspace/code/Skiing_Analysis_PyTorch/preprocess
Created Date: Wednesday April 23rd 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Wednesday April 23rd 2025 12:30:36 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from __future__ import annotations


import logging
from pathlib import Path
import hydra

import torch
from torchvision.io import read_video

from utils.utils import save_to_pt, merge_frame_to_video

from prepare_dataset.preprocess import Preprocess


logger = logging.getLogger(__name__)


def process(parames, person: str):

    RAW_PATH = Path(parames.extract_dataset.data_path)
    SAVE_PATH = Path(parames.extract_dataset.save_path)

    logger.info(f"Start process the {person} video")

    one_person = RAW_PATH / person

    # prepare the preprocess
    preprocess = Preprocess(parames)

    for one_video in one_person.iterdir():

        vframes, _, info = read_video(one_video, pts_unit="sec", output_format="THWC")

        # * use preprocess to get information.
        # the format is: final_frames, bbox_none_index, label, optical_flow, bbox, mask, pose
        (
            bbox_none_index,
            optical_flow,
            bbox,
            mask,
            keypoints,
            keypoints_score,
            depth,
        ) = preprocess(vframes, one_video)

        # * save the video frames keypoint
        pt_info = {
            "frames": vframes.cpu(),  # THWC
            "video_name": one_video.stem,
            "video_path": str(one_video),
            "img_shape": (vframes.shape[1], vframes.shape[2]),
            "frame_count": vframes.shape[0],
            "none_index": bbox_none_index,
            "bbox": bbox.cpu(),  # xywh
            "mask": mask.cpu(),
            "optical_flow": optical_flow.cpu(),
            "depth": depth.cpu(),
            "keypoint": {
                "keypoint": keypoints.cpu(),  # xyn
                "keypoint_score": keypoints_score.cpu(),
            },
        }

        # * save the video frames to json file
        save_to_pt(one_video, SAVE_PATH, pt_info)

        torch.cuda.empty_cache()


@hydra.main(config_path="../configs/", config_name="prepare_dataset", version_base=None)
def main(parames):
    """
    main, for the multiprocessing using.

    Args:
        parames (hydra): hydra config.
    """

    for i in range(3, 7):
        process(parames, "run_{}".format(i))


if __name__ == "__main__":
    main()
