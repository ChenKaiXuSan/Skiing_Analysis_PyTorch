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



import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
from pathlib import Path
from torchvision.io import read_video

import multiprocessing
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

import hydra

from project.utils import del_folder, make_folder
from prepare_dataset.preprocess import Preprocess


def process(parames, person: str):

    RAW_PATH = Path(parames.extract_dataset.data_path)
    SAVE_PATH = Path(parames.extract_dataset.save_path)

    # prepare the log file
    logger = logging.getLogger(f"Logger-{multiprocessing.current_process().name}")

    logger.info(f"Start process the {person} video!")

    # prepare the preprocess
    preprocess = Preprocess(parames)

    res = dict()

    one_person = RAW_PATH / person

    for one_video in one_person.iterdir():

        vframes, audio, _ = read_video(one_video, pts_unit="sec", output_format="TCHW")

        # * step3: use preprocess to get information.
        # the format is: final_frames, bbox_none_index, label, optical_flow, bbox, mask, pose
        # TCHW > CTHW > BCTHW
        m_vframes = vframes.permute(1, 0, 2, 3).unsqueeze(0)
        (
            frames,
            bbox_none_index,
            optical_flow,
            bbox,
            mask,
            keypoints,
            keypoints_score,
        ) = preprocess(m_vframes, 0)

        # * step4: save the video frames keypoint
        anno = dict()

        # * packe the keypoint and keypoint_score
        anno["keypoint"] = keypoints.cpu().numpy()
        anno["keypoint_score"] = keypoints_score.cpu().numpy()

        # * packe the bbox and mask
        sample_json_info = {
            "video_name": one_video.name,
            "video_path": one_video,
            "img_shape": (vframes.shape[2], vframes.shape[3]),
            "frame_count": vframes.shape[0],
            "none_index": bbox_none_index,
            "bbox": [bbox[0, i].tolist() for i in range(bbox.shape[1])],
            "mask": mask,
            "keypoint": anno,
        }

        res[one_video.name] = sample_json_info

    # * step5: save the video frames to json file
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    save_to_json(res, SAVE_PATH, logger)  # save the sample info to json file.


def save_to_json(sample_info, save_path, logger) -> None:
    """save the sample info to json file.

    Args:
        sample_info (dict): _description_
        save_path (Path): _description_
        logger (logging): _description_
    """
    # TODO: 这个需要修改一下，符合格式
    save_path = Path(save_path, "json_file")

    save_path_with_name = (
        save_path / sample_info["disease"] / (sample_info["video_name"] + ".json")
    )

    make_folder(save_path_with_name.parent)
    with open(save_path_with_name, "w") as f:
        json.dump(sample_info, f, indent=4)
    logger.info(f"Save the {sample_info['video_name']} to {save_path}")


@hydra.main(config_path="../configs/", config_name="prepare_dataset")
def main(parames):
    """
    main, for the multiprocessing using.

    Args:
        parames (hydra): hydra config.
    """

    # ! only for test
    process(parames, "run_2")

    # threads = []
    # for d in [["ASD"], ["LCS", "HipOA"]]:

    #     thread = multiprocessing.Process(target=process, args=(parames, "fold0", d))
    #     threads.append(thread)

    # for t in threads:
    #     t.start()

    # for t in threads:
    #     t.join()

    # process(parames, "fold0", ["DHS"])


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')# good solution !!!!
    main()
