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

import torch
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

RAW_CLASS = ["ASD", "DHS", "LCS", "HipOA"]
CLASS = ["ASD", "DHS", "LCS_HipOA", "Normal"]
map_CLASS = {"ASD": 0, "DHS": 1, "LCS_HipOA": 2, "Normal": 3}  # map the class to int

def process(parames, person: str):

    RAW_PATH = Path(parames.extract_dataset.data_path)
    SAVE_PATH = Path(parames.extract_dataset.save_path)

    # prepare the log file
    logger = logging.getLogger(f"Logger-{multiprocessing.current_process().name}")

    logger.info(f"Start process the {person} video!")

    # prepare the preprocess
    preprocess = Preprocess(parames)

    # * step1: load the video path, with the sorted order
    
    # k is disease, v is (video_path, info)
    for video in RAW_PATH.iterdir():

        logger.info(f"Start process the video {video.name}!")

        # * step2: load the video from vieo path
        # get the bbox
        vframes, audio, _ = read_video(
            video, pts_unit="sec", output_format="TCHW"
        )

        # * step3: use preprocess to get information.
        # the format is: final_frames, bbox_none_index, label, optical_flow, bbox, mask, pose
        # TCHW > CTHW > BCTHW
        m_vframes = vframes.permute(1, 0, 2, 3).unsqueeze(0)
        (
            frames,
            bbox_none_index,
            label,
            optical_flow,
            bbox,
            mask,
            keypoints,
            keypoints_score,
        ) = preprocess(m_vframes, label, 0)

        # * step4: save the video frames keypoint
        anno = dict()

        # * when use mmaction, we need convert the keypoint torch to numpy
        anno["keypoint"] = keypoints.cpu().numpy()
        anno["keypoint_score"] = keypoints_score.cpu().numpy()
        anno["frame_dir"] = video_path
        anno["img_shape"] = (vframes.shape[2], vframes.shape[3])
        anno["original_shape"] = (vframes.shape[2], vframes.shape[3])
        anno["total_frames"] = keypoints.shape[1]
        anno["label"] = int(label)

        res[info["flag"]].append(anno)

        # break;

    # save one disease to pkl file.
    SAVE_PATH.mkdir(parents=True, exist_ok=True)

    with open(SAVE_PATH / f'{"_".join(disease)}.pkl', "wb") as file:
        pickle.dump(res, file)

    logging.info(f"Save the {fold} {disease} to {SAVE_PATH}")


def save_to_json(sample_info, save_path, logger, method: str) -> None:
    """save the sample info to json file.

    There have three method to get the gait cycle index, include mix, pose, bbox.

    The sample info include:

    video_name: the video name,
    video_path: the video path, relative path from /workspace/skeleton/data/segmentation_dataset_512
    frame_count: the raw frames of the video,
    label: the label of the video,
    disease: the disease of the video,
    gait_cycle_index: the gait cycle index,
    bbox_none_index: the bbox none index, when use yolo to get the bbox, some frame will not get the bbox.
    bbox: the bbox, [n, 4] (cxcywh)

    Args:
        sample_info (dict): the sample info dict.
        save_path (str): the prefix save path, like /workspace/skeleton/data/segmentation_dataset_512
        logger (Logger): for multiprocessing logging.
        method (str): the method to get the gait cycle index, include mix, pose, bbox.
    """

    save_path = Path(save_path, "json_" + method)

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
    process(parames, "run_1")

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
