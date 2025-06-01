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


import json
import logging
from pathlib import Path
import hydra
import cv2
from tqdm import tqdm

import torch
from torchvision.io import read_video

from project.utils import del_folder, make_folder
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

        res = {}

        vframes, _, info = read_video(one_video, pts_unit="sec", output_format="THWC")

        # * use preprocess to get information.
        # the format is: final_frames, bbox_none_index, label, optical_flow, bbox, mask, pose
        (
            frames,
            bbox_none_index,
            optical_flow,
            bbox,
            mask,
            keypoints,
            keypoints_score,
            depth,
        ) = preprocess(vframes, one_video)

        # * save the video frames keypoint

        # * packe the bbox and mask
        sample_json_info = {
            "video_name": one_video.name,
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

        res[one_video.name] = sample_json_info

        # * step5: save the video frames to json file
        # save_to_json(res, SAVE_PATH, person)  # save the sample info to json file.
        save_to_pt(res, SAVE_PATH, person)  # save the sample info to json file.

        # merge the frame into video
        if parames.YOLO.save:
            merge_frame_to_video(SAVE_PATH, person, one_video.stem)


def merge_frame_to_video(save_path: Path, person: str, video_name: str):
    _save_path = save_path / "vis" / "pose" / person / video_name
    _out_path = save_path / "vis_video" / person

    frames = sorted(list(_save_path.iterdir()), key=lambda x: int(x.stem.split("_")[0]))

    if not _out_path.exists():
        _out_path.mkdir(parents=True, exist_ok=True)

    first_frame = cv2.imread(str(frames[0]))
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        str(_out_path / video_name) + ".mp4", fourcc, 30.0, (width, height)
    )

    for f in tqdm(frames, desc=f"Processing {video_name}", total=len(frames)):
        img = cv2.imread(str(f))
        out.write(img)

    out.release()
    logger.info(f"Video saved to {_out_path / video_name}.mp4")


def save_to_json(sample_info: dict, save_path: Path, person: str) -> None:
    """save the sample info to json file.

    Args:
        sample_info (dict): _description_
        save_path (Path): _description_
        logger (logging): _description_
    """

    for k, v in sample_info.items():
        save_path_with_name = save_path / "json" / person / (k.split(".")[0] + ".json")

        make_folder(save_path_with_name.parent)

        # convert Path to str.
        v["bbox"] = v["bbox"].tolist()  # serialized as list
        v["mask"] = v["mask"].tolist()
        v["keypoint"]["keypoint"] = v["keypoint"]["keypoint"].tolist()
        v["keypoint"]["keypoint_score"] = v["keypoint"]["keypoint_score"].tolist()
        v["depth"] = v["depth"].tolist()

        with open(save_path_with_name, "w") as f:
            json.dump(v, f, indent=4)

        logging.info(f"Save the {v['video_name']} to {save_path_with_name}")


def save_to_pt(sample_info: dict, save_path: Path, person: str) -> None:
    """save the sample info to json file.

    Args:
        sample_info (dict): _description_
        save_path (Path): _description_
        logger (logging): _description_
    """

    for k, v in sample_info.items():
        save_path_with_name = save_path / "pt" / person / (k.split(".")[0] + ".pt")

        make_folder(save_path_with_name.parent)

        torch.save(v, save_path_with_name)

        logging.info(f"Save the {v['video_name']} to {save_path_with_name}")


@hydra.main(config_path="../configs/", config_name="prepare_dataset", version_base=None)
def main(parames):
    """
    main, for the multiprocessing using.

    Args:
        parames (hydra): hydra config.
    """

    for i in range(1, 7):
        process(parames, "run_{}".format(i))


if __name__ == "__main__":
    main()
