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
from torchvision.io import read_video
 
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
        bbox, bbox_none_idx, bbox_res_list = preprocess.yolo_model_bbox(vframes, one_video)

        logger.info(f"Finish the bbox detection for {one_video.name}")

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
