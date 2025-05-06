#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/code/analysis/convert_img_video.py
Project: /workspace/code/analysis
Created Date: Tuesday May 6th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Tuesday May 6th 2025 1:38:34 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''

import os 
import cv2 

from pathlib import Path
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)


if __name__ == "__main__":

    vis_path = Path('/workspace/data/vis')
    vis_video_path = Path('/workspace/data/vis_video')  

    for one_person in vis_path.iterdir():

        for one_video in one_person.iterdir():
            
            output_path = vis_video_path / one_person.name 
            if not output_path.exists():
                os.makedirs(output_path, exist_ok=True)

            frames = sorted(list(one_video.iterdir()), key=lambda x: int(x.stem.split('_')[0]))

            first_frame = cv2.imread(str(frames[0]))
            height, width, _ = first_frame.shape

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path / one_video.name) + '.mp4', fourcc, 30.0, (width, height))
            
            for f in tqdm(frames, desc=f"Processing {one_video.name}", total=len(frames)):
                img = cv2.imread(str(f))
                out.write(img)

            out.release()
            logger.info(f"Video saved to {output_path / one_video.name}.mp4")   

