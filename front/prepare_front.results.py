#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/front/prepare_front.results.py
Project: /workspace/code/front
Created Date: Friday December 12th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Friday December 12th 2025 3:53:39 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from sam3.model_builder import build_sam3_video_predictor


def propagate_in_video(predictor, session_id):
    # we will just propagate from frame 0 to the end of the video
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    return outputs_per_frame


def load_video_frames(video_path: Path):
    # load "video_frames_for_vis" for visualization purposes (they are not used by the model)
    cap = cv2.VideoCapture(video_path)
    video_frames_for_vis = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return video_frames_for_vis


def prepare_front_results(video_path: Path, output_dir: Path):
    """Prepare front dataset results directory structure."""

    gpus_to_use = range(torch.cuda.device_count())
    predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

    video_path = str(video_path)
    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_path,
        )
    )
    session_id = response["session_id"]

    prompt_text_str = "person"
    frame_idx = 0  # add a text prompt on frame 0
    response = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            text=prompt_text_str,
            frame_index=frame_idx,
        )
    )
    outputs_per_frame = propagate_in_video(predictor, session_id)

    video_frames = load_video_frames(video_path)

    # save results
    for frame_idx, outputs in outputs_per_frame.items():
        outputs["frame"] = video_frames[frame_idx]

    np.save(output_dir / "sam3_outputs.npz", outputs_per_frame)

    #
    # finally, close the inference session to free its GPU resources
    # (you may start a new session on another video)
    _ = predictor.handle_request(
        request=dict(
            type="close_session",
            session_id=session_id,
        )
    )

    # after all inference is done, we can shutdown the predictor
    # to free up the multi-GPU process group
    predictor.shutdown()


if __name__ == "__main__":
    base_dir = Path("/workspace/data/front_raw")
    output_dir = Path("/workspace/data/front_sam3_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    for subject_dir in base_dir.iterdir():
        if not subject_dir.is_dir():
            continue
        subject_name = subject_dir.name
        front = subject_dir / "FDR-AX60_1.mp4"

        # Create output directory for the subject
        subject_output_dir = output_dir / subject_name
        subject_output_dir.mkdir(parents=True, exist_ok=True)

        prepare_front_results(front, subject_output_dir)
