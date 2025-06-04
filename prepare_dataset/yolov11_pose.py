#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/Skiing_Analysis_PyTorch/preprocess/yolov8.py
Project: /workspace/code/Skiing_Analysis_PyTorch/preprocess
Created Date: Wednesday April 23rd 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday April 24th 2025 4:30:44 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from tqdm import tqdm
from pathlib import Path

import torch

import logging
import numpy as np
from ultralytics import YOLO

from utils.utils import merge_frame_to_video

logger = logging.getLogger(__name__)


class YOLOv11Pose:
    def __init__(self, configs) -> None:
        super().__init__()

        # load model
        self.yolo_pose = YOLO(configs.YOLO.pose_ckpt)
        self.tracking = configs.YOLO.tracking

        self.conf = configs.YOLO.conf
        self.iou = configs.YOLO.iou
        self.verbose = configs.YOLO.verbose
        self.device = configs.device

        self.img_size = configs.YOLO.img_size

        self.save = configs.YOLO.save
        self.save_path = Path(configs.extract_dataset.save_path)
        self.batch_size = configs.batch_size

    def get_YOLO_pose_result(self, vframes: torch.Tensor):

        vframes_numpy = vframes.numpy()
        vframes_bgr = vframes_numpy[:, :, :, ::-1]
        frame_list_bgr = [img for img in vframes_bgr]

        if self.tracking:
            results = self.yolo_pose.track(
                source=frame_list_bgr,
                conf=self.conf,
                iou=self.iou,
                classes=0,
                stream=True,
                verbose=self.verbose,
                device=self.device,
            )
        else:
            results = self.yolo_pose(
                source=frame_list_bgr,
                conf=self.conf,
                iou=self.iou,
                classes=0,
                stream=True,
                verbose=self.verbose,
                device=self.device,
            )

        return results

    def __call__(self, vframes: torch.Tensor, video_path: Path):
        _video_name = video_path.stem
        _person = video_path.parts[-2]

        _save_path = self.save_path / "vis" / "pose" / _person / _video_name
        if not _save_path.exists():
            _save_path.mkdir(parents=True, exist_ok=True)
        _save_crop_path = self.save_path / "vis" / "pose_crop" / _person / _video_name
        if not _save_crop_path.exists():
            _save_crop_path.mkdir(parents=True, exist_ok=True)

        none_index = []
        bbox_dict = {}
        pose_dict = {}
        pose_dict_score = {}

        # * process bbox
        results = self.get_YOLO_pose_result(vframes)

        for idx, r in tqdm(
            enumerate(results), total=len(vframes), desc="YOLO Pose", leave=False
        ):
            # judge if have bbox.
            if r.boxes is None or r.boxes.shape[0] == 0:
                none_index.append(idx)
                bbox_dict[idx] = torch.tensor([])  # empty tensor
                pose_dict[idx] = torch.tensor([])  # empty tensor
                pose_dict_score[idx] = torch.tensor([])  # empty tensor

            elif r.boxes.shape[0] == 1:
                # if have only one bbox, we use the first one.
                bbox_dict[idx] = r.boxes.xywh[0]
                pose_dict[idx] = r.keypoints.xyn[0] if r.keypoints else torch.tensor([])
                pose_dict_score[idx] = (
                    r.keypoints.conf[0] if r.keypoints else torch.tensor([])
                )

            elif r.boxes.shape[0] > 1:
                if idx == 0:
                    # if the first frame, we just use the first bbox.
                    bbox_dict[idx] = r.boxes.xywh[0]
                    pose_dict[idx] = (
                        r.keypoints.xyn[0] if r.keypoints else torch.tensor([])
                    )
                    pose_dict_score[idx] = (
                        r.keypoints.conf[0] if r.keypoints else torch.tensor([])
                    )

                    continue

                # * save the track history
                if r.boxes and r.boxes.is_track:
                    x, y, w, h = bbox_dict[idx - 1]
                    pre_box_center = [x, y]

                    boxes = r.boxes.xywh.cpu()
                    track_ids = r.boxes.id.int().cpu().tolist()

                    distance_list = []

                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box

                        distance_list.append(
                            torch.norm(
                                torch.tensor([x, y]) - torch.tensor(pre_box_center)
                            )
                        )

                    # find the closest bbox to the previous bbox
                    closest_idx = np.argmin(distance_list)
                    closest_box = boxes[closest_idx]
                    bbox_dict[idx] = closest_box
                    pose_dict[idx] = (
                        r.keypoints.xyn[closest_idx]
                        if r.keypoints
                        else torch.tensor([])
                    )
                    pose_dict_score[idx] = (
                        r.keypoints.conf[closest_idx]
                        if r.keypoints
                        else torch.tensor([])
                    )

            else:
                ValueError(
                    f"the bbox shape is not correct, idx: {idx}, shape: {r.boxes.shape}"
                )

            r.save(filename=str(_save_path / f"{idx}_pose.png"))
            r.save_crop(save_dir=str(_save_crop_path), file_name=f"{idx}_pose_crop.png")

        # * save the result to img
        if self.save:
            # save the video frames to video file
            merge_frame_to_video(
                self.save_path,
                person=video_path.parts[-2],
                video_name=video_path.stem,
                flag="pose",
            )

        # convert dict to tensor
        pose = torch.stack([pose_dict[k] for k in sorted(pose_dict.keys())], dim=0)
        pose_score = torch.stack(
            [pose_dict_score[k] for k in sorted(pose_dict_score.keys())], dim=0
        )

        return pose, pose_score, none_index, results
