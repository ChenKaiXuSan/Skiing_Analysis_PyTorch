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
import cv2
import logging
import numpy as np
from ultralytics import YOLO

from utils.utils import merge_frame_to_video, process_none

logger = logging.getLogger(__name__)

COCO_SKELETON = [
    (5, 7),
    (7, 9),  # 左臂
    (6, 8),
    (8, 10),  # 右臂
    (5, 6),  # 肩膀连线
    (11, 13),
    (13, 15),  # 左腿
    (12, 14),
    (14, 16),  # 右腿
    (11, 12),  # 髋部
    (5, 11),
    (6, 12),  # 上身到下身
    (0, 1),
    (1, 3),
    (0, 2),
    (2, 4),  # 头部
    (1, 2),  # 左右眼
]


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

    def draw_and_save_keypoints(
        self,
        img_tensor: torch.Tensor,
        keypoints: torch.Tensor,
        save_path: str = "keypoints_output.jpg",
        video_path: Path = None,
        radius: int = 3,
        color: tuple = (0, 255, 0),
    ):

        _video_name = video_path.stem
        _person = video_path.parts[-2]

        # filter save path
        _save_path = save_path / "vis" / "filter_img" / "pose" / _person / _video_name

        if not _save_path.exists():
            _save_path.mkdir(parents=True, exist_ok=True)

        for idx, (img_tensor, kpt) in tqdm(
            enumerate(zip(img_tensor, keypoints)),
            total=len(img_tensor),
            desc="Draw and Save Keypoints",
            leave=False,
        ):

            # 转换为 numpy 图像
            img = img_tensor.cpu().numpy()
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

            # 绘制关键点
            for person_kpts in kpt:
                x, y = person_kpts
                if x > 0 and y > 0:
                    cv2.circle(img, (int(x.item()), int(y.item())), radius, color, -1)

            # 绘制骨架
            for i, j in COCO_SKELETON:
                xi, yi = kpt[i]
                xj, yj = kpt[j]
                if xi > 0 and yi > 0 and xj > 0 and yj > 0:
                    cv2.line(
                        img, (int(xi), int(yi)), (int(xj), int(yj)), (255, 0, 0), 2
                    )

            # 保存图像（注意 RGB 转 BGR）
            _img_save_path = Path(_save_path) / f"{idx}_pose_filter.jpg"
            cv2.imwrite(str(_img_save_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        merge_frame_to_video(save_path, _person, _video_name, "pose", filter=True)

    def __call__(self, vframes: torch.Tensor, video_path: Path):
        _video_name = video_path.stem
        _person = video_path.parts[-2]

        _save_path = self.save_path / "vis" / "img" / "pose" / _person / _video_name
        if not _save_path.exists():
            _save_path.mkdir(parents=True, exist_ok=True)
        _save_crop_path = (
            self.save_path / "vis" / "img" / "pose_crop" / _person / _video_name
        )
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

            if idx == 0 and r.boxes is not None and r.boxes.shape[0] > 0:
                # if the first frame, we just use the first bbox.
                bbox_dict[idx] = r.boxes.xywh[0]
                pose_dict[idx] = r.keypoints.xy[0]
                pose_dict_score[idx] = r.keypoints.conf[0]

            # judge if have bbox.
            elif r.boxes is None or r.boxes.shape[0] == 0:
                none_index.append(idx)
                bbox_dict[idx] = None
                pose_dict[idx] = None
                pose_dict_score[idx] = None

            elif r.boxes.shape[0] == 1:
                # if have only one bbox, we use the first one.
                bbox_dict[idx] = r.boxes.xywh[0]
                pose_dict[idx] = r.keypoints.xy[0]
                pose_dict_score[idx] = r.keypoints.conf[0]

            elif r.boxes.shape[0] > 1:

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
                    pose_dict[idx] = r.keypoints.xy[closest_idx]
                    pose_dict_score[idx] = r.keypoints.conf[closest_idx]

            else:
                ValueError(
                    f"the bbox shape is not correct, idx: {idx}, shape: {r.boxes.shape}"
                )

            r.save(filename=str(_save_path / f"{idx}_pose.png"))
            r.save_crop(save_dir=str(_save_crop_path), file_name=f"{idx}_pose_crop.png")

        # * process none index
        if len(none_index) > 0:
            logger.warning(
                f"the {video_path} has {len(none_index)} frames without pose, please check the results."
            )
            # process none index, where from bbox_dict to instead the None value with next frame tensor (or froward frame tensor).
            pose = process_none(batch_Dict=pose_dict, none_index=none_index)
            pose_score = process_none(batch_Dict=pose_dict_score, none_index=none_index)
            # bbox_dict = process_none(batch_Dict=bbox_dict, none_index=none_index)

        # convert dict to tensor
        pose = torch.stack([pose_dict[k] for k in sorted(pose_dict.keys())], dim=0)
        pose_score = torch.stack(
            [pose_dict_score[k] for k in sorted(pose_dict_score.keys())], dim=0
        )

        # * save the result to img
        if self.save:
            # save the video frames to video file
            merge_frame_to_video(
                self.save_path,
                person=_person,
                video_name=_video_name,
                flag="pose",
                filter=False,
            )

            # filter save path
            self.draw_and_save_keypoints(
                img_tensor=vframes,
                keypoints=pose,
                save_path=self.save_path,
                video_path=video_path,
            )

        return pose, pose_score, none_index, results
