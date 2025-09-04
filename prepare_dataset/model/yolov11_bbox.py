#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/prepare_dataset/yolov11 copy.py
Project: /workspace/code/prepare_dataset
Created Date: Tuesday June 3rd 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Tuesday June 3rd 2025 12:40:59 pm
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

from prepare_dataset.utils import merge_frame_to_video, process_none

logger = logging.getLogger(__name__)


class YOLOv11Bbox:
    def __init__(self, configs, person: str) -> None:
        super().__init__()

        # load model
        self.yolo_bbox = YOLO(configs.YOLO.bbox_ckpt)
        self.tracking = configs.YOLO.tracking

        self.conf = configs.YOLO.conf
        self.iou = configs.YOLO.iou
        self.verbose = configs.YOLO.verbose
        self.device = f"cuda:{configs.device}"

        self.img_size = configs.YOLO.img_size

        self.save = configs.YOLO.save
        self.save_path = (
            Path(configs.extract_dataset.save_path) / "vis" / "yolo" / person
        )
        self.batch_size = configs.batch_size

    def get_YOLO_bbox_result(self, vframes: torch.Tensor):

        vframes_numpy = vframes.numpy()
        vframes_bgr = vframes_numpy[:, :, :, ::-1]
        frame_list_bgr = [img for img in vframes_bgr]

        if self.tracking:

            results = self.yolo_bbox.track(
                source=frame_list_bgr,
                conf=self.conf,
                iou=self.iou,
                classes=0,
                stream=True,
                verbose=self.verbose,
                device=self.device,
                # save=True,
                # save_frames=True,
                # save_conf=True,
                # save_crop=True,
                # project=self.save_path / "vis",
                # name="bbox"
            )
        else:

            results = self.yolo_bbox.predict(
                source=frame_list_bgr,
                conf=self.conf,
                iou=self.iou,
                classes=0,
                stream=True,
                verbose=self.verbose,
                device=self.device,
                # save=True,
                # save_frames=True,
                # save_conf=True,
                # save_crop=True,
                # project=self.save_path / "vis",
                # name="bbox"
            )

        return results

    def draw_and_save_boxes(
        self, img_tensor: torch.Tensor, bboxes, save_path: Path, video_path: Path
    ):

        _video_name = video_path.stem

        # filter save path
        _save_path = save_path / "filter_img" / "bbox" / _video_name

        if not _save_path.exists():
            _save_path.mkdir(parents=True, exist_ok=True)

        for i, (img_tensor, xyxy) in tqdm(
            enumerate(zip(img_tensor, bboxes)),
            total=len(img_tensor),
            desc="Draw and Save BBoxes",
            leave=False,
        ):

            # 转换为 numpy 图像（H, W, C）
            img_np = img_tensor.cpu().numpy()
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)

            x1, y1, x2, y2 = map(int, xyxy.tolist())

            cv2.rectangle(
                img_np,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),  # 绿色边框
                2,
            )

            _img_save_path = Path(_save_path) / f"{i}_bbox_filter.jpg"
            # 保存图像
            cv2.imwrite(str(_img_save_path), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

    def __call__(self, vframes: torch.Tensor, video_path: Path):

        _video_name = video_path.stem

        _save_path = self.save_path / "bbox" / _video_name
        if not _save_path.exists():
            _save_path.mkdir(parents=True, exist_ok=True)
        _save_crop_path = self.save_path / "crop_bbox" / _video_name
        if not _save_crop_path.exists():
            _save_crop_path.mkdir(parents=True, exist_ok=True)

        none_index = []
        bbox_dict = {}

        # * process bbox
        results = self.get_YOLO_bbox_result(vframes)

        for idx, r in tqdm(
            enumerate(results), total=len(vframes), desc="YOLO BBox", leave=False
        ):

            # first frame bbox to tracking
            if idx == 0 and r.boxes is not None and r.boxes.shape[0] > 0:
                bbox_dict[idx] = r.boxes.xyxy[0]

            # judge if have bbox.
            elif r.boxes is None or r.boxes.shape[0] == 0:
                none_index.append(idx)
                bbox_dict[idx] = None

            # if have only one bbox, we use the first one.
            # FIXME: if the target lost the bbox, will save the other bbox.
            elif r.boxes.shape[0] == 1:
                bbox_dict[idx] = r.boxes.xyxy[0]

            elif r.boxes.shape[0] > 1:

                # * save the track history
                if r.boxes and r.boxes.is_track:

                    x, y, w, h = bbox_dict[idx - 1]
                    pre_box_center = [x, y]

                    boxes = r.boxes.xyxy.cpu()
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

            else:
                ValueError(
                    f"the bbox shape is not correct, idx: {idx}, shape: {r.boxes.shape}"
                )

            r.save(filename=str(_save_path / f"{idx}_bbox.png"))
            r.save_crop(save_dir=str(_save_crop_path), file_name=f"{idx}_bbox_crop.png")

        # * process none index
        if len(none_index) > 0:
            logger.warning(
                f"the {video_path.stem} has {len(none_index)} frames without bbox."
            )
            bbox_dict = process_none(bbox_dict, none_index)

        # convert bbox_dict to tensor
        bbox = torch.stack(
            [bbox_dict[k] for k in sorted(bbox_dict.keys())],
            dim=0,
        )

        # * save the result to img
        if self.save:
            # save the video frames to video file
            # merge_frame_to_video(
            #     self.save_path,
            #     person=_person,
            #     video_name=_video_name,
            #     flag="bbox",
            #     filter=False,
            # )

            # filter save path
            self.draw_and_save_boxes(
                img_tensor=vframes,
                bboxes=bbox,
                save_path=self.save_path,
                video_path=video_path,
            )

        return bbox, none_index, results
