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
import torch.nn.functional as F

import cv2
import logging
import numpy as np
from ultralytics import YOLO

from utils.utils import merge_frame_to_video, process_none

logger = logging.getLogger(__name__)


class YOLOv11Mask:
    def __init__(self, configs) -> None:
        super().__init__()

        # load model
        self.yolo_mask = YOLO(configs.YOLO.seg_ckpt)
        self.tracking = configs.YOLO.tracking

        self.conf = configs.YOLO.conf
        self.iou = configs.YOLO.iou
        self.verbose = configs.YOLO.verbose
        self.img_size = configs.YOLO.img_size

        self.device = configs.device

        self.save = configs.YOLO.save
        self.save_path = Path(configs.extract_dataset.save_path)

    def get_YOLO_mask_result(self, vframes: torch.Tensor):

        vframes_numpy = vframes.numpy()
        vframes_bgr = vframes_numpy[:, :, :, ::-1]
        frame_list_bgr = [img for img in vframes_bgr]

        if self.tracking:
            results = self.yolo_mask.track(
                source=frame_list_bgr,
                conf=self.conf,
                iou=self.iou,
                classes=0,
                stream=True,
                verbose=self.verbose,
                device=self.device,
            )
        else:
            results = self.yolo_mask(
                source=frame_list_bgr,
                conf=self.conf,
                iou=self.iou,
                classes=0,
                stream=True,
                verbose=self.verbose,
                device=self.device,
            )

        return results

    def resize_masks_to_original(self, masks, orig_shape):
        """
        masks: [N, h, w]
        orig_shape: (H, W)
        returns: [N, H, W]
        """

        masks = masks.unsqueeze(0).unsqueeze(0)  # [N, 1, h, w]
        resized = F.interpolate(
            masks, size=orig_shape, mode="bilinear", align_corners=False
        )
        return resized.squeeze(1)  # [N, H, W]

    def draw_and_save_masks(
        self,
        img_tensor: torch.Tensor,
        masks: torch.Tensor,
        save_path: Path,
        video_path: Path = None,
    ):

        _video_name = video_path.stem
        _person = video_path.parts[-2]

        # filter save path
        _save_path = save_path / "vis" / "filter_img" / "mask" / _person / _video_name

        if not _save_path.exists():
            _save_path.mkdir(parents=True, exist_ok=True)

        for i, (
            img_tensor,
            mask,
        ) in tqdm(
            enumerate(zip(img_tensor, masks)),
            total=len(img_tensor),
            desc="Draw and Save Masks",
            leave=False,
        ):

            # 转为 numpy 图像 [H, W, 3]
            img_np = img_tensor.cpu().numpy()
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
            img_color = img_np.copy()

            mask = mask.squeeze(0)  # [H, W]

            # 生成随机颜色并叠加 mask
            binary_mask = (mask > 0.5).float().cpu().numpy().astype(np.uint8)  # [H, W]
            colored_mask = np.zeros_like(img_color, dtype=np.uint8)
            for c in range(3):
                colored_mask[:, :, c] = binary_mask * (0, 255, 0)[2 - c]  # RGB→BGR

            img_color = cv2.addWeighted(img_color, 1.0, colored_mask, 0.5, 0)

            # 保存图像
            _img_save_path = Path(_save_path) / f"{i}_mask_filter.jpg"
            cv2.imwrite(str(_img_save_path), cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR))

        merge_frame_to_video(save_path, _person, _video_name, "mask", filter=True)

    def __call__(self, vframes: torch.Tensor, video_path: Path):

        _video_name = video_path.stem
        _person = video_path.parts[-2]

        _save_path = self.save_path / "vis" / "img" / "mask" / _person / _video_name
        if not _save_path.exists():
            _save_path.mkdir(parents=True, exist_ok=True)
        _save_crop_path = (
            self.save_path / "vis" / "img" / "mask_crop" / _person / _video_name
        )
        if not _save_crop_path.exists():
            _save_crop_path.mkdir(parents=True, exist_ok=True)

        none_index = []
        bbox_dict = {}
        mask_dict = {}

        # * process bbox
        results = self.get_YOLO_mask_result(vframes)

        for idx, r in tqdm(
            enumerate(results), total=len(vframes), desc="YOLO Mask", leave=False
        ):

            if idx == 0 and r.boxes is not None and r.boxes.shape[0] > 0:
                # if the first frame, we just use the first bbox.
                bbox_dict[idx] = r.boxes.xywh[0]
                mask_dict[idx] = self.resize_masks_to_original(
                    r.masks.data[0], r.masks.orig_shape
                )

            # judge if have bbox.
            elif r.boxes is None or r.boxes.shape[0] == 0:
                none_index.append(idx)
                bbox_dict[idx] = None  # empty tensor
                mask_dict[idx] = None  # empty tensor

            elif r.boxes.shape[0] == 1:
                # if have only one bbox, we use the first one.
                bbox_dict[idx] = r.boxes.xywh[0]
                mask_dict[idx] = self.resize_masks_to_original(
                    r.masks.data[0], r.masks.orig_shape
                )

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
                    mask_dict[idx] = self.resize_masks_to_original(
                        r.masks.data[closest_idx], r.masks.orig_shape
                    )

            else:
                ValueError(
                    f"the bbox shape is not correct, idx: {idx}, shape: {r.boxes.shape}"
                )

            r.save(filename=str(_save_path / f"{idx}_mask.png"))
            r.save_crop(save_dir=str(_save_crop_path), file_name=f"{idx}_mask_crop.png")

        # process none index
        if len(none_index) > 0:
            logger.warning(
                f"the {video_path.stem} has {len(none_index)} frames without bbox."
            )
            mask_dict = process_none(mask_dict, none_index)

        # convert dict to tensor
        mask = torch.stack([mask_dict[k] for k in sorted(mask_dict.keys())], dim=0)

        # * save the result to img
        if self.save:
            # save the video frames to video file
            merge_frame_to_video(
                self.save_path,
                person=video_path.parts[-2],
                video_name=video_path.stem,
                flag="mask",
            )

            self.draw_and_save_masks(
                img_tensor=vframes,
                masks=mask,
                save_path=self.save_path,
                video_path=video_path,
            )

        return mask, none_index, results
