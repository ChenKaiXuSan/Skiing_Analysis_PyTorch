#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
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
'''

from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

import torch

import logging
import numpy as np
from PIL import Image
from ultralytics import YOLO

from utils.utils import merge_frame_to_video
from utils.utils import clip_pad_with_bbox

logger = logging.getLogger(__name__)


class YOLOv11Bbox:
    def __init__(self, configs) -> None:
        super().__init__()

        # load model
        self.yolo_bbox = YOLO(configs.YOLO.bbox_ckpt)
        self.tracking = configs.YOLO.tracking

        self.conf = configs.YOLO.conf
        self.iou = configs.YOLO.iou
        self.verbose = configs.YOLO.verbose
        self.device = configs.device

        self.img_size = configs.YOLO.img_size

        self.save = configs.YOLO.save
        self.save_path = Path(configs.extract_dataset.save_path)
        self.batch_size = configs.batch_size

    def save_image(self, r, video_path: Path, flag: str = "pose"):

        # TODO: save the crop img for after process.
        person = video_path.parts[-2]
        video_name = video_path.stem

        _save_path = self.save_path / "vis" / flag / person / video_name
        if not _save_path.exists():
            _save_path.mkdir(parents=True, exist_ok=True)

        for i, res in tqdm(
            enumerate(r), total=len(r), desc=f"Save image-{flag}", leave=False
        ):

            img = Image.fromarray(res.plot())
            img.save(_save_path / f"{i}_{flag}.png")

    def save_crop_image(self, r, video_path: Path, flag: str = "pose"):
        """
        save_crop_image, save the crop image for pose or bbox.

        Args:
            r (list): results from YOLO model.
            video_path (Path): video path for save.
            flag (str, optional): flag for save. Defaults to "pose".
        """

        person = video_path.parts[-2]
        video_name = video_path.stem

        _save_path = self.save_path / "vis" / flag / person / video_name
        if not _save_path.exists():
            _save_path.mkdir(parents=True, exist_ok=True)

        for i, res in tqdm(
            enumerate(r), total=len(r), desc=f"Save crop image-{flag}", leave=False
        ):
            res.crop = clip_pad_with_bbox(
                res.original_img, res.boxes.xyxy[0].cpu().numpy(), self.img_size
            )
            res.save(save_dir=str(_save_path), save_name=f"{i}_{flag}_crop.png")

    def get_YOLO_bbox_result(self, frame_list_bgr: list[np.ndarray], video_path: Path = None):
        """
        get_YOLO_mask_result, from frame_batch, for mask.

        Args:
            frame_batch (np.ndarry): for processed frame batch, (t, h, w, c)

        Returns:
            dict, list: two return values, with one batch mask in Dict, and none index in list.
        """

        if self.tracking:
            track_history = defaultdict(lambda: [])

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

    def save_track_history(self, res_list: list):

        track_history = defaultdict(lambda: [])
        # * save the track history
        for r in res_list:
            if r.boxes and r.boxes.is_track:
                boxes = r.boxes.xywh.cpu()
                track_ids = r.boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append([float(x), float(y)])

        return track_history

    def __call__(self, vframes: torch.Tensor, video_path: Path):

        _video_name = video_path.stem
        _person = video_path.parts[-2]

        _save_path = self.save_path / "vis" / "bbox" / _person / _video_name
        if not _save_path.exists():
            _save_path.mkdir(parents=True, exist_ok=True)
        _save_crop_path = self.save_path / "vis" / "bbox_crop" / _person / _video_name 
        if not _save_crop_path.exists():
            _save_crop_path.mkdir(parents=True, exist_ok=True)

        none_index = []
        batch_bbox = {}
        res_list = []

        vframes_numpy = vframes.numpy()
        vframes_bgr = vframes_numpy[:, :, :, ::-1]
        frame_list_bgr = [ img for img in vframes_bgr]

        # * process bbox
        results = (
            self.get_YOLO_bbox_result(frame_list_bgr)
        )

        for idx, r in tqdm(
            enumerate(results), total=len(frame_list_bgr), desc="YOLO BBox", leave=False
        ):

            # judge if have bbox.
            if r.boxes is None or r.boxes.shape[0] == 0:
                none_index.append(idx)
                batch_bbox[idx] = torch.tensor([])  # empty tensor
            else:
                batch_bbox[idx] = r.boxes.xywh  # n, 4

            r.save(filename=str(_save_path / f"{idx}_bbox.png"))
            r.save_crop(save_dir=str(_save_crop_path), file_name=f"{idx}_bbox_crop.png")

            res_list.append(r)

        # * save the result to img
        if self.save:
            # save the video frames to video file
            merge_frame_to_video(
                self.save_path,
                person=video_path.parts[-2],
                video_name=video_path.stem,
                flag="bbox",
            )

        return batch_bbox, none_index, res_list
