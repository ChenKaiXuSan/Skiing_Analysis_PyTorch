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
from collections import defaultdict

import torch

import logging
import numpy as np
from PIL import Image
from ultralytics import YOLO

from utils.utils import merge_frame_to_video
from utils.utils import clip_pad_with_bbox

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

    def process_batch(self, vframes: torch.Tensor, video_path: Path):

        t, h, w, c = vframes.shape

        # for one batch prepare.
        pred_mask_list = []
        pred_bbox_list = []
        pred_keypoint_list = []
        pred_keypoint_score_list = []
        pred_none_index = []

        vframes_numpy = vframes.numpy()

        # * process bbox
        one_batch_bbox_Dict, one_bbox_none_index, one_bbox_res_list = (
            self.get_YOLO_bbox_result(vframes_numpy)
        )

        # ! notice, if there have none index, we also need copy the next frame to none index.
        one_batch_bbox, filter_batch = self.process_none(
            vframes_numpy, one_batch_bbox_Dict, one_bbox_none_index
        )

        # * process mask
        one_batch_mask_Dict, one_mask_none_index, one_mask_res_list = (
            self.get_YOLO_mask_result(vframes_numpy)
        )
        one_batch_mask, _ = self.process_none(
            vframes_numpy, one_batch_mask_Dict, one_mask_none_index
        )

        # * process keypoint
        (
            one_batch_keypoint_Dict,
            one_pose_none_index,
            one_batch_keypoint_score_Dict,
            one_pose_res_list,
        ) = self.get_YOLO_pose_result(vframes_numpy)
        one_batch_keypoint, _ = self.process_none(
            vframes_numpy, one_batch_keypoint_Dict, one_pose_none_index
        )
        one_batch_keypoint_score, _ = self.process_none(
            vframes_numpy, one_batch_keypoint_score_Dict, one_pose_none_index
        )

        # * save the result to img
        if self.save:

            # save the crop image for bbox, mask, pose
            # self.save_crop_image(one_bbox_res_list, video_path, "bbox")
            # self.save_crop_image(one_mask_res_list, video_path, "mask")
            # self.save_crop_image(one_pose_res_list, video_path, "pose")

            # save the image for bbox, mask, pose
            self.save_image(one_bbox_res_list, video_path, "bbox")
            self.save_image(one_mask_res_list, video_path, "mask")
            self.save_image(one_pose_res_list, video_path, "pose")

            # save the video frames to video file
            merge_frame_to_video(
                self.save_path,
                person=video_path.parts[-2],
                video_name=video_path.stem,
                flag="bbox",
            )
            merge_frame_to_video(
                self.save_path,
                person=video_path.parts[-2],
                video_name=video_path.stem,
                flag="mask",
            )
            merge_frame_to_video(
                self.save_path,
                person=video_path.parts[-2],
                video_name=video_path.stem,
                flag="pose",
            )

        track_history = self.save_track_history(one_pose_res_list)

        # TODO: 这里的逻辑可以修改一下，这些操作是不需要的
        pred_bbox_list.append(torch.stack(one_batch_bbox, dim=0).squeeze())  # t, cxcywh
        pred_mask_list.append(torch.stack(one_batch_mask, dim=1))  # c, t, h, w
        pred_keypoint_list.append(
            torch.stack(one_batch_keypoint, dim=0).squeeze()
        )  # t, keypoint, value
        pred_keypoint_score_list.append(
            torch.cat(one_batch_keypoint_score, dim=0).squeeze()
        )  # t, keypoint, value
        pred_none_index.append(one_bbox_none_index)

        # return batch, label, bbox, mask, keypoint
        return (
            filter_batch,  # b, c, t, h, w
            one_bbox_none_index,  # list
            torch.stack(pred_bbox_list, dim=0),  # b, t, h, w
            torch.stack(pred_mask_list, dim=0),  # b, c, t, h, w
            torch.stack(pred_keypoint_list, dim=0),  # b, t, keypoint, value
            torch.stack(pred_keypoint_score_list, dim=0),  # b, t, keypoint, value
        )

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
        pose = torch.stack(
            [pose_dict[k] for k in sorted(pose_dict.keys())], dim=0
        )
        pose_score = torch.stack(
            [pose_dict_score[k] for k in sorted(pose_dict_score.keys())], dim=0
        )

        return pose, pose_score, none_index, results
