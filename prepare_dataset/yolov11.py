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

import torch
import torchvision

import os

import logging
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class MultiPreprocess(torch.nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()

        # load model
        self.yolo_bbox = YOLO(configs.bbox_ckpt)
        self.yolo_pose = YOLO(configs.pose_ckpt)
        self.yolo_mask = YOLO(configs.seg_ckpt)

        self.conf = configs.conf
        self.iou = configs.iou
        self.verbose = configs.verbose
        self.device = configs.device

        self.img_size = configs.img_size
        self.save_path = configs.save_path
        

    def get_YOLO_pose_result(self, frame_batch: np.ndarray):
        """
        get_YOLO_pose_result, from frame_batch, which (t, h, w, c)

        Args:
            frame_batch (np.ndarray): for processed frame batch, (t, h, w, c)

        Returns:
            dict, list: two return value, with one batch keypoint in Dict, and none index in list.
        """

        t, h, w, c = frame_batch.shape

        one_batch_keypoint = {}
        none_index = []
        one_batch_keypoint_score = {}

        with torch.no_grad():
            for frame in range(t):

                results = self.yolo_pose(
                    source=np.ascontiguousarray(frame_batch[frame]),
                    conf=self.conf,
                    iou=self.iou,
                    save_crop=False,
                    classes=0,
                    vid_stride=True,
                    stream=False,
                    verbose=self.verbose,
                    device=self.device,
                )

                for i, r in enumerate(results):
                    # judge if have keypoints.
                    # one_batch_keypoint.append(r.keypoints.data) # 1, 17, 3
                    if list(r.keypoints.xyn.shape) != [1, 17, 2]:
                        none_index.append(frame)
                        one_batch_keypoint[frame] = None
                        one_batch_keypoint_score[frame] = None
                    else:
                        one_batch_keypoint[frame] = r.keypoints.xyn  # 1, 17
                        one_batch_keypoint_score[frame] = r.keypoints.conf  # 1, 17

        return one_batch_keypoint, none_index, one_batch_keypoint_score

    def save_result(self, r, frame):

        # save res to img
        _path = self.save_path + "/" + self.video_name

        if not os.path.exists(_path):
            os.makedirs(_path)

        torchvision.io.write_png(
            torch.tensor(r.plot()).permute(2, 0, 1),
            os.path.join(_path, f"{frame}_keypoint.png"),
        )


    def get_YOLO_mask_result(self, frame_batch: np.ndarray):
        """
        get_YOLO_mask_result, from frame_batch, for mask.

        Args:
            frame_batch (np.ndarry): for processed frame batch, (t, h, w, c)

        Returns:
            dict, list: two return values, with one batch mask in Dict, and none index in list.
        """

        t, h, w, c = frame_batch.shape

        one_batch_mask = {}
        none_index = []

        with torch.no_grad():
            for frame in range(t):
                results = self.yolo_mask(
                    source=frame_batch[frame],
                    conf=self.conf,
                    iou=self.iou,
                    save_crop=False,
                    classes=0,
                    vid_stride=True,
                    stream=False,
                    verbose=self.verbose,
                    device=self.device,
                )

                for i, r in enumerate(results):
                    # judge if have mask.
                    if r.masks is None:
                        none_index.append(frame)
                        one_batch_mask[frame] = None
                    elif list(r.masks.data.shape) == [1, 224, 224]:
                        one_batch_mask[frame] = r.masks.data  # 1, 224, 224
                    else:
                        # when mask > 2, just use the first mask.
                        # ? sometime will get two type for masks.
                        one_batch_mask[frame] = r.masks.data[:1, ...]  # 1, 224, 224

        return one_batch_mask, none_index

    def get_YOLO_bbox_result(self, frame_batch: np.ndarray):
        """
        get_YOLO_mask_result, from frame_batch, for mask.

        Args:
            frame_batch (np.ndarry): for processed frame batch, (t, h, w, c)

        Returns:
            dict, list: two return values, with one batch mask in Dict, and none index in list.
        """

        t, h, w, c = frame_batch.shape

        one_batch_bbox = {}
        none_index = []

        with torch.no_grad():
            for frame in range(t):
                results = self.yolo_bbox(
                    source=frame_batch[frame],
                    conf=self.conf,
                    iou=self.iou,
                    save_crop=False,
                    classes=0,
                    vid_stride=True,
                    stream=False,
                    verbose=self.verbose,
                    device=self.device,
                )

                for i, r in enumerate(results):
                    # judge if have bbox.
                    if r.boxes is None or r.boxes.shape[0] == 0:
                        none_index.append(frame)
                        one_batch_bbox[frame] = None
                    elif list(r.boxes.xywh.shape) == [1, 4]:
                        one_batch_bbox[frame] = r.boxes.xywh  # 1, 4, xywh
                    else:
                        # when mask > 2, just use the first mask.
                        # ? sometime will get two type for bbox.
                        one_batch_bbox[frame] = r.boxes.xywh[:1, ...]  # 1, 4

        return one_batch_bbox, none_index

    def delete_tensor(self, video: torch.tensor, delete_idx: int, next_idx: int):
        """
        delete_tensor, from video, we delete the delete_idx tensor and insert the next_idx tensor.

        Args:
            video (torch.tensor): video tensor for process.
            delete_idx (int): delete tensor index.
            next_idx (int): insert tensor index.

        Returns:
            torch.tensor: deleted and processed video tensor.
        """

        c, t, h, w = video.shape
        left = video[:, :delete_idx, ...]
        right = video[:, delete_idx + 1 :, ...]
        insert = video[:, next_idx, ...].unsqueeze(dim=1)

        ans = torch.cat([left, insert, right], dim=1)

        # check frame
        assert ans.shape[1] == t
        return ans

    def process_none(self, batch: torch.tensor, batch_Dict: dict, none_index: list):
        """
        process_none, where from batch_Dict to instead the None value with next frame tensor (or froward frame tensor).

        Args:
            batch_Dict (dict): batch in Dict, where include the None value when yolo dont work.
            none_index (list): none index list map to batch_Dict, here not use this.

        Returns:
            list: list include the replace value for None value.
        """

        boundary = len(batch_Dict) - 1  # 8
        filter_batch = batch

        for k, v in batch_Dict.items():
            if v == None:
                if (
                    None in list(batch_Dict.values())[k:]
                    and len(set(list(batch_Dict.values())[k:])) == 1
                ):
                    next_idx = k - 1
                else:
                    next_idx = k + 1
                    while batch_Dict[next_idx] == None and next_idx < boundary:
                        next_idx += 1

                batch_Dict[k] = batch_Dict[next_idx]

                # * delete none index from video frames
                # batch b, c, t, h, w
                # filter_batch = torch.cat(
                #     [batch[:, :, :k, ...], batch[:, :, k + 1 :, ...]], dim=2
                # )

                # * copy the next frame to none index
                filter_batch[:, :, k, ...] = batch[:, :, next_idx, ...]

        return list(batch_Dict.values()), filter_batch

    def process_batch(self, batch: torch.Tensor):
        b, c, t, h, w = batch.shape

        # for one batch prepare.
        pred_mask_list = []
        pred_bbox_list = []
        pred_keypoint_list = []
        pred_keypoint_score_list = []
        pred_none_index = []

        for batch_index in range(b):

            # ! now, the ultralytics support torch.tensor type, but here have some strange problem. So had better use numpy type.
            # ! np.ndarray type, HWC format with BGR channels uint8 (0-255).
            # c, t, h, w > t, h, w, c, RGB > BGR
            one_batch_numpy = (
                batch[batch_index, [2, 1, 0], ...]
                .permute(1, 2, 3, 0)
                .to(torch.uint8)
                .numpy()
            )

            # one_batch_numpy = np.resize(
            #     one_batch_numpy, (t, h, w, c)
            # )  # (t, h, w, c) > (t, h, w, c)

            # check shape and dtype in numpy
            assert one_batch_numpy.shape == (t, h, w, c)
            assert one_batch_numpy.dtype == np.uint8

            # * process one batch bbox
            one_batch_bbox_Dict, one_bbox_none_index = self.get_YOLO_bbox_result(
                one_batch_numpy
            )

            # ! notice, if there have none index, we also need copy the next frame to none index.
            one_batch_bbox, filter_batch = self.process_none(
                batch, one_batch_bbox_Dict, one_bbox_none_index
            )

            # * process one batch mask
            one_batch_mask_Dict, one_mask_none_index = self.get_YOLO_mask_result(
                one_batch_numpy
            )
            one_batch_mask, _ = self.process_none(
                batch, one_batch_mask_Dict, one_mask_none_index
            )

            # * process one batch keypoint
            (
                one_batch_keypoint_Dict,
                one_pose_none_index,
                one_batch_keypoint_score_Dict,
            ) = self.get_YOLO_pose_result(one_batch_numpy)
            one_batch_keypoint, _ = self.process_none(
                batch, one_batch_keypoint_Dict, one_pose_none_index
            )
            one_batch_keypoint_score, _ = self.process_none(
                batch, one_batch_keypoint_score_Dict, one_pose_none_index
            )

            pred_bbox_list.append(
                torch.stack(one_batch_bbox, dim=0).squeeze()
            )  # t, cxcywh
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

    def forward(self, batch):

        b, c, t, h, w = batch.shape
        # batch, (b, c, t, h, w)
        # bbox_none_index, (b, t)
        # bbox, (b, t, 4) (cxcywh)
        # mask, (b, 1, t, h, w)
        # keypoint, (b, t, 17, 2)
        # keypoint score, (b, t, 17, 1)
        video, bbox_none_index, bbox, mask, keypoint, keypoint_score = (
            self.process_batch(batch)
        )

        # shape check
        assert video.shape == batch.shape
        assert bbox.shape[0] == b and bbox.shape[1] == t
        assert mask.shape[2] == t and mask.shape[0] == b
        assert keypoint.shape[0] == b and keypoint.shape[1] == t
        assert keypoint_score.shape[0] == b and keypoint_score.shape[1] == t

        return video, bbox_none_index, bbox, mask, keypoint, keypoint_score
