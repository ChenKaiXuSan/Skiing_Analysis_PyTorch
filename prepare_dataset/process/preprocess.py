import logging
from pathlib import Path

import torch

from prepare_dataset.model.depth_estimation import DepthEstimator
from prepare_dataset.model.optical_flow import OpticalFlow

from prepare_dataset.process.preprocess_yolo import PreprocessYOLO
from prepare_dataset.process.preprocess_d2 import PreprocessD2

logger = logging.getLogger(__name__)


class Preprocess:
    def __init__(self, config) -> None:
        super(Preprocess, self).__init__()

        self.task = config.task
        logger.info(f"Preprocess task: {self.task}")

        if "depth" in self.task:
            self.depth_estimator = DepthEstimator(config)
        else:
            self.depth_estimator = None

        if "optical_flow" in self.task:
            self.of_model = OpticalFlow(config)
        else:
            self.of_model = None

        if "yolo" in self.task:
            self.yolo_model = PreprocessYOLO(config)
        else:
            self.yolo_model = None

        if "detectron2" in self.task:
            self.d2_model = PreprocessD2(config)
        else:
            self.d2_model = None

    def __call__(self, vframes: torch.Tensor, video_path: Path):

        # * process depth
        if self.depth_estimator:
            depth = self.depth_estimator(vframes, video_path)
        else:
            depth = torch.empty(
                (0, 1, vframes.shape[1], vframes.shape[2]), dtype=torch.float32
            )

        # * process optical flow
        if self.of_model:
            optical_flow = self.of_model(vframes, video_path)
        else:
            optical_flow = torch.empty(
                (0, 2, vframes.shape[1], vframes.shape[2]), dtype=torch.float32
            )

        # * process yolo
        if self.yolo_model:
            (
                yolo_bbox,
                bbox_none_index,
                yolo_mask,
                yolo_keypoints,
                yolo_keypoints_score,
            ) = self.yolo_model(vframes, video_path)
        else:
            yolo_bbox = torch.empty((0, 4), dtype=torch.float32)
            bbox_none_index = []
            yolo_mask = torch.empty(
                (0, vframes.shape[1], vframes.shape[2]), dtype=torch.bool
            )
            yolo_keypoints = torch.empty((0, 17, 3), dtype=torch.float32)
            yolo_keypoints_score = torch.empty((0, 17), dtype=torch.float32)

        # * process detectron2
        if self.d2_model:
            d2_bbox, d2_kpts, d2_kpts_score = self.d2_model(vframes, video_path)
        else:
            d2_bbox = torch.empty((0, 5), dtype=torch.float32)
            d2_kpts = torch.empty((0, 17, 3), dtype=torch.float32)
            d2_kpts_score = torch.empty((0, 17), dtype=torch.float32)

        pt_info = {
            "optical_flow": optical_flow.cpu(),
            "depth": depth.cpu(),
            "none_index": bbox_none_index,
            "YOLO": {
                "bbox": yolo_bbox.cpu(),  # xywh
                "mask": yolo_mask.cpu(),
                "keypoint": yolo_keypoints.cpu(),  # xyn
                "keypoint_score": yolo_keypoints_score.cpu(),
            },
            "detectron2": {
                "bbox": d2_bbox.cpu(),
                "keypoints": d2_kpts.cpu(),
                "keypoints_score": d2_kpts_score.cpu(),
            },
        }

        return pt_info
