# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch
import copy
from pathlib import Path
from VideoPose3D.common.skeleton import Skeleton
from VideoPose3D.common.camera import normalize_screen_coordinates, image_coordinates

h36m_skeleton = Skeleton(
    parents=[
        -1,
        0,
        1,
        2,
        3,
        4,
        0,
        6,
        7,
        8,
        9,
        0,
        11,
        12,
        13,
        14,
        12,
        16,
        17,
        18,
        19,
        20,
        19,
        22,
        12,
        24,
        25,
        26,
        27,
        28,
        27,
        30,
    ],
    joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
    joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31],
)


custom_camera_params = {
    "id": None,
    "res_w": None,  # Pulled from metadata
    "res_h": None,  # Pulled from metadata
    # Dummy camera parameters (taken from Human3.6M), only for visualization purposes
    "azimuth": 70,  # Only used for visualization
    "orientation": [
        0.1407056450843811,
        -0.1500701755285263,
        -0.755240797996521,
        0.6223280429840088,
    ],
    "translation": [1841.1070556640625, 4955.28466796875, 1563.4454345703125],
}


class MocapDataset:
    def __init__(self, fps, skeleton):
        self._skeleton = skeleton
        self._fps = fps
        self._data = None  # Must be filled by subclass
        self._cameras = None  # Must be filled by subclass

    def remove_joints(self, joints_to_remove):
        kept_joints = self._skeleton.remove_joints(joints_to_remove)
        for subject in self._data.keys():
            for action in self._data[subject].keys():
                s = self._data[subject][action]
                if "positions" in s:
                    s["positions"] = s["positions"][:, kept_joints]

    def __getitem__(self, key):
        return self._data[key]

    def subjects(self):
        return self._data.keys()

    def fps(self):
        return self._fps

    def skeleton(self):
        return self._skeleton

    def cameras(self):
        return self._cameras

    def supports_semi_supervised(self):
        # This method can be overridden
        return False


class CustomDataset(MocapDataset):
    def __init__(self, pt_path: Path, remove_static_joints=True):
        one_skeleton = copy.deepcopy(h36m_skeleton) # FIXME: 这里加载一个人的骨架，所以在循环中就越来越少了，需要修改才行
        super().__init__(fps=None, skeleton=one_skeleton)

        # TODO: 这里从pt文件加载
        pt_data = torch.load(pt_path)
        video_name = pt_data["video_name"]

        self._cameras = {}
        self._data = {}

        cam = {}
        cam.update(custom_camera_params) # TODO：这些东西从哪里来的
        cam["orientation"] = np.array(cam["orientation"], dtype="float32")
        cam["translation"] = np.array(cam["translation"], dtype="float32")
        cam["translation"] = cam["translation"] / 1000  # mm to meters

        cam["id"] = video_name
        self.video_name = video_name
        self.video_path = pt_data['video_path']

        H, W = pt_data["img_shape"]
        cam["res_w"] = W
        cam["res_h"] = H

        self._cameras[video_name] = [cam]

        self._data[video_name] = {"custom": {"cameras": cam, "detectron2": pt_data["detectron2"]}}

        if remove_static_joints:
            # Bring the skeleton to 17 joints instead of the original 32
            self.remove_joints(
                [4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31]
            )

            # Rewire shoulders to the correct parents
            self._skeleton._parents[11] = 8
            self._skeleton._parents[14] = 8

    def supports_semi_supervised(self):
        return False
    
    def get_video_name(self):
        return self.video_name

    def get_video_path(self):
        return self.video_path    
