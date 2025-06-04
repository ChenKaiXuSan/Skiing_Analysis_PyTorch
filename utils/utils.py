"""
File: utils.py
Project: utils
Created Date: 2023-09-03 13:02:25
Author: chenkaixu
-----
Comment:

Have a good code time!
-----
Last Modified: 2023-09-03 13:03:05
Modified By: chenkaixu
-----
HISTORY:
Date 	By 	Comments
------------------------------------------------

"""

import os
import shutil
import json

import cv2
from tqdm import tqdm

from pathlib import Path

import torch
from torchvision.transforms.functional import crop, pad, resize

import logging

logger = logging.getLogger(__name__)


def clip_pad_with_bbox(
    imgs: torch.tensor, boxes: list, img_size: int = 256, bias: int = 10
):
    """
    based torchvision function to crop, pad, resize img.

    clip with the bbox, (x1-bias, y1) and padd with the (gap-bais) in left and right.

    Args:
        imgs (list): imgs with (h, w, c)
        boxes (list): (x1, y1, x2, y2)
        img_size (int, optional): croped img size. Defaults to 256.
        bias (int, optional): the bias of bbox, with the (x1-bias) and (x2+bias). Defaults to 5.

    Returns:
        tensor: (c, t, h, w)
    """
    object_list = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)  # dtype must int for resize, crop function

        box_width = x2 - x1
        box_height = y2 - y1

        width_gap = int(((box_height - box_width) / 2))  # keep int type

        img = imgs  # (h, w, c) to (c, h, w), for pytorch function

        # give a bias for the left and right crop bbox.
        croped_img = crop(
            img,
            top=y1,
            left=(x1 - bias),
            height=box_height,
            width=(box_width + 2 * bias),
        )

        pad_img = pad(croped_img, padding=(width_gap - bias, 0), fill=0)

        resized_img = resize(pad_img, size=(img_size, img_size))

        object_list.append(resized_img)

    return object_list  # c, t, h, w


def del_folder(path, *args):
    """
    delete the folder which path/version

    Args:
        path (str): path
        version (str): version
    """
    if os.path.exists(os.path.join(path, *args)):
        shutil.rmtree(os.path.join(path, *args))


def make_folder(path, *args):
    """
    make folder which path/version

    Args:
        path (str): path
        version (str): version
    """
    if not os.path.exists(os.path.join(path, *args)):
        os.makedirs(os.path.join(path, *args))
        print("success make dir! where: %s " % os.path.join(path, *args))
    else:
        print("The target path already exists! where: %s " % os.path.join(path, *args))


def merge_frame_to_video(
    save_path: Path, person: str, video_name: str, flag: str
) -> None:
    _save_path = save_path / "vis" / flag / person / video_name
    _out_path = save_path / "vis_video" / flag / person

    frames = sorted(list(_save_path.iterdir()), key=lambda x: int(x.stem.split("_")[0]))

    if not _out_path.exists():
        _out_path.mkdir(parents=True, exist_ok=True)

    first_frame = cv2.imread(str(frames[0]))
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        str(_out_path / video_name) + ".mp4", fourcc, 30.0, (width, height)
    )

    for f in tqdm(frames, desc=f"Save {flag}-{video_name}", total=len(frames)):
        img = cv2.imread(str(f))
        out.write(img)

    out.release()

    logger.info(f"Video saved to {_out_path / video_name}.mp4")


def save_to_json(sample_info: dict, save_path: Path, person: str) -> None:
    """save the sample info to json file.

    Args:
        sample_info (dict): _description_
        save_path (Path): _description_
        logger (logging): _description_
    """

    for k, v in sample_info.items():
        save_path_with_name = save_path / "json" / person / (k.split(".")[0] + ".json")

        make_folder(save_path_with_name.parent)

        # convert Path to str.
        v["bbox"] = v["bbox"].tolist()  # serialized as list
        v["mask"] = v["mask"].tolist()
        v["keypoint"]["keypoint"] = v["keypoint"]["keypoint"].tolist()
        v["keypoint"]["keypoint_score"] = v["keypoint"]["keypoint_score"].tolist()
        v["depth"] = v["depth"].tolist()

        with open(save_path_with_name, "w") as f:
            json.dump(v, f, indent=4)

        logger.info(f"Save the {v['video_name']} to {save_path_with_name}")


def save_to_pt(one_video: Path, save_path: Path, pt_info: dict[torch.Tensor]) -> None:
    """save the sample info to json file.

    Args:
        sample_info (dict): _description_
        save_path (Path): _description_
        logger (logging): _description_
    """

    person = one_video.parts[-2]
    video_name = one_video.stem

    save_path_with_name = save_path / "pt" / person / (video_name + ".pt")

    make_folder(save_path_with_name.parent)

    torch.save(pt_info, save_path_with_name)

    logger.info(f"Save the {video_name} to {save_path_with_name}")


def process_none(batch_Dict: dict[torch.Tensor], none_index: list):
    """
    process_none, where from batch_Dict to instead the None value with next frame tensor (or froward frame tensor).

    Args:
        batch_Dict (dict): batch in Dict, where include the None value when yolo dont work.
        none_index (list): none index list map to batch_Dict, here not use this.

    Returns:
        list: list include the replace value for None value.
    """

    boundary = len(batch_Dict) - 1 
    filter_batch = batch_Dict.copy()

    for i in none_index:

        # * if the index is None, we need to replace it with next frame.
        if batch_Dict[i] is None:
            next_idx = i + 1

            if next_idx < boundary:
                filter_batch[i] = batch_Dict[next_idx]
            else:
                filter_batch[i] = batch_Dict[boundary]

    return filter_batch
