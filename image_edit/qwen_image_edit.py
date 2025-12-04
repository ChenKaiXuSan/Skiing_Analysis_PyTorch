#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/image_edit/qwen_image_edit.py
Project: /workspace/code/image_edit
Created Date: Wednesday December 3rd 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Wednesday December 3rd 2025 5:16:35 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import os
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

from image_edit.qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from image_edit.qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
from image_edit.qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3

import logging
from omegaconf import DictConfig


logger = logging.getLogger(__name__)


class CameraEditor:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        self.model_path = cfg.model
        self.pipe = self.load_model()

        self.device = self.cfg.infer.gpu

        self.MAX_SEED = np.iinfo(np.int32).max

    # ============== 模型加载部分 ==============
    def load_model(self):
        dtype = torch.bfloat16  # maybe bfloat16

        print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
        print("torch.__version__    =", torch.__version__)
        print("torch.version.cuda   =", torch.version.cuda)
        print("cuda available       :", torch.cuda.is_available())
        print("cuda device count    :", torch.cuda.device_count())
        if torch.cuda.is_available():
            print("current device       :", torch.cuda.current_device())
            print(
                "device name          :",
                torch.cuda.get_device_name(torch.cuda.current_device()),
            )

        pipe = QwenImageEditPlusPipeline.from_pretrained(
            self.model_path.qwen_image_edit,
            transformer=QwenImageTransformer2DModel.from_pretrained(
                self.model_path.qwen_image_edit_rapid,
                subfolder="transformer",
                torch_dtype=dtype,
                device_map="auto",  # 如果你只想用 cuda:1，也可以改成 device_map=None 然后 .to(device)
            ),
            torch_dtype=dtype,
        )

        # 加载 LoRA：镜头转换
        pipe.load_lora_weights(
            self.model_path.qwen_image_multiple_angles,
            weight_name="镜头转换.safetensors",
            adapter_name="angles",
        )

        pipe.set_adapters(["angles"], adapter_weights=[1.0])
        pipe.fuse_lora(adapter_names=["angles"], lora_scale=1.25)
        pipe.unload_lora_weights()

        pipe.transformer.__class__ = QwenImageTransformer2DModel
        pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())

        return pipe

    @staticmethod
    def build_camera_prompt(
        rotate_deg: float = 0.0,
        move_forward: float = 0.0,
        vertical_tilt: float = 0.0,
        wideangle: bool = False,
    ) -> str:
        """
        根据控制参数拼出「镜头运动」的 prompt。
        """
        prompt_parts = []

        # 水平旋转
        if rotate_deg != 0:
            direction = "left" if rotate_deg > 0 else "right"
            if direction == "left":
                prompt_parts.append(
                    f"将镜头向左旋转{abs(rotate_deg)}度 Rotate the camera {abs(rotate_deg)} degrees to the left."
                )
            else:
                prompt_parts.append(
                    f"将镜头向右旋转{abs(rotate_deg)}度 Rotate the camera {abs(rotate_deg)} degrees to the right."
                )

        # 向前移动 / 特写
        if move_forward > 5:
            prompt_parts.append("将镜头转为特写镜头 Turn the camera to a close-up.")
        elif move_forward >= 1:
            prompt_parts.append("将镜头向前移动 Move the camera forward.")

        # 俯仰角
        if vertical_tilt <= -1:
            prompt_parts.append(
                "将相机转向鸟瞰视角 Turn the camera to a bird's-eye view."
            )
        elif vertical_tilt >= 1:
            prompt_parts.append(
                "将相机切换到仰视视角 Turn the camera to a worm's-eye view."
            )

        # 广角
        if wideangle:
            prompt_parts.append(
                " 将镜头转为广角镜头 Turn the camera to a wide-angle lens."
            )

        final_prompt = " ".join(prompt_parts).strip()
        return final_prompt if final_prompt else "no camera movement"

    # ============== 推理核心函数 ==============

    def infer_camera_edit(
        self,
        image: Image.Image,
        rotate_deg: float = 0.0,
        move_forward: float = 0.0,
        vertical_tilt: float = 0.0,
        wideangle: bool = False,
        seed: int = 0,
        randomize_seed: bool = True,
        true_guidance_scale: float = 1.0,
        num_inference_steps: int = 4,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> Tuple[Image.Image, int, str]:
        """
        对单张 PIL Image 做镜头角度编辑。

        返回：
            result_img, used_seed, prompt_str
        """
        prompt = self.build_camera_prompt(
            rotate_deg, move_forward, vertical_tilt, wideangle
        )
        print(f"[Prompt] {prompt}")

        if randomize_seed:
            seed = random.randint(0, self.MAX_SEED)
        print(f"[Seed] {seed}")

        generator = torch.Generator(device=self.device).manual_seed(seed)

        if image is None:
            raise ValueError("image 不能为空")

        pil_image = (
            Image.fromarray(np.array(image)).convert("RGB")
            if not isinstance(image, Image.Image)
            else image.convert("RGB")
        )

        # 高宽为空时，可以不传，走模型默认
        h = height if (height is not None and height != 0) else None
        w = width if (width is not None and width != 0) else None

        # 没有镜头变化的话，直接返回原图
        if prompt == "no camera movement":
            return pil_image, seed, prompt

        out = self.pipe(
            image=[pil_image],
            prompt=prompt,
            height=h,
            width=w,
            num_inference_steps=num_inference_steps,
            generator=generator,
            true_cfg_scale=true_guidance_scale,
            num_images_per_prompt=1,
        ).images[0]

        return out, seed, prompt
