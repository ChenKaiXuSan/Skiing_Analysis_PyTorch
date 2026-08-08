#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/sam3d_body/utils.py
Project: /workspace/code/sam3d_body
Created Date: Thursday December 4th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday December 4th 2025 4:24:51 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

"""
Utility functions for SAM 3D Body demo notebook
"""

from typing import Any, Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from .sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info
from .sam_3d_body.visualization.renderer import Renderer
from .sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def visualize_2d_results(
    img_cv2: np.ndarray, outputs: List[Dict[str, Any]], visualizer: SkeletonVisualizer
) -> List[np.ndarray]:
    """Visualize 2D keypoints and bounding boxes"""
    results = []

    for pid, person_output in enumerate(outputs):
        img_vis = img_cv2.copy()

        # Draw keypoints
        keypoints_2d = person_output["pred_keypoints_2d"]
        keypoints_2d_vis = np.concatenate(
            [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
        )
        img_vis = visualizer.draw_skeleton(img_vis, keypoints_2d_vis)

        # Draw bounding box
        bbox = person_output["bbox"]
        img_vis = cv2.rectangle(
            img_vis,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (0, 255, 0),  # Green color
            2,
        )

        # Add person ID text
        cv2.putText(
            img_vis,
            f"Person {pid}",
            (int(bbox[0]), int(bbox[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        results.append(img_vis)

    return results


def visualize_3d_mesh(
    img_cv2: np.ndarray, outputs: List[Dict[str, Any]], faces: np.ndarray
) -> List[np.ndarray]:
    """Visualize 3D mesh overlaid on image and side view"""
    results = []

    for pid, person_output in enumerate(outputs):
        # Create renderer for this person
        renderer = Renderer(focal_length=person_output["focal_length"], faces=faces)

        # 1. Original image
        img_orig = img_cv2.copy()

        # 2. Mesh overlay on original image
        img_mesh_overlay = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                img_cv2.copy(),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        ).astype(np.uint8)

        # 3. Mesh on white background (front view)
        white_img = np.ones_like(img_cv2) * 255
        img_mesh_white = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                white_img,
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        ).astype(np.uint8)

        # 4. Side view
        img_mesh_side = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                white_img.copy(),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                side_view=True,
            )
            * 255
        ).astype(np.uint8)

        # Combine all views
        combined = np.concatenate(
            [img_orig, img_mesh_overlay, img_mesh_white, img_mesh_side], axis=1
        )
        results.append(combined)

    return results


def display_results_grid(
    images: List[np.ndarray], titles: List[str], figsize_per_image: tuple = (6, 6)
):
    """Display multiple images in a grid"""
    n_images = len(images)
    if n_images == 0:
        print("No images to display")
        return

    # Calculate grid dimensions
    cols = min(3, n_images)  # Max 3 columns
    rows = (n_images + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols, figsize=(figsize_per_image[0] * cols, figsize_per_image[1] * rows)
    )

    # Handle single image case
    if n_images == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = axes.flatten()

    for i, (img, title) in enumerate(zip(images, titles)):
        if len(img.shape) == 3 and img.shape[2] == 3:
            # Convert BGR to RGB if needed
            if img.dtype == np.uint8 and np.mean(img) > 1:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img
        else:
            img_rgb = img

        axes[i].imshow(img_rgb)
        axes[i].set_title(title)
        axes[i].axis("off")

    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def process_image_with_mask(estimator, image_path: str, mask_path: str):
    """
    Process image with external mask input.

    Note: The refactored code requires bboxes to be provided along with masks.
    This function automatically computes bboxes from the mask.
    """
    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask from {mask_path}")

    # Ensure mask is binary (0 or 255)
    mask_binary = (mask > 127).astype(np.uint8) * 255

    print(f"Processing image with external mask: {mask_path}")
    print(f"Mask shape: {mask_binary.shape}, unique values: {np.unique(mask_binary)}")

    # Compute bounding box from mask (required by refactored code)
    # Find all non-zero pixels in the mask
    coords = cv2.findNonZero(mask_binary)
    if coords is None:
        print("Warning: Mask is empty, no objects detected")
        return []

    # Get bounding box from mask contours
    x, y, w, h = cv2.boundingRect(coords)
    bbox = np.array([[x, y, x + w, y + h]], dtype=np.float32)

    print(f"Computed bbox from mask: {bbox[0]}")

    # Process with external mask and computed bbox
    # Note: The mask needs to match the number of bboxes (1 bbox -> 1 mask)
    outputs = estimator.process_one_image(image_path, bboxes=bbox, masks=mask_binary)

    return outputs
