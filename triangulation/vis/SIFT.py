#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/triangulation/vis/SIFT.py
Project: /workspace/code/triangulation/vis
Created Date: Wednesday September 3rd 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Wednesday September 3rd 2025 12:00:20 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""
import cv2
import matplotlib
import os

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_sift_keypoints_image(img_gray, keypoints, save_path, figsize=(10, 8)):
    """
    可视化并保存 SIFT 关键点图像

    参数:
        img_gray: np.ndarray, 灰度图像
        keypoints: list of cv2.KeyPoint, SIFT 提取的关键点
        save_path: str, 要保存的文件路径（如 'keypoints.jpg'）
        figsize: tuple, 图像大小（用于 matplotlib）
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    img_kp = cv2.drawKeypoints(
        img_gray,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        color=(0, 255, 0),
    )

    plt.figure(figsize=figsize)
    plt.imshow(img_kp, cmap="gray")
    plt.title("SIFT Keypoints")
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


def save_sift_matches_image(
    img1, img2, kp1, kp2, matches, save_path, top_n=50, figsize=(16, 8)
):
    """
    可视化并保存 SIFT 匹配结果图像

    参数:
        img1, img2: np.ndarray, 原始图像（可为彩色或灰度）
        kp1, kp2: list of cv2.KeyPoint, 图像1和图像2的关键点
        matches: list of cv2.DMatch, 过滤后的匹配结果（如 ratio test 后）
        save_path: str, 要保存的文件路径（如 'matches.jpg'）
        top_n: int, 显示前多少个匹配点
        figsize: tuple, 图像大小
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    img_match = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        matches[:top_n],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    plt.figure(figsize=figsize)
    plt.imshow(img_match)
    plt.title(f"SIFT Matches (Top {top_n})")
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


def visualize_SIFT_matches(img1, img2, kp1, kp2, save_path, frame_num):
    """
    可视化并保存 SIFT 匹配结果图像

    参数:
        img1, img2: np.ndarray, 原始图像（可为彩色或灰度）
        kp1, kp2: list of cv2.KeyPoint, 图像1和图像2的关键点
        matches: list of cv2.DMatch, 过滤后的匹配结果（如 ratio test 后）
        save_path: str, 要保存的文件路径（如 'matches.jpg'）
        top_n: int, 显示前多少个匹配点
        figsize: tuple, 图像大小
    """

    os.makedirs(save_path, exist_ok=True)

    # SIFT 特征提取
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 保存关键点图
    save_sift_keypoints_image(
        img1, kp1, os.path.join(save_path, "left_keypoints", f"{frame_num}.jpg")
    )
    save_sift_keypoints_image(
        img2, kp2, os.path.join(save_path, "right_keypoints", f"{frame_num}.jpg")
    )

    # 匹配与 ratio test
    bf = cv2.BFMatcher()
    matches_all = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches_all:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # 保存匹配图
    save_sift_matches_image(img1, img2, kp1, kp2, good_matches, 
                            os.path.join(save_path, f"sift_matches_{frame_num}.jpg"))
