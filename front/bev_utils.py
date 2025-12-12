from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

# ==================================================
# Utils
# ==================================================


def foot_from_bbox_xyxy(bbox: np.ndarray) -> np.ndarray:
    """bbox: [x1,y1,x2,y2] -> bottom-center (u,v)"""
    bbox = np.asarray(bbox, dtype=np.float64).reshape(-1)
    if bbox.shape[0] != 4:
        raise ValueError(f"bbox must have 4 numbers, got {bbox.shape}")
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) * 0.5, y2], dtype=np.float64)


def image_points_to_bev(uv: np.ndarray, H: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """(N,2) uv -> (N,2) XY using homography image->BEV."""
    uv = np.asarray(uv, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)

    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(f"uv shape must be (N,2), got {uv.shape}")
    if H.shape != (3, 3):
        raise ValueError(f"H shape must be (3,3), got {H.shape}")
    if not np.isfinite(H).all():
        raise ValueError("H contains NaN/Inf")

    ones = np.ones((uv.shape[0], 1), dtype=np.float64)
    uv_h = np.concatenate([uv, ones], axis=1)  # (N,3)

    bev_h = (H @ uv_h.T).T  # (N,3)
    z = bev_h[:, 2:3]
    if np.any(np.abs(z) < eps):
        raise RuntimeError("Homogeneous coordinate too close to zero (bad H or points)")

    xy = bev_h[:, :2] / z
    return xy


def warp_image_to_bev(
    image: np.ndarray,
    H: np.ndarray,
    bev_size: Tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Warp full image to BEV canvas."""
    if image is None:
        raise ValueError("Input image is None")

    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"H shape must be (3,3), got {H.shape}")

    bev_w, bev_h = bev_size
    return cv2.warpPerspective(image, H, (bev_w, bev_h), flags=interpolation)


def draw_points(
    img: np.ndarray, pts: np.ndarray, color=(0, 0, 255), radius=6
) -> np.ndarray:
    out = img.copy()
    pts = np.asarray(pts, dtype=np.int32)
    for i, (x, y) in enumerate(pts):
        cv2.circle(out, (int(x), int(y)), radius, color, -1)
        cv2.putText(
            out,
            str(i + 1),
            (int(x) + 8, int(y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )
    return out


def check_homography(H: np.ndarray) -> None:
    """Basic sanity checks."""
    if H is None or H.shape != (3, 3):
        raise ValueError("H is invalid")
    if not np.isfinite(H).all():
        raise ValueError("H contains NaN/Inf")
    det = np.linalg.det(H)
    if abs(det) < 1e-12:
        raise ValueError(f"H is near-singular, det={det}")


# ==================================================
# Config
# ==================================================


@dataclass
class BeVConfig:
    # Snow lane size (meters)
    lane_width_m: float = 30.0  # X range: [-15, +15]
    lane_length_m: float = 60.0  # Y range: [0, 60]

    # Canvas scale: pixels per meter
    px_per_m: float = 20.0

    # Extra margin in BEV to avoid cropping near camera / edges
    margin_x_m: float = 5.0
    margin_y_m: float = 10.0  # extend toward camera and far end


def make_bev_canvas(cfg: BeVConfig) -> Tuple[Tuple[int, int], np.ndarray]:
    """
    关键点：你计算出来的 H 是 image->(X,Y)(米).
    但 warpPerspective 需要映射到像素画布(0..Wpx,0..Hpx).
    所以要再乘一个 S，把米坐标变成像素坐标。
    """
    Xmin = -cfg.lane_width_m / 2 - cfg.margin_x_m
    Xmax = +cfg.lane_width_m / 2 + cfg.margin_x_m
    Ymin = 0.0 - cfg.margin_y_m
    Ymax = cfg.lane_length_m + cfg.margin_y_m

    bev_w_px = int(np.ceil((Xmax - Xmin) * cfg.px_per_m))
    bev_h_px = int(np.ceil((Ymax - Ymin) * cfg.px_per_m))

    # S maps meters (X,Y) to pixel (x,y)
    # x = (X - Xmin) * s
    # y = (Ymax - Y) * s   (让 BEV 图像上方是远处，下方是近处，更直观)
    s = cfg.px_per_m
    S = np.array(
        [
            [s, 0, -Xmin * s],
            [0, -s, Ymax * s],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    return (bev_w_px, bev_h_px), S


def hconcat_resize(img_left, img_right):
    """
    将两张图按高度对齐后横向拼接
    """
    h = min(img_left.shape[0], img_right.shape[0])

    def resize_by_height(img, target_h):
        scale = target_h / img.shape[0]
        w = int(img.shape[1] * scale)
        return cv2.resize(img, (w, target_h))

    img_left_r = resize_by_height(img_left, h)
    img_right_r = resize_by_height(img_right, h)

    return cv2.hconcat([img_left_r, img_right_r])


# ==================================================
# Main demo
# ==================================================


def make_bev(img: np.ndarray, bbox: np.ndarray, out_path: Path) -> None:
    cfg = BeVConfig()

    # ----------------------------
    # 0) Load image
    # ----------------------------
    if img is None:
        raise FileNotFoundError("Failed to load ski.png")
    H_img, W_img = img.shape[:2]

    # ----------------------------
    # 1) Your picked pixel points (must be on ground!)
    #    ORDER: near-left, near-right, far-right, far-left
    # ----------------------------
    img_pts = np.array(
        [
            [0, 1080],  # near-left  (⚠️最好换成雪道左边界的地面点)
            [1920, 1080],  # near-right (⚠️最好换成雪道右边界的地面点)
            [1336, 130],  # far-right
            [600, 130],  # far-left
        ],
        dtype=np.float32,
    )

    # Basic bounds check
    if (
        np.any(img_pts[:, 0] < 0)
        or np.any(img_pts[:, 0] > W_img + 5)
        or np.any(img_pts[:, 1] < 0)
        or np.any(img_pts[:, 1] > H_img + 5)
    ):
        print("⚠️ Warning: some img_pts are outside image size. Double-check.")

    # ----------------------------
    # 2) World points in meters (60m x 30m)
    # ----------------------------
    bev_pts_m = np.array(
        [
            [-15.0, 0.0],  # near-left
            [15.0, 0.0],  # near-right
            [15.0, 60.0],  # far-right
            [-15.0, 60.0],  # far-left
        ],
        dtype=np.float32,
    )

    # ----------------------------
    # 3) Homography: image -> meters
    # ----------------------------
    H_m, mask = cv2.findHomography(
        img_pts, bev_pts_m, method=0
    )  # 4点直接解，不用RANSAC
    check_homography(H_m)
    print("H (image -> meters) =\n", H_m)

    # ----------------------------
    # 4) For warping image: need image -> BEV pixel canvas
    #    H_px = S * H_m
    # ----------------------------
    bev_size, S = make_bev_canvas(cfg)
    H_px = S @ H_m
    print("BEV canvas size (w,h) =", bev_size)
    print("Saved: H_meters.npy, H_bev_px.npy")

    # ----------------------------
    # 5) Debug visualize selected points
    # ----------------------------
    dbg = draw_points(img, img_pts, color=(0, 0, 255))
    cv2.imshow("Picked ground points", dbg)

    # ----------------------------
    # 6) Example: bbox -> foot -> meters -> bev pixel
    # ----------------------------
    bbox = bbox
    foot_uv = foot_from_bbox_xyxy(bbox)[None, :]  # (1,2)

    foot_xy_m = image_points_to_bev(foot_uv, H_m)[0]
    print("Foot in meters (X,Y) =", foot_xy_m)

    foot_xy_px = image_points_to_bev(foot_uv, H_px)[0]
    print("Foot in BEV pixel (x,y) =", foot_xy_px)

    # 原图调试图：标定点 + 脚点（绿）
    dbg2 = draw_points(img, img_pts, color=(0, 0, 255))
    fu, fv = foot_uv[0]
    cv2.circle(dbg2, (int(round(fu)), int(round(fv))), 6, (0, 255, 0), -1)

    # ----------------------------
    # 7) Warp image to BEV (sanity check) -> bev_show
    # ----------------------------
    bev_img = warp_image_to_bev(img, H_px, bev_size)

    bev_show = bev_img.copy()
    x, y = foot_xy_px
    cv2.circle(bev_show, (int(round(x)), int(round(y))), 6, (0, 0, 255), -1)

    # ----------------------------
    # Save images
    # ----------------------------
    cv2.imwrite(f"{out_path}/debug_src_points.png", dbg2)
    cv2.imwrite(f"{out_path}/bev_result.png", bev_show)

    # 并排拼接（现在 bev_show 已经存在了）
    pair = hconcat_resize(dbg2, bev_show)
    cv2.imwrite(f"{out_path}/compare_src_bev.png", pair)

