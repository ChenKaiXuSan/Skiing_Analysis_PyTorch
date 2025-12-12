from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np


# ---------------- Utils (你原来的保持不变，只略) ----------------
def foot_from_bbox_xyxy(bbox: np.ndarray) -> np.ndarray:
    bbox = np.asarray(bbox, dtype=np.float64).reshape(-1)
    if bbox.shape[0] != 4:
        raise ValueError(f"bbox must have 4 numbers, got {bbox.shape}")
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) * 0.5, y2], dtype=np.float64)


def image_points_to_bev(uv: np.ndarray, H: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    uv = np.asarray(uv, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)

    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(f"uv shape must be (N,2), got {uv.shape}")
    if H.shape != (3, 3):
        raise ValueError(f"H shape must be (3,3), got {H.shape}")
    if not np.isfinite(H).all():
        raise ValueError("H contains NaN or Inf")

    uv_h = np.concatenate([uv, np.ones((uv.shape[0], 1), dtype=np.float64)], axis=1)
    bev_h = (H @ uv_h.T).T
    z = bev_h[:, 2:3]
    if np.any(np.abs(z) < eps):
        raise RuntimeError("Homogeneous coordinate too close to zero (bad H or points)")
    return bev_h[:, :2] / z


def warp_image_to_bev(
    image: np.ndarray, H: np.ndarray, bev_size: Tuple[int, int]
) -> np.ndarray:
    bev_w, bev_h = bev_size
    return cv2.warpPerspective(image, H, (bev_w, bev_h), flags=cv2.INTER_LINEAR)


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


def hconcat_resize(img_left: np.ndarray, img_right: np.ndarray) -> np.ndarray:
    h = min(img_left.shape[0], img_right.shape[0])

    def resize_by_height(img, target_h):
        scale = target_h / img.shape[0]
        w = int(img.shape[1] * scale)
        return cv2.resize(img, (w, target_h))

    return cv2.hconcat([resize_by_height(img_left, h), resize_by_height(img_right, h)])


def check_homography(H: np.ndarray) -> None:
    if H is None or H.shape != (3, 3):
        raise ValueError("H is invalid")
    if not np.isfinite(H).all():
        raise ValueError("H contains NaN/Inf")
    if abs(np.linalg.det(H)) < 1e-12:
        raise ValueError("H is near-singular")


# ---------------- Config ----------------
@dataclass
class BeVConfig:
    lane_width_m: float = 30.0
    lane_length_m: float = 60.0
    px_per_m: float = 20.0
    margin_x_m: float = 5.0
    margin_y_m: float = 10.0


def make_bev_canvas(cfg: BeVConfig) -> Tuple[Tuple[int, int], np.ndarray]:
    Xmin = -cfg.lane_width_m / 2 - cfg.margin_x_m
    Xmax = +cfg.lane_width_m / 2 + cfg.margin_x_m
    Ymin = 0.0 - cfg.margin_y_m
    Ymax = cfg.lane_length_m + cfg.margin_y_m

    bev_w_px = int(np.ceil((Xmax - Xmin) * cfg.px_per_m))
    bev_h_px = int(np.ceil((Ymax - Ymin) * cfg.px_per_m))

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


# ---------------- Main API ----------------
def make_bev(
    img: np.ndarray,
    bboxes_xyxy: np.ndarray,
    out_dir: Path,
    img_pts: Optional[np.ndarray] = None,
    cfg: BeVConfig = BeVConfig(),
    debug_show: bool = False,
) -> None:
    """
    Args:
        img: (H,W,3) BGR
        bboxes_xyxy: (N,4) or (4,)
        out_dir: output folder
        img_pts: (4,2) 手动标定的地面像素点（near-left, near-right, far-right, far-left）
        cfg: BEV config
        debug_show: 是否imshow（服务器建议False）
    """
    if img is None:
        raise ValueError("img is None")

    out_dir.mkdir(parents=True, exist_ok=True)

    # --------- 标定点默认值（你务必换成真实雪道地面点） ----------
    if img_pts is None:
        img_pts = np.array(
            [
                [0, 1080],
                [1920, 1080],
                [1336, 130],
                [600, 130],
            ],
            dtype=np.float32,
        )
    else:
        img_pts = np.asarray(img_pts, dtype=np.float32)
        if img_pts.shape != (4, 2):
            raise ValueError(f"img_pts must be (4,2), got {img_pts.shape}")

    # --------- 固定 BEV 世界点（米） ----------
    bev_pts_m = np.array(
        [
            [-15.0, 0.0],
            [15.0, 0.0],
            [15.0, 60.0],
            [-15.0, 60.0],
        ],
        dtype=np.float32,
    )

    # --------- 只算一次 H ----------
    H_m, _ = cv2.findHomography(img_pts, bev_pts_m, method=0)
    check_homography(H_m)

    bev_size, S = make_bev_canvas(cfg)
    H_px = S @ H_m

    # --------- 只 warp 一次整图 ----------
    bev_img = warp_image_to_bev(img, H_px, bev_size)

    # 原图调试底图：标定点
    src_base = draw_points(img, img_pts, color=(0, 0, 255))
    if debug_show:
        cv2.imshow("Picked ground points", src_base)

    # --------- 规范 bboxes ----------
    bboxes_xyxy = np.asarray(bboxes_xyxy, dtype=np.float64)
    if bboxes_xyxy.ndim == 1:
        bboxes_xyxy = bboxes_xyxy[None, :]
    if bboxes_xyxy.shape[1] != 4:
        raise ValueError(f"bboxes_xyxy must be (N,4), got {bboxes_xyxy.shape}")

    # --------- 对每个人画点并保存 ----------
    for i, one_bbox in enumerate(bboxes_xyxy):
        # --------- 统一的底图（只拷贝一次） ----------
        dbg_all = src_base.copy()      # 原图 + 标定点
        bev_all = bev_img.copy()       # BEV 底图

        COLORS = [
            (0, 255, 0),    # green
            (0, 0, 255),    # red
            (255, 0, 0),    # blue
            (0, 255, 255),  # yellow
            (255, 0, 255),  # magenta
            (255, 255, 0),  # cyan
        ]

        # --------- 对每个人画点（画到同一张图上） ----------
        for i, one_bbox in enumerate(bboxes_xyxy):
            color = COLORS[i % len(COLORS)]

            # foot point in image
            foot_uv = foot_from_bbox_xyxy(one_bbox)
            fu, fv = foot_uv
            cv2.circle(
                dbg_all,
                (int(round(fu)), int(round(fv))),
                6,
                color,
                -1,
            )
            cv2.putText(
                dbg_all,
                f"P{i}",
                (int(fu) + 8, int(fv) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

            # foot point in BEV
            foot_xy_px = image_points_to_bev(foot_uv[None, :], H_px)[0]
            x, y = foot_xy_px
            cv2.circle(
                bev_all,
                (int(round(x)), int(round(y))),
                6,
                color,
                -1,
            )
            cv2.putText(
                bev_all,
                f"P{i}",
                (int(x) + 8, int(y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        # --------- 并排拼接 ----------
        pair = hconcat_resize(dbg_all, bev_all)

        # --------- 保存 ----------
        cv2.imwrite(str(out_dir / "debug_src_points_all.png"), dbg_all)
        cv2.imwrite(str(out_dir / "bev_result_all.png"), bev_all)
        cv2.imwrite(str(out_dir / "compare_src_bev_all.png"), pair)

        # --------- 可视化 ----------
        if debug_show:
            cv2.imshow("Source | BEV (All Persons)", pair)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

