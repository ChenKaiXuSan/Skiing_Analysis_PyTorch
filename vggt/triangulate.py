import os
import numpy as np
import logging

logger = logging.getLogger(__name__)

from vggt.reproject import reproject_and_visualize
from vggt.vis.pose_visualization import visualize_3d_joints


# ------------------- 基础函数 ------------------- #
def make_P(K, R, t):
    """K: (3,3), R: (3,3), t: (3,) -> P: (3,4)"""
    Rt = np.concatenate([R, t.reshape(3, 1)], axis=1)
    return K @ Rt


def triangulate_point(P1, P2, x1, x2):
    """线性三角测量 (DLT)"""
    u1, v1 = x1
    u2, v2 = x2
    A = np.stack([
        u1 * P1[2] - P1[0],
        v1 * P1[2] - P1[1],
        u2 * P2[2] - P2[0],
        v2 * P2[2] - P2[1],
    ], axis=0)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return (X / X[3])[:3]


# ------------------- 单帧三角测量主函数 ------------------- #
def triangulate_one_frame(
    K, R, T,
    kptL, kptR,
    frame_L=None, frame_R=None,
    save_dir=None,
    dist=None,
    visualize_3d=False,
    reproject_check=False,
):
    """
    单帧三角测量
    K: (2,3,3)
    R: (2,3,3)
    T: (2,3)
    kptL, kptR: (J,2)
    """

    assert K.shape == (2,3,3)
    assert R.shape == (2,3,3)
    assert T.shape == (2,3)
    H, W, C = frame_L.shape

    P1 = make_P(K[0], R[0], T[0])
    P2 = make_P(K[1], R[1], T[1])

    J = kptL.shape[0]
    X3d = np.zeros((J,3), dtype=np.float32)

    for j in range(J):
        X3d[j] = triangulate_point(P1, P2, kptL[j], kptR[j])

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "triangulated_3d.npy"), X3d)
        logger.info(f"[3D Saved] triangulated_3d.npy | shape={X3d.shape}")

    # ---- 可视化 3D ---- #
    if visualize_3d and save_dir:
        out = os.path.join(save_dir, "3d_joints.png")
        visualize_3d_joints(
            R=R,
            T=T,
            K=K,
            joints_3d=X3d,
            save_path=out,
            title=f"3D Triangulated Result",
            image_size=(W, H),
        )
        logger.info("[Visualization] 3D joints rendered.")

    # ---- 重投影误差 ---- #
    if reproject_check and frame_L is not None and frame_R is not None:
        out = os.path.join(save_dir, "reprojection.jpg")
        res = reproject_and_visualize(
            img1=frame_L,
            img2=frame_R,
            X3=X3d,
            kptL=kptL,
            kptR=kptR,
            K1=K[0], K2=K[1],
            dist1=dist, dist2=dist,
            R=R, T=T,
            out_path=out,
        )
        logger.info(
            f"[Reproj] L={res['mean_err_L']:.2f}px  "
            f"R={res['mean_err_R']:.2f}px | saved to {out}"
        )

    return X3d
