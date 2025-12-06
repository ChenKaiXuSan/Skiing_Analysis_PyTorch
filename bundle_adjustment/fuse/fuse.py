import numpy as np
from typing import Optional, Tuple, Dict, Union

from VideoPose3D.fuse.fuse_check import estimate_rigid_umeyama, check_rigid_validity

# ---- H36M-17 关节顺序（VideoPose3D 常用）----
# 0 Hip(pelvis), 1 RHip, 2 RKnee, 3 RAnkle,
# 4 LHip, 5 LKnee, 6 LAnkle,
# 7 Spine, 8 Thorax, 9 Neck/Nose, 10 Head,
# 11 LShoulder, 12 LElbow, 13 LWrist,
# 14 RShoulder, 15 RElbow, 16 RWrist
HIP, NECK = 0, 9
L_HIP, R_HIP = 4, 1
L_SHO, R_SHO = 11, 14

TORSO_IDX = [HIP, NECK, L_HIP, R_HIP, L_SHO, R_SHO]


def _center_scale_h36m(X: np.ndarray) -> Tuple[np.ndarray, float]:
    """以 pelvis 为原点，按 pelvis–neck 距离归一。X: (17,3)"""
    X = X.copy()
    pelvis = X[HIP]
    neck = X[NECK]
    X -= pelvis
    s = np.linalg.norm(neck - pelvis)
    s = s if s > 1e-8 else 1.0
    X /= s
    return X, s


def _umeyama(
    X: np.ndarray, Y: np.ndarray, allow_scale: bool = False
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    求 s,R,t 使 s*R*Y + t ≈ X；X,Y: (N,3)
    返回: (s, R(3x3), t(3,))
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    muX, muY = X.mean(0), Y.mean(0)
    Xc, Yc = X - muX, Y - muY
    Sigma = (Yc.T @ Xc) / len(X)
    U, S, Vt = np.linalg.svd(Sigma)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    if allow_scale:
        varY = (Yc**2).sum() / len(Y)
        s = S.sum() / (varY + 1e-9)
    else:
        s = 1.0
    t = muX - s * (R @ muY)
    return s, R, t


def _fuse_two(
    L: np.ndarray,
    R_align: np.ndarray,
    tau: Union[float, np.ndarray] = 0.08,
    wL: Optional[np.ndarray] = None,
    wR: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    逐关节融合（加权+阈值选择）
    L, R_align: (17,3)
    tau: 标量或(17,)逐关节阈值（在归一化坐标中，0.06~0.12常用）
    wL, wR: (17,) 置信度，可为None
    """
    J = L.shape[0]
    if isinstance(tau, (float, int)):
        tau = np.full(J, float(tau))
    if wL is None:
        wL = np.ones(J, dtype=float)
    if wR is None:
        wR = np.ones(J, dtype=float)

    out = np.empty_like(L)
    for j in range(J):
        Lj, Rj = L[j], R_align[j]
        L_ok = np.all(np.isfinite(Lj))
        R_ok = np.all(np.isfinite(Rj))
        if not (L_ok or R_ok):
            out[j] = np.array([np.nan, np.nan, np.nan])
            continue
        if L_ok and not R_ok:
            out[j] = Lj
            continue
        if R_ok and not L_ok:
            out[j] = Rj
            continue

        d = np.linalg.norm(Lj - Rj)
        if d > tau[j]:
            out[j] = Lj if wL[j] >= wR[j] else Rj
        else:
            out[j] = (wL[j] * Lj + wR[j] * Rj) / (wL[j] + wR[j] + 1e-9)
    return out


def fuse_pose_no_extrinsics_h36m(
    left_3d: np.ndarray,  # (17,3) 或 (T,17,3)
    right_3d: np.ndarray,  # (17,3) 或 (T,17,3)
    tau: float = 0.08,
    allow_scale: bool = False,
    mirror_right_x: bool = False,  # 若右视坐标系左右相反，打开这个
    wL: Optional[np.ndarray] = None,  # (17,) 或 (T,17)
    wR: Optional[np.ndarray] = None,  # (17,) 或 (T,17)
    return_diagnostics: bool = True,
) -> Tuple[np.ndarray, Optional[Dict]]:
    """
    将左右视点的 H36M-17 3D 关键点在无外参条件下融合为一个“pelvis 原点 + pelvis–neck 归一化”的坐标系。
    支持序列输入 (T,17,3)。返回 fused 以及可选诊断信息。
    """
    # 统一成 (T,17,3)
    L = np.asarray(left_3d)
    R = np.asarray(right_3d)
    if L.ndim == 2:
        L = L[None, ...]
    if R.ndim == 2:
        R = R[None, ...]
    assert L.shape == R.shape and L.shape[1:] == (17, 3), "输入形状应一致，且为(*,17,3)"

    T = L.shape[0]
    fused_seq = np.empty_like(L)

    # 权重准备
    def _expand_w(w):
        if w is None:
            return np.ones((T, 17), dtype=float)
        w = np.asarray(w)
        if w.ndim == 1:
            w = np.tile(w[None, :], (T, 1))
        return w

    wL_seq = _expand_w(wL)
    wR_seq = _expand_w(wR)

    diag = (
        {"per_frame": [], "mean_gain": None, "bad_frames": []}
        if return_diagnostics
        else None
    )

    for t in range(T):
        Lt = L[t].copy()
        Rt = R[t].copy()

        if mirror_right_x:
            Rt[:, 0] *= -1  # 仅镜像X轴，必要时也可镜像Y/Z，取决于你的坐标定义
            Rt[:, 2 ] *= -1  # flip z axis

        # 1) 分别规范化（pelvis 原点 + pelvis–neck 归一）
        Lt_n, _ = _center_scale_h36m(Lt)
        Rt_n, _ = _center_scale_h36m(Rt)

        # 2) 用躯干点估计 Rt_n → Lt_n 的相似变换（默认不缩放）
        torso_L, torso_R = Lt_n[TORSO_IDX], Rt_n[TORSO_IDX]
        # s, Rm, tvec = _umeyama(torso_L, torso_R, allow_scale=allow_scale)
        R_hat, t_hat, s_hat, info = estimate_rigid_umeyama(
            torso_L, torso_R, allow_scale=allow_scale
        )
        # Rt_aligned = s * (Rm @ Rt_n.T).T + tvec
        Rt_aligned = s_hat * (R_hat @ Rt_n.T).T + t_hat

        print("det(R_hat):", np.linalg.det(R_hat))
        print("t_hat:", t_hat)
        print("s_hat:", s_hat)
        print("reflect_fix:", info["reflect_fixed"])

        # 检查
        report = check_rigid_validity(
            torso_L, torso_R, R_hat, t_hat, allow_scale=False, tol=1e-6
        )
        for k, v in report.items():
            if k not in ("notes",):
                print(f"{k:30s}: {v}")
        print("notes:", report["notes"])

        # 3) 逐关节融合
        fused = _fuse_two(Lt_n, Rt_aligned, tau=tau, wL=wL_seq[t], wR=wR_seq[t])

        # 4) 融合后再次统一到 pelvis 原点 + pelvis–neck 归一（保证骨长统计稳定）
        fused, _ = _center_scale_h36m(fused)
        fused_seq[t] = fused

        # 5) 诊断（可选）
        if return_diagnostics:
            # 融合前 L-R 平均距离（规范化域）
            lr_before = float(np.linalg.norm(Lt_n - Rt_n, axis=-1).mean())
            # 融合后与左右的平均距离
            fl = float(np.linalg.norm(fused - Lt_n, axis=-1).mean())
            fr = float(np.linalg.norm(fused - Rt_n, axis=-1).mean())
            gain = lr_before - 0.5 * (fl + fr)
            diag["per_frame"].append(
                {
                    "frame": t,
                    "LR_before": lr_before,
                    "Fused_vs_L": fl,
                    "Fused_vs_R": fr,
                    "gain": gain,
                    "s": s_hat,
                    "R": R_hat,
                    "t": t_hat,
                }
            )
            if gain < 0:
                diag["bad_frames"].append(t)

    if return_diagnostics:
        gains = [d["gain"] for d in diag["per_frame"]]
        diag["mean_gain"] = float(np.nanmean(gains)) if len(gains) else np.nan
        return fused_seq if left_3d.ndim == 3 else fused_seq[0], diag
    else:
        return fused_seq if left_3d.ndim == 3 else fused_seq[0], None
