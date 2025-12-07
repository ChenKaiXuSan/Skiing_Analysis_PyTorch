import numpy as np
from typing import Optional, Tuple, Dict, Union

from .fuse_check import estimate_rigid_umeyama, check_rigid_validity

# sam 3d body 的关键关节索引（你已经给定）
NECK = 69
L_HIP, R_HIP = 9, 10
L_SHO, R_SHO = 5, 6

# 用于估计刚体变换的“躯干点”
TORSO_IDX = [NECK, L_HIP, R_HIP, L_SHO, R_SHO]


def _center_scale_sam(
    X: np.ndarray,
    neck_idx: int = NECK,
    l_hip_idx: int = L_HIP,
    r_hip_idx: int = R_HIP,
) -> Tuple[np.ndarray, float]:
    """
    以 pelvis(左右髋中点) 为原点，按 pelvis–neck 距离归一。

    X: (J,3)
    返回:
        X_norm: 中心化+尺度归一后的坐标
        s     : pelvis–neck 距离（原始尺度）
    """
    X = X.copy()
    # pelvis 用左右髋的中点估计，更通用，也不依赖显式 pelvis 关节
    pelvis = 0.5 * (X[l_hip_idx] + X[r_hip_idx])
    neck = X[neck_idx]

    X -= pelvis
    s = np.linalg.norm(neck - pelvis)
    s = s if s > 1e-8 else 1.0
    X /= s
    return X, s


def _fuse_two(
    L: np.ndarray,
    R_align: np.ndarray,
    tau: Union[float, np.ndarray] = 0.08,
    wL: Optional[np.ndarray] = None,
    wR: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    逐关节融合（加权 + 阈值选择）

    L, R_align: (J,3)
    tau       : 标量或(J,)逐关节阈值（在归一化坐标中，0.06~0.12 常用）
    wL, wR    : (J,) 置信度，可为 None
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

        # 两边都挂了
        if not (L_ok or R_ok):
            out[j] = np.array([np.nan, np.nan, np.nan])
            continue
        # 只有一边有效
        if L_ok and not R_ok:
            out[j] = Lj
            continue
        if R_ok and not L_ok:
            out[j] = Rj
            continue

        # 两边都有效：看距离
        d = np.linalg.norm(Lj - Rj)
        if d > tau[j]:
            # 分歧太大：相信置信度大的那一侧
            out[j] = Lj if wL[j] >= wR[j] else Rj
        else:
            # 距离小：做加权平均
            out[j] = (wL[j] * Lj + wR[j] * Rj) / (wL[j] + wR[j] + 1e-9)
    return out

def rigid_transform_3D(
    target: np.ndarray,          # (J,3) 或 (T,J,3)  左视角
    source: np.ndarray,          # (J,3) 或 (T,J,3)  右视角
    tau: float = 0.08,
    allow_scale: bool = False,
    wL: Optional[Union[np.ndarray, float]] = None,  # (J,) 或 (T,J) 或 None
    wR: Optional[Union[np.ndarray, float]] = None,
    return_diagnostics: bool = True,
    verbose: bool = False,
) -> Tuple[np.ndarray, Optional[Dict]]:
    """
    将左右视点的 sam-3d-body 3D 关键点在无外参条件下融合为一个
    “pelvis 原点 + pelvis–neck 归一”的坐标系。

    支持:
        - 单帧:  (J,3)
        - 序列:  (T,J,3)

    返回:
        fused_seq:  (T,J,3) 或 (J,3)  (与输入维度一致)
        diag     :  诊断信息字典（可选）
    """
    # 统一成 (T,J,3)
    L = np.asarray(target, dtype=float)
    R = np.asarray(source, dtype=float)
    if L.ndim == 2:  # (J,3) -> (1,J,3)
        L = L[None, ...]
    if R.ndim == 2:
        R = R[None, ...]
    assert L.shape == R.shape and L.shape[-1] == 3, "输入形状应一致，且为(*,J,3)"

    T, J, _ = L.shape
    assert max(TORSO_IDX) < J, f"躯干索引超出关节数量: max(TORSO_IDX)={max(TORSO_IDX)}, J={J}"

    fused_seq = np.empty_like(L)

    # ---------- 权重准备 ----------
    def _expand_w(w_in, T: int, J: int) -> np.ndarray:
        if w_in is None:
            return np.ones((T, J), dtype=float)
        w = np.asarray(w_in, dtype=float)
        if w.ndim == 0:
            w = np.full((T, J), float(w))
        elif w.ndim == 1:
            # (J,) -> (T,J)
            assert w.shape[0] == J
            w = np.tile(w[None, :], (T, 1))
        elif w.ndim == 2:
            assert w.shape == (T, J)
        else:
            raise ValueError("wL/wR shape 必须为 (J,), (T,J) 或标量")
        return w

    wL_seq = _expand_w(wL, T=T, J=J)
    wR_seq = _expand_w(wR, T=T, J=J)

    diag = (
        {"per_frame": [], "mean_gain": None, "bad_frames": []}
        if return_diagnostics
        else None
    )

    for t in range(T):
        Lt = L[t].copy()
        Rt = R[t].copy()

        # 1) 分别规范化（pelvis 原点 + pelvis–neck 归一）
        Lt_n, _ = _center_scale_sam(Lt)
        Rt_n, _ = _center_scale_sam(Rt)

        # 2) 用躯干点估计 Rt_n → Lt_n 的相似变换
        torso_L, torso_R = Lt_n[TORSO_IDX], Rt_n[TORSO_IDX]

        R_hat, t_hat, s_hat, info = estimate_rigid_umeyama(
            torso_L, torso_R, allow_scale=allow_scale
        )
        Rt_aligned = s_hat * (R_hat @ Rt_n.T).T + t_hat  # (J,3)

        if verbose:
            print(f"[frame {t}] det(R_hat) = {np.linalg.det(R_hat):.6f}, s={s_hat:.4f}")
            print("t_hat:", t_hat, "reflect_fix:", info.get("reflect_fixed", None))

            report = check_rigid_validity(
                torso_L, torso_R, R_hat, t_hat, allow_scale=allow_scale, tol=1e-6
            )
            for k, v in report.items():
                if k not in ("notes",):
                    print(f"{k:30s}: {v}")
            print("notes:", report.get("notes", ""))

        # 3) 逐关节融合（在归一化坐标中）
        fused = _fuse_two(
            Lt_n,
            Rt_aligned,
            tau=tau,
            wL=wL_seq[t],
            wR=wR_seq[t],
        )

        # 4) 融合后再次统一到 pelvis 原点 + pelvis–neck 归一
        fused, _ = _center_scale_sam(fused)
        fused_seq[t] = fused

        # 5) 诊断信息
        if return_diagnostics:
            lr_before = float(np.linalg.norm(Lt_n - Rt_n, axis=-1).mean())
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
                    "s": float(s_hat),
                    "R": R_hat.copy(),
                    "t": t_hat.copy(),
                }
            )
            if gain < 0:
                diag["bad_frames"].append(t)

    if return_diagnostics:
        gains = [d["gain"] for d in diag["per_frame"]]
        diag["mean_gain"] = float(np.nanmean(gains)) if len(gains) else np.nan
        return (fused_seq if target.ndim == 3 else fused_seq[0]), diag
    else:
        return (fused_seq if target.ndim == 3 else fused_seq[0]), None
