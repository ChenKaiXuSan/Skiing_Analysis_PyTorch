import numpy as np


def _nanmask_xy(X: np.ndarray) -> np.ndarray:
    """valid mask for (N,D): finite on all dims"""
    return np.isfinite(X).all(axis=1)


def fit_weakpersp_3d_to_2d(
    X3d: np.ndarray,  # (N,3)
    U2d: np.ndarray,  # (N,2)
    min_points: int = 8,
):
    """
    Fit u ≈ s * (X @ M) + t, where M is (3,2) with orthonormal columns.
    Returns: s, M(3,2), t(2,), valid_mask_used
    """
        
    X3d = np.asarray(X3d, dtype=np.float64)
    U2d = np.asarray(U2d, dtype=np.float64)

    assert X3d.ndim == 2 and X3d.shape[1] == 3
    assert U2d.ndim == 2 and U2d.shape[1] == 2
    N = X3d.shape[0]
    assert U2d.shape[0] == N

    valid = _nanmask_xy(X3d) & _nanmask_xy(U2d)
    idx = np.where(valid)[0]
    if idx.size < min_points:
        raise ValueError(f"Not enough valid points to fit: {idx.size} < {min_points}")

    X = X3d[idx]
    U = U2d[idx]

    muX = X.mean(axis=0, keepdims=True)
    muU = U.mean(axis=0, keepdims=True)
    Xc = X - muX
    Uc = U - muU

    # Cross-covariance: (3,2)
    C = Xc.T @ Uc  # (3,2)

    # SVD of 3x2: U(3x3), S(2,), Vt(2x2)
    U_svd, S, Vt = np.linalg.svd(C, full_matrices=True)

    # Orthonormal mapping M (3,2)
    M = U_svd[:, :2] @ Vt  # (3,2), columns ~ orthonormal

    # Scale (least squares under orthonormal constraint)
    denom = (Xc**2).sum()
    if denom < 1e-12:
        raise ValueError("Degenerate 3D points (too small variance).")
    s = S.sum() / denom

    # Translation in 2D
    t = (muU - s * (muX @ M)).reshape(
        2,
    )

    return float(s), M, t, valid


def weakpersp_reproj_confidence(
    X3d: np.ndarray,  # (N,3)
    U2d: np.ndarray,  # (N,2)
    sigma_px: float = 12.0,  # 控制“残差→置信度”的软硬程度
    min_points: int = 8,
    eps: float = 1e-12,
):
    """
    Returns:
      conf: (N,) in [0,1]
      err_px: (N,) reprojection residual (pixels)
      Uhat: (N,2) fitted reprojection
      params: dict with s,M,t,valid_used
    """

    # from dict to np.array
    X3d = np.asarray([v for k, v in X3d.items()], dtype=np.float64)
    U2d = np.asarray([v for k, v in U2d.items()], dtype=np.float64)
    
    U2d = np.asarray(U2d, dtype=np.float64)
    s, M, t, valid_used = fit_weakpersp_3d_to_2d(X3d, U2d, min_points=min_points)
    X3d = np.asarray(X3d, dtype=np.float64)
    U2d = np.asarray(U2d, dtype=np.float64)

    Uhat = s * (X3d @ M) + t  # (N,2)

    err = np.full((X3d.shape[0],), np.nan, dtype=np.float64)
    valid_all = _nanmask_xy(U2d) & _nanmask_xy(Uhat)
    d = Uhat[valid_all] - U2d[valid_all]
    err[valid_all] = np.sqrt((d**2).sum(axis=1))

    # confidence: exp(-e^2/(2*sigma^2))
    sig2 = max(float(sigma_px), eps) ** 2
    conf = np.zeros_like(err)
    vv = np.isfinite(err)
    conf[vv] = np.exp(-(err[vv] ** 2) / (2.0 * sig2))
    conf[~vv] = 0.0

    params = {"s": s, "M": M, "t": t, "valid_used": valid_used}
    return conf, err, Uhat, params

import numpy as np

def _normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n

def canonicalize_pose_3d(
    X: np.ndarray,                 # (N,3)
    root_idx: int,                 # e.g., pelvis
    left_hip_idx: int,
    right_hip_idx: int,
    left_shoulder_idx: int,
    right_shoulder_idx: int,
    scale_mode: str = "hip",       # "hip" or "torso"
    eps: float = 1e-9,
):
    """
    Build a canonical coordinate:
      - origin at root (pelvis)
      - x axis: left_hip -> right_hip
      - y axis: mid_hip -> mid_shoulder
      - z axis: x cross y (right-hand)
    Then rotate + scale normalize.

    Returns:
      Xc: (N,3) canonicalized
      R: (3,3) world->canonical rotation
      s: float scale (divided by s)
    """
    X = np.asarray(X, dtype=np.float64)
    assert X.ndim == 2 and X.shape[1] == 3

    # If key joints invalid -> return zeros + low reliability upstream
    key_ids = [root_idx, left_hip_idx, right_hip_idx, left_shoulder_idx, right_shoulder_idx]
    if not np.isfinite(X[key_ids]).all():
        # keep shape, but canonicalization is unreliable
        return np.full_like(X, np.nan), np.full((3,3), np.nan), np.nan

    root = X[root_idx].copy()
    X0 = X - root  # translate

    Lh, Rh = X0[left_hip_idx], X0[right_hip_idx]
    Ls, Rs = X0[left_shoulder_idx], X0[right_shoulder_idx]

    mid_hip = 0.5 * (Lh + Rh)
    mid_sh = 0.5 * (Ls + Rs)

    x_axis = _normalize(Rh - Lh, eps=eps)                # left->right
    y_axis = _normalize(mid_sh - mid_hip, eps=eps)       # hip->shoulder
    z_axis = _normalize(np.cross(x_axis, y_axis), eps=eps)
    # re-orthogonalize y to be perpendicular to x,z
    y_axis = _normalize(np.cross(z_axis, x_axis), eps=eps)

    # Rotation matrix with rows as canonical axes (world->canonical)
    R = np.stack([x_axis, y_axis, z_axis], axis=0)  # (3,3)

    Xr = (R @ X0.T).T  # rotate into canonical

    # scale normalization
    if scale_mode == "hip":
        s = np.linalg.norm(Rh - Lh)
    elif scale_mode == "torso":
        s = np.linalg.norm(mid_sh - mid_hip)
    else:
        raise ValueError("scale_mode must be 'hip' or 'torso'")

    if not np.isfinite(s) or s < eps:
        return np.full_like(X, np.nan), np.full((3,3), np.nan), np.nan

    Xc = Xr / s
    return Xc, R, float(s)

def crossview_consistency_confidence(
    X_a: np.ndarray,   # (N,3) view A 3D
    X_b: np.ndarray,   # (N,3) view B 3D
    *,
    root_idx: int,
    left_hip_idx: int,
    right_hip_idx: int,
    left_shoulder_idx: int,
    right_shoulder_idx: int,
    sigma_3d: float = 0.08,     # canonical空间的阈值（0.05~0.12常见）
    scale_mode: str = "hip",
    eps: float = 1e-12,
):
    """
    Returns:
      conf: (N,) in [0,1]
      dist: (N,) canonical 3D distance
      Xa_c, Xb_c: canonicalized poses
      info: dict with rotations/scales
    """

    # dict to np.array
    X_a = np.asarray([v for k, v in X_a.items()], dtype=np.float64)
    X_b = np.asarray([v for k, v in X_b.items()], dtype=np.float64)
    
    Xa_c, Ra, sa = canonicalize_pose_3d(
        X_a, root_idx, left_hip_idx, right_hip_idx, left_shoulder_idx, right_shoulder_idx,
        scale_mode=scale_mode
    )
    Xb_c, Rb, sb = canonicalize_pose_3d(
        X_b, root_idx, left_hip_idx, right_hip_idx, left_shoulder_idx, right_shoulder_idx,
        scale_mode=scale_mode
    )

    dist = np.full((X_a.shape[0],), np.nan, dtype=np.float64)
    valid = _nanmask_xy(Xa_c) & _nanmask_xy(Xb_c)
    d = Xa_c[valid] - Xb_c[valid]
    dist[valid] = np.sqrt((d**2).sum(axis=1))

    sig2 = max(float(sigma_3d), eps) ** 2
    conf = np.zeros_like(dist)
    vv = np.isfinite(dist)
    conf[vv] = np.exp(-(dist[vv] ** 2) / (2.0 * sig2))
    conf[~vv] = 0.0

    info = {"Ra": Ra, "sa": sa, "Rb": Rb, "sb": sb, "valid": valid}
    return conf, dist, Xa_c, Xb_c, info
