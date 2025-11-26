#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
Solve relative camera extrinsics (R, t) from:
- Left/Right 2D keypoints (x2d_L, x2d_R)
- (Possibly noisy) 3D keypoints X3d (world)
Optionally refine with bundle adjustment, and optionally refine 3D with L2 prior.

Inputs (npy/npz):
  - X3d:      (N,3) or (T,J,3)
  - x2d_L:    (N,2) or (T,J,2)
  - x2d_R:    (N,2) or (T,J,2)
  - conf_L:   (N,)  or (T,J)   [optional, default 1]
  - conf_R:   (N,)  or (T,J)   [optional, default 1]
  - K_L:      (3,3) [optional]
  - K_R:      (3,3) [optional]

If K_* not provided, we infer a weak K from the 2D range: fx=fy=max(std_x,std_y)*2, cx,cy=mean.

Usage:
  python solve_rt_from_2d3d.py \
    --X3d X3d.npy --x2d_left x2d_L.npy --x2d_right x2d_R.npy \
    --out out/rt_result.npz \
    --init pnp --refine camera \
    --huber 1.0 --lambda_x 0.0

  # 若没有靠谱3D，可用 essential 初始化（需 K）：
  python solve_rt_from_2d3d.py \
    --X3d X3d.npy --x2d_left x2d_L.npy --x2d_right x2d_R.npy \
    --K_left K_L.npy --K_right K_R.npy \
    --init essential --refine camera

Author: you + ChatGPT
"""

import os
import argparse
import numpy as np
import cv2
from typing import Tuple
from scipy.optimize import least_squares

EPS = 1e-9

# ----------------------- utils -----------------------
def to_Nx(*arrs):
    """Reshape (T,J,dim) -> (N,dim) and return same order."""
    outs = []
    for a in arrs:
        if a is None:
            outs.append(None)
            continue
        a = np.asarray(a)
        if a.ndim == 3:
            T, J, D = a.shape
            a = a.reshape(T * J, D)
        elif a.ndim == 2:
            pass
        else:
            raise ValueError(f"Bad shape {a.shape}")
        outs.append(a)
    return outs

def infer_K_from_2d(x2d: np.ndarray) -> np.ndarray:
    """Heuristic intrinsics from 2D distribution."""
    cx, cy = x2d[:,0].mean(), x2d[:,1].mean()
    sx, sy = x2d[:,0].std() + 1e-6, x2d[:,1].std() + 1e-6
    f = max(sx, sy) * 2.0
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1.0]], dtype=np.float64)
    return K

def rodrigues_to_mat(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec.reshape(3,1))
    return R

def mat_to_rodrigues(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(R)
    return rvec.reshape(-1)

def project_points(X: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, K: np.ndarray) -> np.ndarray:
    """X: (N,3); rvec,tvec: (3,), K:(3,3) -> (N,2)"""
    x, _ = cv2.projectPoints(X, rvec, tvec, K, None)
    return x.reshape(-1, 2)

def robustify_weights(conf: np.ndarray) -> np.ndarray:
    """Map confidences to [0,1], fall back to ones."""
    if conf is None:
        return None
    c = np.asarray(conf).astype(np.float64)
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    c = np.clip(c, 0.0, 1.0)
    return c

def build_mask(X3d, xL, xR, confL=None, confR=None, min_conf=0.0):
    m = np.isfinite(X3d).all(1) & np.isfinite(xL).all(1) & np.isfinite(xR).all(1)
    if confL is not None: m &= (confL >= min_conf)
    if confR is not None: m &= (confR >= min_conf)
    return m

# -------------------- initialization -----------------
def init_pnp(X3d, x2d, K) -> Tuple[np.ndarray, np.ndarray]:
    """EPnP init. Returns rvec(3,), tvec(3,)"""
    ok, rvec, tvec = cv2.solvePnP(X3d, x2d, K, None, flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        # fallback to iterative
        ok, rvec, tvec = cv2.solvePnP(X3d, x2d, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise RuntimeError("solvePnP failed")
    return rvec.reshape(-1), tvec.reshape(-1)

def init_essential(xL, xR, K):
    """Essential matrix init (up-to-scale t). Returns R, t (unit)."""
    E, mask = cv2.findEssentialMat(xL, xR, K, method=cv2.RANSAC, threshold=1.0, prob=0.999)
    if E is None:
        raise RuntimeError("findEssentialMat failed")
    _, R, t, _ = cv2.recoverPose(E, xL, xR, K)
    t = t.reshape(-1)
    return R, t

# ----------------- bundle adjustment -----------------
def pack_params(rL, tL, rR, tR, Xopt=None):
    if Xopt is None:
        return np.concatenate([rL, tL, rR, tR], axis=0)
    else:
        return np.concatenate([rL, tL, rR, tR, Xopt.reshape(-1)], axis=0)

def unpack_params(p, Npoints=None, refine_points=False):
    rL = p[0:3]; tL = p[3:6]
    rR = p[6:9]; tR = p[9:12]
    if refine_points:
        Xflat = p[12:]
        Xopt = Xflat.reshape(Npoints, 3)
        return rL, tL, rR, tR, Xopt
    else:
        return rL, tL, rR, tR, None

def residuals_bundle(params, X0, xL, xR, K_L, K_R,
                     wL=None, wR=None, lambda_x=0.0,
                     refine_points=False, huber=None):
    """Return stacked residuals: [wL*(projL - xL), wR*(projR - xR), sqrt(lambda_x)*(X - X0)]"""
    if refine_points:
        rL, tL, rR, tR, X = unpack_params(params, Npoints=X0.shape[0], refine_points=True)
    else:
        rL, tL, rR, tR, _ = unpack_params(params, refine_points=False)
        X = X0
    # project
    xL_hat = project_points(X, rL, tL, K_L)
    xR_hat = project_points(X, rR, tR, K_R)
    # residuals
    rL_res = (xL_hat - xL).reshape(-1)
    rR_res = (xR_hat - xR).reshape(-1)
    if wL is not None: rL_res *= np.repeat(wL, 2)
    if wR is not None: rR_res *= np.repeat(wR, 2)

    res = [rL_res, rR_res]
    if refine_points and lambda_x > 0.0:
        res_x = np.sqrt(lambda_x) * (X - X0).reshape(-1)
        res.append(res_x)

    res = np.concatenate(res, axis=0)

    if huber is not None and huber > 0:
        # apply Huber-like clipping for reporting; least_squares has loss option, but we keep raw residuals
        pass
    return res

# ------------------------ main -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--X3d", required=True, help="npy/npz path of X3d (N,3) or (T,J,3)")
    ap.add_argument("--x2d_left", required=True, help="npy/npz path of x2d left (N,2) or (T,J,2)")
    ap.add_argument("--x2d_right", required=True, help="npy/npz path of x2d right (N,2) or (T,J,2)")
    ap.add_argument("--conf_left", default=None, help="optional npy/npz (N,) or (T,J)")
    ap.add_argument("--conf_right", default=None, help="optional npy/npz (N,) or (T,J)")
    ap.add_argument("--K_left", default=None, help="optional K_L (3,3)")
    ap.add_argument("--K_right", default=None, help="optional K_R (3,3)")
    ap.add_argument("--init", choices=["pnp","essential"], default="pnp")
    ap.add_argument("--refine", choices=["none","camera","camera_points"], default="camera")
    ap.add_argument("--lambda_x", type=float, default=0.0, help="L2 prior weight for X (when refining points)")
    ap.add_argument("--huber", type=float, default=0.0, help="use 'soft_l1' loss internally if >0")
    ap.add_argument("--min_conf", type=float, default=0.0)
    ap.add_argument("--out", required=True, help="output npz")
    args = ap.parse_args()

    def load_any(p):
        if p is None: return None
        arr = np.load(p)
        if isinstance(arr, np.lib.npyio.NpzFile):
            # take first array
            k0 = list(arr.keys())[0]
            arr = arr[k0]
        return arr

    X3d = load_any(args.X3d)
    x2dL = load_any(args.x2d_left)
    x2dR = load_any(args.x2d_right)
    confL = load_any(args.conf_left)
    confR = load_any(args.conf_right)
    K_L = load_any(args.K_left)
    K_R = load_any(args.K_right)

    X3d, x2dL, x2dR = to_Nx(X3d, x2dL, x2dR)
    if confL is not None or confR is not None:
        confL, confR = to_Nx(confL, confR)

    # build mask & weights
    wL = robustify_weights(confL)
    wR = robustify_weights(confR)
    m = build_mask(X3d, x2dL, x2dR, wL, wR, min_conf=args.min_conf)
    X = X3d[m]
    xL = x2dL[m]
    xR = x2dR[m]
    wL = wL[m] if wL is not None else None
    wR = wR[m] if wR is not None else None

    # intrinsics
    if K_L is None: K_L = infer_K_from_2d(xL)
    if K_R is None: K_R = infer_K_from_2d(xR)

    # -------- init --------
    if args.init == "pnp":
        rL, tL = init_pnp(X, xL, K_L)
        rR, tR = init_pnp(X, xR, K_R)
    else:
        # essential init for R,t; set left as identity in world, solve right rel.
        Rrel, trel = init_essential(xL, xR, K_L)
        rL, tL = np.zeros(3), np.zeros(3)
        rR = mat_to_rodrigues(Rrel)
        tR = trel  # scale unknown; BA will adjust with 3D prior if provided

    # -------- refine (bundle adjust) --------
    refine_points = (args.refine == "camera_points")
    x0 = pack_params(rL, tL, rR, tR, X if refine_points else None)
    loss_type = "soft_l1" if args.huber > 0 else "linear"

    fun = lambda p: residuals_bundle(
        p, X, xL, xR, K_L, K_R, wL=wL, wR=wR,
        lambda_x=args.lambda_x, refine_points=refine_points, huber=args.huber
    )

    res = least_squares(fun, x0, method="trf", loss=loss_type, f_scale=max(args.huber,1.0), verbose=2, max_nfev=200)

    if refine_points:
        rL, tL, rR, tR, X_opt = unpack_params(res.x, Npoints=X.shape[0], refine_points=True)
    else:
        rL, tL, rR, tR, _ = unpack_params(res.x, refine_points=False)
        X_opt = X

    RL = rodrigues_to_mat(rL); RR = rodrigues_to_mat(rR)
    R_rel = RR @ RL.T
    t_rel = (tR.reshape(3,1) - R_rel @ tL.reshape(3,1)).reshape(3)

    # reprojection errors (px)
    xL_hat = project_points(X_opt, rL, tL, K_L)
    xR_hat = project_points(X_opt, rR, tR, K_R)
    errL = np.linalg.norm(xL_hat - xL, axis=1)
    errR = np.linalg.norm(xR_hat - xR, axis=1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(
        args.out,
        RL=RL, tL=tL, RR=RR, tR=tR, R_rel=R_rel, t_rel=t_rel,
        K_L=K_L, K_R=K_R,
        mean_err_L=float(errL.mean()), median_err_L=float(np.median(errL)),
        mean_err_R=float(errR.mean()), median_err_R=float(np.median(errR)),
        success=int(res.success),
        n_points=int(X_opt.shape[0]),
    )
    print("Saved:", args.out)
    print(f"Reproj px: L mean {errL.mean():.2f} / med {np.median(errL):.2f} | "
          f"R mean {errR.mean():.2f} / med {np.median(errR):.2f}")
    print("R_rel:\n", R_rel)
    print("t_rel (up to scale if essential-init):", t_rel)

if __name__ == "__main__":
    main()
