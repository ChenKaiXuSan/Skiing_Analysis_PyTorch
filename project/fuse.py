#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/code/project/fuse.py
Project: /workspace/code/project
Created Date: Friday January 30th 2026
Author: Kaixu Chen
-----
Comment:
Have a good code time :)
-----
'''
from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional, Iterable

EPS = 1e-8


# ---------------------------- Side prior ----------------------------
LEFT_JOINTS = {
    "Upperarm_L", "lowerarm_l", "Hand_L",
    "Thigh_L", "calf_l", "Foot_L",
}

RIGHT_JOINTS = {
    "Upperarm_R", "lowerarm_r", "Hand_R",
    "Thigh_R", "calf_r", "Foot_R",
}


def body_side_bias(
    target_ids: List[int],
    id_to_name: Dict[int, str],
    bias_val: float = 1.0,
) -> np.ndarray:
    """
    b(j): (J,)
      left body joints  -> +bias
      right body joints -> -bias
      center joints     -> 0

    NOTE:
      This is a *soft prior* to favor:
        left camera for left body side
        right camera for right body side
      It MUST be combined with data-driven quality.
    """
    b = np.zeros((len(target_ids),), dtype=np.float64)
    for k, jid in enumerate(target_ids):
        name = id_to_name[jid]
        # robust suffix check
        if name.endswith("_L") or name.endswith("_l"):
            b[k] = +bias_val
        elif name.endswith("_R") or name.endswith("_r"):
            b[k] = -bias_val
    return b


# ---------------------------- IO helpers ----------------------------
def dict_to_array(d: Dict[int, Iterable[float]], target_ids: List[int]) -> np.ndarray:
    """{id: (x,y,...) } -> (J,dim) with NaN for missing."""
    if not d:
        raise ValueError("dict_to_array got empty dict; cannot infer dim.")
    dim = len(next(iter(d.values())))
    arr = np.full((len(target_ids), dim), np.nan, dtype=np.float64)
    for k_i, jid in enumerate(target_ids):
        if jid in d:
            arr[k_i] = np.asarray(d[jid], dtype=np.float64)
    return arr


def array_to_dict(arr: np.ndarray, target_ids: List[int]) -> Dict[int, np.ndarray]:
    """(J,dim) -> {id: np.array(dim)}; only finite rows kept."""
    out: Dict[int, np.ndarray] = {}
    for k_i, jid in enumerate(target_ids):
        if np.all(np.isfinite(arr[k_i])):
            out[jid] = arr[k_i].copy()
    return out


# ---------------------------- Weighting ----------------------------
def softmax2(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """2-way softmax weights from quality scores a,b (shape J,) -> w_a,w_b."""
    m = np.maximum(a, b)
    ea = np.exp(a - m)
    eb = np.exp(b - m)
    s = ea + eb + EPS
    return ea / s, eb / s


# ---------------------------- (Optional) With GT ----------------------------
def compute_q_from_2d_err(
    p2d_dict: Dict[int, Iterable[float]],
    g2d_dict: Dict[int, Iterable[float]],
    target_ids: List[int],
) -> np.ndarray:
    """
    q = -||p2d-gt2d|| ; missing -> very low quality.
    """
    p = dict_to_array(p2d_dict, target_ids)  # (J,2)
    g = dict_to_array(g2d_dict, target_ids)  # (J,2)

    ok = np.all(np.isfinite(p), axis=1) & np.all(np.isfinite(g), axis=1)
    err = np.full((len(target_ids),), np.inf, dtype=np.float64)
    err[ok] = np.linalg.norm(p[ok] - g[ok], axis=1)

    q = -err
    q[~ok] = -1e9
    return q


# ---------------------------- No-GT quality (recommended) ----------------------------
def q_2d_sanity(
    p2d_dict: Optional[Dict[int, Iterable[float]]],
    target_ids: List[int],
    width: int,
    height: int,
) -> np.ndarray:
    """
    Basic 2D sanity without GT:
      - finite
      - in image bounds
    Output: (J,) with 0 for ok, negative penalties for bad/missing.
    """
    q = np.full((len(target_ids),), -50.0, dtype=np.float64)
    if p2d_dict is None:
        return q

    for k, jid in enumerate(target_ids):
        if jid not in p2d_dict:
            continue
        u, v = p2d_dict[jid]
        if np.isfinite(u) and np.isfinite(v) and (0 <= u < width) and (0 <= v < height):
            q[k] = 0.0
        else:
            q[k] = -50.0
    return q


def estimate_bone_median_lengths(
    seq_3d_dicts: List[Dict[int, Iterable[float]]],
    target_ids: List[int],
    edges: List[Tuple[int, int]],
) -> np.ndarray:
    """
    Estimate median bone lengths per edge from a 3D sequence for one view.
    Return: (E,)
    """
    idx = {jid: i for i, jid in enumerate(target_ids)}
    E = len(edges)
    L = []

    for frame in seq_3d_dicts:
        X = dict_to_array(frame, target_ids)  # (J,3)
        lens = np.full((E,), np.nan, dtype=np.float64)
        for e_i, (a, b) in enumerate(edges):
            if a not in idx or b not in idx:
                continue
            ia, ib = idx[a], idx[b]
            if np.all(np.isfinite(X[ia])) and np.all(np.isfinite(X[ib])):
                lens[e_i] = np.linalg.norm(X[ia] - X[ib])
        L.append(lens)

    L = np.asarray(L, dtype=np.float64)  # (T,E)
    med = np.nanmedian(L, axis=0)
    return med


def q_from_bone_deviation(
    frame_3d_dict: Dict[int, Iterable[float]],
    target_ids: List[int],
    edges: List[Tuple[int, int]],
    med_lens: np.ndarray,
) -> np.ndarray:
    """
    q_bone(j) = -mean_{incident edges}( |len(e)-med_len(e)| )
    Output: (J,)
    """
    X = dict_to_array(frame_3d_dict, target_ids)
    idx = {jid: i for i, jid in enumerate(target_ids)}
    J = len(target_ids)
    E = len(edges)

    # incident edge list for each joint index
    inc = [[] for _ in range(J)]
    for e_i, (a, b) in enumerate(edges):
        if e_i >= E or not np.isfinite(med_lens[e_i]):
            continue
        if a in idx and b in idx:
            ia, ib = idx[a], idx[b]
            inc[ia].append((e_i, ia, ib))
            inc[ib].append((e_i, ib, ia))

    q = np.full((J,), -1e9, dtype=np.float64)

    for j in range(J):
        if not np.all(np.isfinite(X[j])):
            continue
        dev_sum = 0.0
        cnt = 0
        for (e_i, ja, jb) in inc[j]:
            if np.all(np.isfinite(X[ja])) and np.all(np.isfinite(X[jb])):
                l = np.linalg.norm(X[ja] - X[jb])
                dev_sum += abs(l - med_lens[e_i])
                cnt += 1
        if cnt == 0:
            q[j] = -100.0  # weak info
        else:
            q[j] = -(dev_sum / (cnt + EPS))
    return q


def q_from_temporal(
    prev_3d_dict: Optional[Dict[int, Iterable[float]]],
    curr_3d_dict: Dict[int, Iterable[float]],
    target_ids: List[int],
    beta: float = 1.0,
) -> np.ndarray:
    """
    q_temp(j) = -beta * ||x_t - x_{t-1}||
    If prev missing: return zeros for finite curr, else -inf.
    """
    curr = dict_to_array(curr_3d_dict, target_ids)
    ok_c = np.all(np.isfinite(curr), axis=1)

    q = np.full((len(target_ids),), -1e9, dtype=np.float64)
    q[ok_c] = 0.0

    if prev_3d_dict is None:
        return q

    prev = dict_to_array(prev_3d_dict, target_ids)
    ok = ok_c & np.all(np.isfinite(prev), axis=1)
    q[ok] = -beta * np.linalg.norm(curr[ok] - prev[ok], axis=1)
    return q


def combine_q(
    q_bone: np.ndarray,
    q_temp: Optional[np.ndarray] = None,
    q_sanity: Optional[np.ndarray] = None,
    w_bone: float = 1.0,
    w_temp: float = 0.3,
    w_san: float = 0.2,
) -> np.ndarray:
    q = w_bone * q_bone
    if q_temp is not None:
        q = q + w_temp * q_temp
    if q_sanity is not None:
        q = q + w_san * q_sanity
    return q


def compute_q_nogt_frame(
    curr_3d: Dict[int, Iterable[float]],
    prev_3d: Optional[Dict[int, Iterable[float]]],
    curr_2d: Optional[Dict[int, Iterable[float]]],
    target_ids: List[int],
    edges: List[Tuple[int, int]],
    med_lens: np.ndarray,
    width: int,
    height: int,
    beta_temp: float = 1.0,
    w_bone: float = 1.0,
    w_temp: float = 0.3,
    w_san: float = 0.2,
) -> np.ndarray:
    """
    No-GT joint-wise quality score.
    """
    q_b = q_from_bone_deviation(curr_3d, target_ids, edges, med_lens)
    q_t = q_from_temporal(prev_3d, curr_3d, target_ids, beta=beta_temp)
    q_s = q_2d_sanity(curr_2d, target_ids, width=width, height=height)
    return combine_q(q_b, q_t, q_s, w_bone=w_bone, w_temp=w_temp, w_san=w_san)


# ---------------------------- Fusion ----------------------------
def fuse_frame_3d(
    p3d_l_dict: Dict[int, Iterable[float]],
    p3d_r_dict: Dict[int, Iterable[float]],
    q_l: np.ndarray,
    q_r: np.ndarray,
    target_ids: List[int],
) -> Dict[int, np.ndarray]:
    """
    Per-joint weighted fusion.
    Inputs 3D dict are in SAME coordinate system.
    q_l/q_r: (J,) quality (higher is better).
    """
    Xl = dict_to_array(p3d_l_dict, target_ids)  # (J,3)
    Xr = dict_to_array(p3d_r_dict, target_ids)  # (J,3)

    # Xl = p3d_l_dict
    # Xr = p3d_r_dict

    ok_l = np.all(np.isfinite(Xl), axis=1)
    ok_r = np.all(np.isfinite(Xr), axis=1)

    wl, wr = softmax2(q_l, q_r)  # (J,)

    fused = np.full_like(Xl, np.nan)
    both = ok_l & ok_r
    fused[both] = (wl[both, None] * Xl[both] + wr[both, None] * Xr[both]) / (wl[both, None] + wr[both, None] + EPS)

    only_l = ok_l & ~ok_r
    fused[only_l] = Xl[only_l]

    only_r = ok_r & ~ok_l
    fused[only_r] = Xr[only_r]

    return array_to_dict(fused, target_ids)

# ---------------------------- Temporal smoothing ----------------------------
def temporal_smooth_ema(
    fused_seq_dicts: List[Dict[int, Iterable[float]]],
    target_ids: List[int],
    alpha: float = 0.7,
) -> List[Dict[int, np.ndarray]]:
    """
    EMA smoothing on fused 3D sequence.
    alpha: larger -> follow current more; smaller -> smoother
    """
    T = len(fused_seq_dicts)
    J = len(target_ids)

    X = np.full((T, J, 3), np.nan, dtype=np.float64)
    for t in range(T):
        X[t] = dict_to_array(fused_seq_dicts[t], target_ids)

    Y = np.full_like(X, np.nan)
    Y[0] = X[0]

    for t in range(1, T):
        xt = X[t]
        yt_prev = Y[t - 1]

        ok_x = np.all(np.isfinite(xt), axis=1)
        ok_prev = np.all(np.isfinite(yt_prev), axis=1)

        both = ok_x & ok_prev
        Y[t, both] = alpha * xt[both] + (1 - alpha) * yt_prev[both]

        miss_x = ~ok_x & ok_prev
        Y[t, miss_x] = yt_prev[miss_x]

        miss_prev = ok_x & ~ok_prev
        Y[t, miss_prev] = xt[miss_prev]

    out: List[Dict[int, np.ndarray]] = []
    for t in range(T):
        out.append(array_to_dict(Y[t], target_ids))
    return out
