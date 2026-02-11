import numpy as np
import json
import os

# --- 設定 ---
UNITY_MHR70_MAPPING = {
    1: "Bone_Eye_L",
    2: "Bone_Eye_R",
    5: "Upperarm_L",
    6: "Upperarm_R",
    7: "lowerarm_l",
    8: "lowerarm_r",
    9: "Thigh_L",
    10: "Thigh_R",
    11: "calf_l",
    12: "calf_r",
    13: "Foot_L",
    14: "Foot_R",
    41: "Hand_R",
    62: "Hand_L",
    69: "neck_01",
}
TARGET_IDS = list(UNITY_MHR70_MAPPING.keys())

# --- 座標変換関数 ---


def get_unity_gt_dicts(gt_2d_raw, gt_3d_raw, height=1080):
    """Unityの1フレーム分のGTデータを辞書形式に整理"""
    name_to_id = {v: k for k, v in UNITY_MHR70_MAPPING.items()}

    # 2D GT (Pixel coords)
    gt_2d = {
        name_to_id[item["name"]]: np.array(
            [float(item["u"]), height - float(item["v"])]
        )
        for item in gt_2d_raw.get("joints2d", [])
        if item["name"] in name_to_id
    }

    # 3D GT (Unity to Sam3D axis conversion)
    gt_3d = {
        name_to_id[item["name"]]: np.array(
            [-float(item["z"]), -float(item["y"]), float(item["x"])]
        )
        for item in gt_3d_raw.get("joints3d", [])
        if item["name"] in name_to_id
    }

    return gt_2d, gt_3d


def get_sam_pred_dicts(sam_frame):
    """Sam3Dの1フレーム分の予測データを辞書形式に整理"""
    # 2D Prediction
    pred_2d = sam_frame["pred_keypoints_2d"][TARGET_IDS]

    # 3D Prediction

    pred_3d = sam_frame["pred_keypoints_3d"][TARGET_IDS]

    return pred_2d, pred_3d


# --- 評価指標 ---
def calculate_mpjpe(pred_dict, gt_dict):
    """共通ID間の平均欧州距離を計算（タプル/リストにも対応）"""
    errors = []
    common_ids = set(pred_dict.keys()) & set(gt_dict.keys())

    for m_id in common_ids:
        # np.array() で囲むことで、タプル同士の引き算エラーを回避
        p = np.array(pred_dict[m_id])
        g = np.array(gt_dict[m_id])

        err = np.linalg.norm(p - g)
        errors.append(err)

    return np.mean(errors) if errors else np.nan


def calculate_per_joint_errors(pred_dict, gt_dict):
    """
    Returns:
      per_joint_err: dict[m_id] = float(err)
      common_ids: sorted list of ids used
    """
    per_joint_err = {}
    common_ids = sorted(set(pred_dict.keys()) & set(gt_dict.keys()))
    for m_id in common_ids:
        p = np.asarray(pred_dict[m_id], dtype=np.float64)
        g = np.asarray(gt_dict[m_id], dtype=np.float64)
        if not (np.isfinite(p).all() and np.isfinite(g).all()):
            per_joint_err[m_id] = np.nan
            continue
        per_joint_err[m_id] = float(np.linalg.norm(p - g))
    return per_joint_err, common_ids


def init_joint_stat_container(target_ids):
    return {m_id: [] for m_id in target_ids}


def accumulate_joint_errors(stat_container, per_joint_err):
    for m_id, e in per_joint_err.items():
        stat_container.setdefault(m_id, []).append(e)


def summarize_joint_errors(stat_container):
    """
    Returns:
      summary: dict[m_id] = {"mean":..., "std":..., "median":..., "n":...}
    """
    summary = {}
    for m_id, arr in stat_container.items():
        a = np.asarray(arr, dtype=np.float64)
        a = a[np.isfinite(a)]
        if a.size == 0:
            summary[m_id] = {"mean": np.nan, "std": np.nan, "median": np.nan, "n": 0}
        else:
            summary[m_id] = {
                "mean": float(np.mean(a)),
                "std": float(np.std(a)),
                "median": float(np.median(a)),
                "n": int(a.size),
            }
    return summary


def print_joint_error_table(summary, mapping_dict, title, unit=""):
    """
    mapping_dict: {m_id: name}
    """
    print(f"\n--- {title} ---")
    print(
        f"{'ID':>4}  {'Joint':<16}  {'mean':>10}  {'std':>10}  {'median':>10}  {'n':>4}"
    )
    print("-" * 62)
    for m_id in sorted(summary.keys()):
        name = mapping_dict.get(m_id, "Unknown")
        s = summary[m_id]
        mean = s["mean"]
        std = s["std"]
        med = s["median"]
        n = s["n"]
        print(
            f"{m_id:>4}  {name:<16}  {mean:>10.3f}{unit:<1}  {std:>10.3f}{unit:<1}  {med:>10.3f}{unit:<1}  {n:>4}"
        )


def harmonize_to_pixel_coords(
    sam3d_raw,
    unity_raw,
    mapping_dict,
    target_ids,
    width=1920,
    height=1080,
    scale_x=1.0,
    scale_y=1.0,
):
    """
    Sam3D(ピクセル座標)とUnity(負のピクセル座標)を共通のピクセル座標系に統一する
    2d版本
    """
    unified_sam3d = {}
    unified_unity = {}

    # --- 1. Sam3Dデータの統一 (型を float に整理) ---
    sam3d_coords = np.atleast_2d(sam3d_raw.squeeze())
    for i, m_id in enumerate(target_ids):
        if i < len(sam3d_coords):
            # np.float32 を標準の float にキャストしてタプル化
            unified_sam3d[m_id] = (float(sam3d_coords[i][0]), float(sam3d_coords[i][1]))

    # --- 2. Unityデータの統一 (V軸の負の値を補正) ---
    name_to_id = {name: m_id for m_id, name in mapping_dict.items()}
    joints_list = (
        unity_raw.get("joints2d", unity_raw)
        if isinstance(unity_raw, dict)
        else unity_raw
    )

    for item in joints_list:
        m_id = name_to_id.get(item["name"])
        if m_id in target_ids:
            # Unityデータがピクセル値で、かつVが負の値(-461.9など)の場合
            # 画像の上端を0とするには絶対値(abs)を取るのが一般的です
            u_px = float(item["u"]) * scale_x
            v_px = height - (float(item["v"]) * scale_y)

            unified_unity[m_id] = (u_px, v_px)

    return unified_sam3d, unified_unity


def convert_unity_to_sam3d_coords(unity_kpts_3d):
    """
    将unity的3D坐标转换为Sam3D的3D坐标系
    """
    sam3d_coords = {}

    for i, (x, y, z) in unity_kpts_3d.items():
        name = UNITY_MHR70_MAPPING[i]

        # Unity座標系からSam3D座標系への変換
        # x=[z],
        # y=[-y],
        # z=[x],
        x_sam = -z
        y_sam = -y
        z_sam = x

        sam3d_coords[i] = np.array([x_sam, y_sam, z_sam])

    return sam3d_coords

def sam3d_3d_array_to_dict(pred_3d_array, target_ids):
    """
    (N,3) -> {joint_id: np.array([x,y,z])}
    """
    pred_3d_array = np.asarray(pred_3d_array)
    out = {}
    for i, m_id in enumerate(target_ids):
        if i < len(pred_3d_array):
            out[m_id] = pred_3d_array[i].astype(np.float64)
    return out
# --- メイン実行 ---


def run_individual_analysis():
    # パス (適宜書き換えてください)
    paths = {
        "sam_l": "/workspace/data/sam3d_body_results/unity/male/left_sam_3d_body_outputs.npz",
        "sam_r": "/workspace/data/sam3d_body_results/unity/male/right_sam_3d_body_outputs.npz",
        "gt_2d_l": "/workspace/data/unity_data/RecordingsPose/cam_left camera/male_kpt2d_left camera_trimmed.jsonl",
        "gt_2d_r": "/workspace/data/unity_data/RecordingsPose/cam_right camera/male_kpt2d_right camera_trimmed.jsonl",
        "gt_3d": "/workspace/data/unity_data/RecordingsPose/male_pose3d_trimmed.jsonl",
    }

    joint_stats = {
        "L_2D": init_joint_stat_container(TARGET_IDS),
        "R_2D": init_joint_stat_container(TARGET_IDS),
        "L_3D": init_joint_stat_container(TARGET_IDS),
        "R_3D": init_joint_stat_container(TARGET_IDS),
    }

    # データ読み込み
    sam_l = np.load(paths["sam_l"], allow_pickle=True)["arr_0"]
    sam_r = np.load(paths["sam_r"], allow_pickle=True)["arr_0"]
    gt_2d_l = [
        json.loads(line) for line in open(paths["gt_2d_l"], "r", encoding="utf-8-sig")
    ]
    gt_2d_r = [
        json.loads(line) for line in open(paths["gt_2d_r"], "r", encoding="utf-8-sig")
    ]
    gt_3d = [
        json.loads(line) for line in open(paths["gt_3d"], "r", encoding="utf-8-sig")
    ]

    num_frames = min(len(sam_l), len(sam_r), len(gt_2d_l), len(gt_2d_r), len(gt_3d))
    results = {"L_2D": [], "R_2D": [], "L_3D": [], "R_3D": []}

    for i in range(num_frames):
        # GT整理
        g2d_l, g3d_l = get_unity_gt_dicts(gt_2d_l[i], gt_3d[i])
        g2d_r, g3d_r = get_unity_gt_dicts(gt_2d_r[i], gt_3d[i])

        # Pred整理
        p2d_l, p3d_l = get_sam_pred_dicts(sam_l[i])
        p2d_r, p3d_r = get_sam_pred_dicts(sam_r[i])

        # 2d 坐标转换
        p2d_l, g2d_l = harmonize_to_pixel_coords(
            sam3d_raw=p2d_l,
            unity_raw=gt_2d_l[i],
            mapping_dict=UNITY_MHR70_MAPPING,
            target_ids=TARGET_IDS,
            width=1920,
            height=1080,
            scale_x=1.0,
            scale_y=1.0,
        )

        p2d_r, g2d_r = harmonize_to_pixel_coords(
            sam3d_raw=p2d_r,
            unity_raw=gt_2d_r[i],
            mapping_dict=UNITY_MHR70_MAPPING,
            target_ids=TARGET_IDS,
            width=1920,
            height=1080,
            scale_x=1.0,
            scale_y=1.0,
        )

        # --- Sam3D prediction (dict) ---
        p3d_l_dict = sam3d_3d_array_to_dict(p3d_l, TARGET_IDS)
        p3d_r_dict = sam3d_3d_array_to_dict(p3d_r, TARGET_IDS)

        # --- Unity GT (already converted to Sam3D coords) ---
        # g3d_l, g3d_r 已经是 dict[m_id] = np.array([x,y,z])

        # --- per-joint errors (2D) ---
        per_j_err, _ = calculate_per_joint_errors(p2d_l, g2d_l)
        accumulate_joint_errors(joint_stats["L_2D"], per_j_err)

        per_j_err, _ = calculate_per_joint_errors(p2d_r, g2d_r)
        accumulate_joint_errors(joint_stats["R_2D"], per_j_err)

        # --- per-joint errors (3D) ---
        per_j_err, _ = calculate_per_joint_errors(p3d_l_dict, g3d_l)
        accumulate_joint_errors(joint_stats["L_3D"], per_j_err)

        per_j_err, _ = calculate_per_joint_errors(p3d_r_dict, g3d_r)
        accumulate_joint_errors(joint_stats["R_3D"], per_j_err)

        # 3d 坐标转换
        # p3d_l = convert_unity_to_sam3d_coords(g3d_l)

        # p3d_r = convert_unity_to_sam3d_coords(g3d_r)

        # エラー計算
        results["L_2D"].append(calculate_mpjpe(p2d_l, g2d_l))
        results["R_2D"].append(calculate_mpjpe(p2d_r, g2d_r))

        # results["L_3D"].append(calculate_mpjpe(p3d_l, g3d_l))
        # results["R_3D"].append(calculate_mpjpe(p3d_r, g3d_r))
        results["L_3D"].append(calculate_mpjpe(p3d_l_dict, g3d_l))
        results["R_3D"].append(calculate_mpjpe(p3d_r_dict, g3d_r))

    # レポート表示
    print(f"=== Individual Perspective Analysis ({num_frames} frames) ===")
    print(
        f"LEFT View  | 2D MPJPE: {np.nanmean(results['L_2D']):.2f} px | 3D MPJPE: {np.nanmean(results['L_3D']):.4f} units"
    )
    print(
        f"RIGHT View | 2D MPJPE: {np.nanmean(results['R_2D']):.2f} px | 3D MPJPE: {np.nanmean(results['R_3D']):.4f} units"
    )

    summary_L2D = summarize_joint_errors(joint_stats["L_2D"])
    summary_R2D = summarize_joint_errors(joint_stats["R_2D"])
    summary_L3D = summarize_joint_errors(joint_stats["L_3D"])
    summary_R3D = summarize_joint_errors(joint_stats["R_3D"])

    print_joint_error_table(summary_L2D, UNITY_MHR70_MAPPING, "LEFT View Per-Joint 2D Error", unit="")
    print_joint_error_table(summary_R2D, UNITY_MHR70_MAPPING, "RIGHT View Per-Joint 2D Error", unit="")
    print_joint_error_table(summary_L3D, UNITY_MHR70_MAPPING, "LEFT View Per-Joint 3D Error", unit="")
    print_joint_error_table(summary_R3D, UNITY_MHR70_MAPPING, "RIGHT View Per-Joint 3D Error", unit="")


if __name__ == "__main__":
    run_individual_analysis()
