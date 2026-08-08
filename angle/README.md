# スキー動作角度解析モジュール - 詳細説明書

## 📋 概要

本モジュールは、**スキー動作の解析**に向けた Python ツールです。3D キーポイント（keypoints）に基づき、人体の関節角度や身体の傾斜などの運動学的指標を計算し、可視化結果を生成します。

**プロジェクト情報：**
- 作成日：2026年2月11日
- 作者：Kaixu Chen
- 所属：筑波大学
- ファイル位置：`/workspace/code/angle/main.py`

---

## 🎯 コア機能

### 1. **キーポイント対応表と骨格定義**
Unity の人体骨格システム（MHR70 モデル）とキーポイントの対応関係を定義します。

| キーポイントID | 名称（英語） | 体の部位 |
|---------|-----------|---------|
| 1, 2 | Eye_L/R | 目 |
| 5, 6 | Upperarm_L/R | 上腕 |
| 7, 8 | lowerarm_l/r | 前腕 |
| 9, 10 | Thigh_L/R | 大腿 |
| 11, 12 | calf_l/r | 下腿 |
| 13, 14 | Foot_L/R | 足 |
| 41 | Hand_R | 右手 |
| 62 | Hand_L | 左手 |
| 69 | neck_01 | 首 |

### 2. **主要計算モジュール**

#### 📐 関節角度の計算
人体の主要関節の屈曲角度（度数）を算出します。

**対応する角度タイプ：**
- **膝角** (knee_l, knee_r)：大腿→下腿→足のなす角
- **肘角** (elbow_l, elbow_r)：上腕→前腕→手のなす角
- **肩角** (shoulder_l, shoulder_r)：首→上腕→前腕のなす角
- **股関節角** (hip_l, hip_r)：首→大腿→下腿のなす角

#### 🏂 体幹傾斜角 (Tilt Angles)
上半身・下半身の鉛直方向に対する傾斜を計算します。
- **上半身傾斜** (tilt_upper)：肩中心が骨盤中心に対して前後に傾く角度
- **下半身傾斜** (tilt_lower)：膝中心が骨盤中心に対して前後に傾く角度
- 角度の符号：正は前傾、負は後傾

#### 📊 複合角度指標
- **体幹-膝角** (torso_knee_angle)：肩中心→骨盤中心→膝中心のなす角
- **膝角差** (knee_diff_lr)：左膝角 - 右膝角
- **肘の中線距離** (elbow_distance_l/r)：肘から身体中線までの水平距離

---

## 🔧 コア関数の説明

### 補助関数

#### `_center_from_ids(frame, ids, id_to_index) → np.ndarray`
**機能：** 複数キーポイントの中心位置を計算します。  
**引数：**
- `frame`：単一フレームのキーポイント配列 (J, 3)
- `ids`：キーポイント ID のタプル（例：`(9, 10)` は左右の股関節）
- `id_to_index`：ID から配列インデックスへのマップ

**戻り値：**
- 有効な点が存在する場合は平均位置 (3,) を返す
- 有効な点がない場合は NaN 配列を返す

**図示：**
```
身体部位中心の計算：
pelvis center = mean([hip_L, hip_R])      # 骨盤中心
shoulder center = mean([shoulder_L, shoulder_R])  # 肩中心
knee center = mean([knee_L, knee_R])     # 膝中心
```

#### `_unit(v: np.ndarray) → np.ndarray`
**機能：** ベクトルの正規化（単位ベクトル化）  
**引数：** `v` - 3次元ベクトル (3,)  
**戻り値：**
- 正規化後の単位ベクトル
- 長さが 0 の場合は NaN 配列

---

### 主要計算関数

#### `compute_angles(kpts, angle_defs, id_to_index) → Dict[str, np.ndarray]`
**機能：** 関節角度の時系列を計算します。  
**引数：**
- `kpts`：キーポイント配列 (T, J, 3)
  - T：フレーム数（時間ステップ数）
  - J：キーポイント数
  - 3：座標 (x, y, z)
- `angle_defs`：角度定義辞書（形式：`{"angle_name": (点A_ID, 点B_ID, 点C_ID)}`）
- `id_to_index`：ID からインデックスへのマップ

**戻り値：** 角度時系列の辞書
```python
{
    "knee_l": [角度1, 角度2, ...],      # shape (T,)
    "knee_r": [角度1, 角度2, ...],
    ...
}
```

**計算原理：**
```
3点 A, B, C の角度 ∠ABC を求める
ベクトル BA = A - B
ベクトル BC = C - B
cos(∠ABC) = (BA · BC) / (|BA| × |BC|)
∠ABC = arccos(cos値) を度数へ変換
```

#### `compute_tilt_angles(kpts, id_to_index, up_axis) → Dict[str, np.ndarray]`
**機能：** 体の前後傾斜角度を計算します。  
**引数：**
- `kpts`：キーポイント配列 (T, J, 3)
- `id_to_index`：ID マップ
- `up_axis`：鉛直方向ベクトル（例：`[0, -1, 0]` は Y 軸下向き）

**戻り値：**
```python
{
    "tilt_upper": [度数...],   # 上半身傾斜
    "tilt_lower": [度数...],   # 下半身傾斜
}
```

**角度の符号：**
- **正**：前傾（スキーでの標準的な減速姿勢）
- **負**：後傾

**幾何計算：**
1. 左右方向の決定：LR = shoulder_R - shoulder_L
2. 前後方向：forward = up_axis × LR
3. 傾斜ベクトルの算出：
   - 上半身：shoulder_center - pelvis_center
   - 下半身：knee_center - pelvis_center
4. 鉛直軸に直交する平面へ投影し角度を算出
5. 前後方向により符号を決定

#### `compute_torso_knee_angle(kpts, id_to_index) → Dict[str, np.ndarray]`
**機能：** 体幹と膝の相対角度を計算します。  
**計算方法：** 3つの中心点の角度を点積で算出
- 点 A：肩中心（shoulder_L + shoulder_R）/ 2
- 点 B：骨盤中心（hip_L + hip_R）/ 2
- 点 C：膝中心（knee_L + knee_R）/ 2
- ∠ABC を返す

**用途：** 滑走中の体の曲げ具合を評価

#### `compute_knee_difference(kpts, id_to_index) → Dict[str, np.ndarray]`
**機能：** 左右膝角度の差を計算します。  
**計算式：** knee_diff_lr = knee_angle_left - knee_angle_right  
**用途：** 左右対称性の評価

#### `compute_elbow_distance_from_midline(kpts, id_to_index) → Dict[str, np.ndarray]`
**機能：** 肘から身体中線までの水平距離を計算します。  
**定義：**
- **身体中線**：骨盤中心を通る鉛直面
- **水平距離**：XZ 平面上で測定（Y 軸は無視）

**計算式：**
```
dist = sqrt((elbow_x - pelvis_x)² + (elbow_z - pelvis_z)²)
```

**用途：** 腕の位置と体幹の関係を評価

---

### データ保存と可視化関数

#### `save_angles_csv(out_path, angles) → None`
**機能：** 角度データを CSV に保存します。  
**出力例：**
```csv
frame,knee_l,knee_r,elbow_l,elbow_r,...
0,120.5,118.3,95.2,96.1,...
1,121.2,119.1,94.8,95.9,...
...
```

#### `plot_angles(out_path, angles) → None`
**機能：** 角度時系列を複数行のグラフに描画します。  
**出力：** 角度ごとに 1 行の PNG

**図の形式：**
- X 軸：フレーム
- Y 軸：角度（度）
- グリッドと凡例付き

#### `visualize_elbow_position(kpts, id_to_index, output_dir) → None`
**機能：** 肘の位置を可視化します。  
**2つのサブプロット：**

1. **トップビュー**
   - 骨盤中心の軌跡（黒）
   - 左肘の軌跡（青）
   - 右肘の軌跡（赤）
   - 座標軸：X（左右）、Z（前後）

2. **時系列グラフ**
   - X 軸：フレーム
   - Y 軸：肘から中線までの水平距離（m）
   - 左肘：青、右肘：赤

#### `visualize_3d_keypoints(kpts, id_to_index, output_dir, num_frames_to_save=5) → None`
**機能：** 代表フレームを選び 3D 骨格を描画します。  
**引数：**
- `num_frames_to_save`：保存するフレーム数（均等分布）

**出力：**
- 保存先：`skeleton_visualization/skeleton_frames/`
- ファイル形式：`skeleton_frame_XXXX.png`
- 各画像には以下を表示：
  - 3D 骨格の接続（青線）
  - キーポイント（赤点）
  - 軸ラベルと座標範囲

---

## 🚀 使用手順

### 入力データ要件
**キーポイント形式：**
- ファイル形式：NumPy 配列（`.npy`）
- 形状：(T, J, 3)
  - T：フレーム数
  - J：キーポイント数（少なくとも 15 点）
  - 3：座標 (x, y, z)
- 順序：`TARGET_IDS` で定義

### 実行フロー

```python
main()  # エントリポイント
  ├─ 入力ディレクトリ内の各人物を処理
  │
  └─ process_person(input_path, output_dir)
      ├─ np.load(input_path)  # データ読み込み
      │
      ├─ compute_angles()  # 関節角度
      │   └─ 出力：angles_joint.csv/.png
      │
      ├─ compute_tilt_angles()  # 傾斜角
      │   └─ 出力：angles_body_y_down.csv/.png
      │
      ├─ compute_torso_knee_angle()  # 体幹-膝角
      │   └─ 出力：angles_torso_knee.csv/.png
      │
      ├─ compute_knee_difference()  # 膝角差
      │   └─ 出力：angles_knee_diff.csv/.png
      │
      ├─ compute_elbow_distance_from_midline()  # 肘距離
      │   └─ 出力：distance_elbow_midline.csv/.png
      │
      ├─ visualize_elbow_position()  # 肘位置の可視化
      │   └─ 出力：elbow_position_visualization.png
      │
      └─ visualize_3d_keypoints()  # 3D 骨格
          └─ 出力：skeleton_frames/ 内の PNG
```

### 出力ファイル構成

```
output_dir/
├── angles_joint.csv          # 関節角度データ
├── angles_joint.png          # 関節角度グラフ
├── angles_body_y_down.csv    # 体幹傾斜データ
├── angles_body_y_down.png    # 体幹傾斜グラフ
├── angles_torso_knee.csv     # 体幹-膝角データ
├── angles_torso_knee.png     # 体幹-膝角グラフ
├── angles_knee_diff.csv      # 膝角差データ
├── angles_knee_diff.png      # 膝角差グラフ
├── distance_elbow_midline.csv  # 肘距離データ
├── distance_elbow_midline.png  # 肘距離グラフ
├── elbow_position_visualization.png  # 肘位置可視化
└── skeleton_frames/          # 3D 骨格可視化
    ├── skeleton_frame_0000.png
    ├── skeleton_frame_0001.png
    └── ...
```

---

## 📊 データ形式の詳細

### CSV 出力例

**angles_joint.csv**
```
frame,knee_l,knee_r,elbow_l,elbow_r,shoulder_l,shoulder_r,hip_l,hip_r
0,125.5,124.3,98.2,99.1,45.2,46.1,115.3,114.8
1,126.1,125.0,97.8,98.7,44.9,45.8,116.1,115.5
2,126.8,125.7,97.2,98.2,44.6,45.5,116.9,116.2
...
```

**angles_body_y_down.csv**
```
frame,tilt_upper,tilt_lower
0,15.2,8.5
1,16.1,9.2
2,17.3,10.1
...
```

### 数値の解釈

**関節角（膝角など）：**
- 範囲：0° ~ 180°
- 0°：完全伸展
- 90°：直角屈曲
- 180°：反対伸展（通常は少ない）

**傾斜角：**
- 範囲：-90° ~ +90°
- 正：前傾
- 負：後傾
- 0°：完全に鉛直

**肘距離：**
- 範囲：0 以上
- 単位：メートル（座標がメートルの場合）
- 0：肘が中線上

---

## 🔍 主要パラメータと定数

### 骨格接続定義 (SKELETON_CONNECTIONS)
3D 可視化と幾何計算に使用：
```python
[
    # 左腕
    (69, 5),    # 首 → 左肩
    (5, 7),     # 左肩 → 左肘
    (7, 62),    # 左肘 → 左手
    # 右腕
    (69, 6),    # 首 → 右肩
    (6, 8),     # 右肩 → 右肘
    (8, 41),    # 右肘 → 右手
    # ...（他の接続）
]
```

### 角度定義 (ANGLE_DEFS)
```python
{
    "knee_l": (9, 11, 13),      # 左大腿 → 左下腿 → 左足
    "knee_r": (10, 12, 14),     # 右大腿 → 右下腿 → 右足
    "elbow_l": (5, 7, 62),      # 左上腕 → 左前腕 → 左手
    "elbow_r": (6, 8, 41),      # 右上腕 → 右前腕 → 右手
    "shoulder_l": (69, 5, 7),   # 首 → 左上腕 → 左前腕
    "shoulder_r": (69, 6, 8),   # 首 → 右上腕 → 右前腕
    "hip_l": (69, 9, 11),       # 首 → 左大腿 → 左下腿
    "hip_r": (69, 10, 12),      # 首 → 右大腿 → 右下腿
}
```

---

## ⚠️ 注意事項と制限

### データ有効性チェック
各関数で以下を確認します。
1. **形状検証**：入力は (T, J, 3)
2. **有限性チェック**：`np.isfinite()` で NaN / inf を除外
3. **欠損対応**：無効データは NaN で返す

### 座標系の前提
- **X 軸**：左右（左が負、右が正）
- **Y 軸**：鉛直（コードは Y 下向き）
- **Z 軸**：前後（前が正、後が負）

### 性能上の注意
- 大規模データ（T > 100,000）では可視化保存に時間がかかる
- `visualize_3d_keypoints` の `num_frames_to_save` を調整推奨

---

## 🎓 適用例：スキー動作解析

スキー動作では以下に活用できます。

1. **姿勢評価**
   - 膝や肘の屈曲度の監視
   - 体幹傾斜の適切性評価

2. **対称性評価**
   - 左右膝角差で協調性を確認
   - 肘距離差で上肢バランスを評価

3. **運動学特徴の抽出**
   - 周期運動の周期算出
   - 機械学習用の特徴量抽出

4. **技術指導**
   - リプレイ時の角度データ付加
   - コーチングや改善指導の支援

---

## 📝 依存ライブラリ

```python
import csv              # CSV 出力
from pathlib import Path  # パス操作
from typing import Dict, Tuple  # 型注釈
import matplotlib.pyplot as plt  # 描画
from mpl_toolkits.mplot3d import Axes3D  # 3D 描画
import numpy as np      # 数値計算
```

---

## 🔗 ファイル位置

| 項目 | パス |
|------|------|
| 入力データ | `/workspace/data/fused_smoothed_results` |
| 出力ディレクトリ | `/workspace/data/angle_outputs` |
| 主モジュール | `/workspace/code/angle/main.py` |

---

## 📞 変更・拡張のヒント

### 新しい角度計算を追加する場合
1. `ANGLE_DEFS` に定義を追加
2. 対応するキーポイント ID が `UNITY_MHR70_MAPPING` に存在することを確認

### 座標系を変更する場合
`compute_tilt_angles` の `up_axis` を調整

### 性能を改善する場合
- NumPy のベクトル化でループを削減
- 大規模データはマルチプロセス化

---

**最終更新日：2026年2月26日**
