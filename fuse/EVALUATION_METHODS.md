# 评价方法说明文档

## 概述

本文档详细说明在双视角3D姿态估计融合过程中使用的各种评价方法。这些方法主要用于：
1. 计算单视角预测的置信度
2. 评估双视角预测的一致性
3. 量化最终融合结果与真实值的误差

---

## 1. 弱透视重投影置信度 (Weak Perspective Reprojection Confidence)

### 函数签名
```python
def weakpersp_reproj_confidence(
    X3d: np.ndarray,        # (N,3) 3D关节点坐标
    U2d: np.ndarray,        # (N,2) 2D关节点坐标
    sigma_px: float = 12.0, # 像素误差阈值
    min_points: int = 8,    # 最小有效点数
    eps: float = 1e-12,
)
```

### 功能说明
评估3D姿态预测与2D观测的一致性，通过弱透视相机模型拟合并计算重投影误差。

### 工作原理

1. **弱透视相机模型拟合**
   - 模型公式：`u ≈ s * (X @ M) + t`
   - 其中：
     - `s`: 缩放因子
     - `M`: (3,2) 正交投影矩阵
     - `t`: (2,) 平移向量

2. **拟合过程** (`fit_weakpersp_3d_to_2d`)
   ```
   a. 筛选有效点（非NaN）
   b. 中心化3D和2D点
   c. 计算交叉协方差矩阵 C = X.T @ U
   d. SVD分解得到正交映射矩阵 M
   e. 最小二乘求解缩放因子 s
   f. 计算2D平移向量 t
   ```

3. **重投影误差计算**
   ```python
   Uhat = s * (X3d @ M) + t     # 预测的2D投影
   err = ||Uhat - U2d||          # 欧氏距离
   ```

4. **置信度计算**
   ```python
   conf = exp(-err² / (2 * sigma_px²))
   ```
   - 使用高斯核函数将误差转换为[0,1]区间的置信度
   - `sigma_px`控制软硬程度：
     - 较小值(~5)：对误差更敏感
     - 较大值(~20)：更宽容

### 返回值
- `conf`: (N,) 每个关节的置信度 [0,1]
- `err_px`: (N,) 每个关节的重投影误差（像素）
- `Uhat`: (N,2) 拟合的2D重投影坐标
- `params`: 拟合参数字典 {s, M, t, valid_used}

### 典型参数设置
```python
conf, err, uhat, params = weakpersp_reproj_confidence(
    p3d_raw, p2d_raw, 
    sigma_px=12.0  # 12像素的标准差
)
```

### 应用场景
- 单视角3D姿态估计的自我一致性检查
- 筛选低质量的关节预测
- 作为融合权重的一部分

---

## 2. 跨视角一致性置信度 (Cross-view Consistency Confidence)

### 函数签名
```python
def crossview_consistency_confidence(
    X_a: np.ndarray,              # (N,3) 视角A的3D关节点
    X_b: np.ndarray,              # (N,3) 视角B的3D关节点
    *,
    root_idx: int,                # 根关节索引（如骨盆）
    left_hip_idx: int,            # 左髋索引
    right_hip_idx: int,           # 右髋索引
    left_shoulder_idx: int,       # 左肩索引
    right_shoulder_idx: int,      # 右肩索引
    sigma_3d: float = 0.08,       # 3D距离阈值
    scale_mode: str = "hip",      # 归一化模式："hip"或"torso"
)
```

### 功能说明
评估两个视角的3D姿态预测在标准化空间中的一致性。

### 工作原理

1. **姿态标准化** (`canonicalize_pose_3d`)
   
   为每个视角建立标准坐标系：
   
   a. **原点设置**：平移到根关节（骨盆）
   ```python
   X0 = X - X[root_idx]
   ```
   
   b. **坐标轴构建**：
   ```python
   x轴: 左髋 → 右髋 (left_hip -> right_hip)
   y轴: 髋中点 → 肩中点 (mid_hip -> mid_shoulder)
   z轴: x × y (右手系)
   ```
   
   c. **旋转对齐**：
   ```python
   R = [x_axis, y_axis, z_axis]  # (3,3)旋转矩阵
   Xr = (R @ X0.T).T
   ```
   
   d. **尺度归一化**：
   - `scale_mode="hip"`: 除以髋宽 `||right_hip - left_hip||`
   - `scale_mode="torso"`: 除以躯干长度 `||mid_shoulder - mid_hip||`
   ```python
   Xc = Xr / scale
   ```

2. **一致性距离计算**
   ```python
   dist = ||Xa_canonical - Xb_canonical||
   ```
   在标准化空间中比较对应关节的欧氏距离

3. **置信度计算**
   ```python
   conf = exp(-dist² / (2 * sigma_3d²))
   ```
   - `sigma_3d`典型范围：0.05-0.12（标准化空间单位）

### 返回值
- `conf`: (N,) 每个关节的跨视角一致性置信度 [0,1]
- `dist`: (N,) 标准化空间中的3D距离
- `Xa_c`, `Xb_c`: 两个视角的标准化姿态
- `info`: 包含旋转矩阵和尺度的字典

### 典型参数设置
```python
# MHR70关键点索引
IDX_PELVIS = 14
IDX_LHIP = 11
IDX_RHIP = 12
IDX_LSHO = 5
IDX_RSHO = 6

conf, dist, Xlc, Xrc, info = crossview_consistency_confidence(
    p3d_left, p3d_right,
    root_idx=IDX_PELVIS,
    left_hip_idx=IDX_LHIP,
    right_hip_idx=IDX_RHIP,
    left_shoulder_idx=IDX_LSHO,
    right_shoulder_idx=IDX_RSHO,
    sigma_3d=0.08,
    scale_mode="hip"  # 使用髋宽归一化
)
```

### 应用场景
- 双视角3D预测的交叉验证
- 检测视角间的姿态冲突
- 作为融合权重的关键组成部分

### 为什么需要标准化？
- **消除全局变换**：不同视角的3D坐标系可能不同
- **尺度归一化**：不同人/场景的身高体型差异
- **关注姿态本身**：只比较相对关节配置

---

## 3. 平均关节点位置误差 (MPJPE - Mean Per Joint Position Error)

### 函数签名
```python
def calculate_mpjpe(pred_dict, gt_dict):
    """计算预测与真值之间的平均关节点误差"""
```

### 功能说明
计算预测关节点与真实关节点之间的平均欧氏距离，是3D姿态估计最常用的评价指标。

### 工作原理

1. **关节匹配**
   ```python
   common_ids = set(pred_dict.keys()) & set(gt_dict.keys())
   ```
   只计算两者都有的关节

2. **逐关节误差计算**
   ```python
   for joint_id in common_ids:
       err = ||pred[joint_id] - gt[joint_id]||  # L2距离
       errors.append(err)
   ```

3. **平均**
   ```python
   MPJPE = mean(errors)
   ```

### 返回值
- `float`: 平均误差值
- 单位取决于输入：
  - 2D情况：像素 (px)
  - 3D情况：米 (m) 或毫米 (mm)

### 使用示例

```python
# 计算2D MPJPE（像素）
mpjpe_2d = calculate_mpjpe(pred_2d_dict, gt_2d_dict)
print(f"2D MPJPE: {mpjpe_2d:.2f} px")

# 计算3D MPJPE（米）
mpjpe_3d = calculate_mpjpe(fused_3d_dict, gt_3d_dict)
print(f"3D MPJPE: {mpjpe_3d:.4f} m")
# 或转换为毫米
print(f"3D MPJPE: {mpjpe_3d*1000:.1f} mm")
```

### 相关函数

#### `calculate_per_joint_errors(pred_dict, gt_dict)`
计算每个关节的单独误差，用于详细分析。

返回：
- `per_joint_err`: 字典 {joint_id: error_value}
- `common_ids`: 使用的关节ID列表

---

## 4. 综合融合策略

在实际应用中，这些方法组合使用：

### 完整流程示例

```python
# 1. 计算单视角重投影置信度
conf_l, _, _, _ = weakpersp_reproj_confidence(
    p3d_left, p2d_left, sigma_px=12.0
)
conf_r, _, _, _ = weakpersp_reproj_confidence(
    p3d_right, p2d_right, sigma_px=12.0
)

# 2. 计算跨视角一致性置信度
conf_cross, _, _, _, _ = crossview_consistency_confidence(
    p3d_left, p3d_right,
    root_idx=14, left_hip_idx=11, right_hip_idx=12,
    left_shoulder_idx=5, right_shoulder_idx=6,
    sigma_3d=0.08, scale_mode="hip"
)

# 3. 组合置信度（几何平均）
q_left = np.sqrt(conf_l * conf_cross)   # 左视角最终权重
q_right = np.sqrt(conf_r * conf_cross)  # 右视角最终权重

# 4. 加权融合3D姿态
fused_3d = fuse_frame_3d(p3d_left, p3d_right, q_left, q_right, target_ids)

# 5. 时序平滑（可选）
smooth_seq = temporal_smooth_ema(fused_sequence, target_ids, alpha=0.7)

# 6. 评估最终结果
mpjpe = calculate_mpjpe(fused_3d, gt_3d)
print(f"Final 3D MPJPE: {mpjpe:.2f}")
```

### 权重组合原理

**几何平均 vs 算术平均**：
```python
# 几何平均（推荐）：更保守，任一项低则总权重低
q = sqrt(conf_reproj * conf_cross)

# 算术平均：较宽松
q = (conf_reproj + conf_cross) / 2
```

选择几何平均的原因：
- 如果重投影误差大或跨视角不一致，该关节可信度应该低
- 两个条件都满足才给高权重
- 更鲁棒地处理异常值

---

## 5. 参数调优指南

### `sigma_px` (重投影误差阈值)
- **默认值**: 12.0 px
- **含义**: e^(-1)时的误差（约37%置信度）
- **调整建议**:
  - 高分辨率图像 (>1920px)：10-15 px
  - 低分辨率图像 (<1280px)：8-12 px
  - 远距离拍摄：增大到15-20 px

### `sigma_3d` (3D一致性阈值)
- **默认值**: 0.08 (标准化空间单位)
- **含义**: 标准化后的3D距离阈值
- **调整建议**:
  - 严格一致性检查：0.05-0.06
  - 平衡设置：0.08-0.10
  - 宽松设置（视角差异大）：0.12-0.15

### `scale_mode` (归一化模式)
- **"hip"**: 使用髋宽归一化（推荐）
  - 更稳定，髋部通常检测准确
- **"torso"**: 使用躯干长度
  - 对上半身动作更敏感

### `alpha` (时序平滑系数)
- **默认值**: 0.7
- **公式**: `smoothed[t] = alpha * smoothed[t-1] + (1-alpha) * current[t]`
- **调整建议**:
  - 快速动作：0.5-0.6（响应快）
  - 慢速动作：0.7-0.8（更平滑）
  - 静态场景：0.8-0.9（最平滑）

---

## 6. 常见问题与解决方案

### Q1: 置信度全为0或NaN？
**可能原因**:
- 输入数据包含过多NaN值
- `min_points`不满足（需要至少8个有效点）
- 关键骨架点（髋、肩）缺失

**解决方案**:
```python
# 检查有效点数
valid_count = np.isfinite(X3d).all(axis=1).sum()
print(f"Valid points: {valid_count}")

# 降低min_points要求（谨慎）
conf, _, _, _ = weakpersp_reproj_confidence(
    X3d, U2d, min_points=5  # 从8降到5
)
```

### Q2: 跨视角一致性很低但单视角置信度高？
**可能原因**:
- 两个视角的相机标定不准确
- 时间同步问题（视频帧不对齐）
- 不同人被检测为主体

**解决方案**:
- 检查相机外参
- 验证视频时间戳同步
- 使用人物跟踪确保一致性

### Q3: MPJPE很高但视觉效果不错？
**可能原因**:
- 全局平移/旋转偏移
- 尺度不匹配
- 真值坐标系与预测不同

**解决方案**:
```python
# 使用对齐的MPJPE（PA-MPJPE）
from scipy.spatial.transform import Rotation

def align_poses(pred, gt):
    """Procrustes对齐"""
    # 实现略...
    return aligned_pred

aligned_pred = align_poses(pred_3d, gt_3d)
mpjpe = calculate_mpjpe(aligned_pred, gt_3d)
```

---

## 7. 文件位置索引

### 实现文件
- **置信度计算**: `/workspace/code/fuse/confidence.py`
  - `weakpersp_reproj_confidence()`
  - `crossview_consistency_confidence()`
  - `fit_weakpersp_3d_to_2d()`
  - `canonicalize_pose_3d()`

- **误差计算**: `/workspace/code/fuse/unity_data_compare.py`
  - `calculate_mpjpe()`
  - `calculate_per_joint_errors()`

### 使用示例
- **Unity数据处理**: `/workspace/code/fuse/main_unity.py`
- **原始数据处理**: `/workspace/code/fuse/main_raw.py`

### 数据加载
- **数据加载**: `/workspace/code/fuse/load_raw.py`
  - `load_sam_data()` - 支持单文件和按帧保存两种格式

---

## 8. 参考文献与理论基础

### 弱透视相机模型
- Hartley & Zisserman. "Multiple View Geometry in Computer Vision"
- 简化的正交相机模型，适用于远距离拍摄

### 姿态标准化
- Procrustes Analysis
- 消除平移、旋转、缩放的影响

### 多视角融合
- Triangulation methods
- Confidence-weighted average

---

## 更新日志

- **2026-02-11**: 初始版本，包含三个主要评价方法
- 文档作者: Kaixu Chen
- 联系方式: chenkaixusan@gmail.com
