# 滑雪动作分段与关节角度分析程序说明（angle/main.py）

## 1. 研究目的

本程序针对滑雪运动的 3D 关键点数据，自动将完整滑雪动作序列按滑雪者面向前方的朝向变化划分为若干 turn（转弯段），并在每个 turn 内统计和比较融合前后各关节角度的变化。该方法解决了 run/pro 等动作长度不一、难以直接对齐比较的问题，实现了动作标准化分段与 turn 级别的技术分析。

## 2. 数据输入与预处理

- **输入**：每个滑雪者的 3D 关键点序列（shape: T × J × 3），J 为关节点数，T 为帧数，采用 UNITY_MHR70_MAPPING 关节点编号。
- **关节筛选**：仅分析 TARGET_IDS 指定的主要关节（如大腿、膝、肘、肩、手等）。
- **缺失值处理**：对关键点序列中的 NaN 采用线性插值（_fill_nan_linear），保证分析连续性。
- **平滑处理**：对角度、朝向等序列采用滑动窗口均值滤波（_smooth_1d），抑制噪声。

## 3. 滑雪动作分段（turn 检测）

- **朝向计算**：compute_facing_heading 利用关键点（如肩、髋、颈等）计算每一帧滑雪者在地面的朝向（heading）。
- **极值检测**：对朝向序列进行平滑和一阶差分，检测极大/极小点作为 turn 的分界点（detect_turn_segments）。
- **turn 有效性筛选**：每个 turn 至少包含 min_turn_frames 帧，且朝向变化幅度需大于 min_heading_change_deg。
- **边界处理**：自动补全首尾 turn，保证全序列覆盖。

## 4. 关节角度与身体姿态分析

- **角度定义**：ANGLE_DEFS 定义膝、肘、肩、髋等主要关节的三点夹角，采用 angle_deg 计算每帧角度序列。
- **身体倾角**：compute_tilt_angles 计算上/下半身相对地面的倾斜角度。
- **膝差异**：compute_knee_difference 计算左右膝角度的差异，反映动作对称性。
- **肘部距离**：compute_elbow_distance_from_midline 计算肘部到身体中线的水平距离。
- **躯干-膝角**：compute_torso_knee_angle 衡量下肢屈伸。

## 5. 分段统计与对比

- **turn 内统计**：对每个 turn，统计各角度/指标的均值、极值、变化幅度等（save_turn_reports, save_turn_detail_files）。
- **全帧变化**：compute_series_changes 计算每帧角度变化量，分析动作流畅性与爆发力。
- **融合前后对比**：save_turn_comparison_report 支持 turn 级别的融合前后指标对齐与差异分析。

## 6. 可视化与结果输出

- **分段可视化**：输出每个 turn 的朝向变化曲线、分段边界、角度变化趋势图（plot_angles, save_turn_reports）。
- **肘部轨迹**：visualize_elbow_position 绘制肘部在水平面内的运动轨迹。
- **3D 骨架可视化**：visualize_3d_keypoints 支持关键帧的 3D 骨架渲染。
- **CSV 报告**：所有 turn 统计、全帧变化、分段对比等均输出为 csv 文件，便于论文制图和统计分析。

## 7. 主要函数说明

- `_fill_nan_linear`：线性插值填充 1D 序列中的 NaN。
- `_smooth_1d`：滑动窗口均值滤波。
- `compute_facing_heading`：计算滑雪者每帧的朝向角度。
- `detect_turn_segments`：根据朝向极值自动分段。
- `compute_angles`：批量计算各关节夹角序列。
- `compute_tilt_angles`：计算身体倾角。
- `compute_knee_difference`：左右膝角度差异。
- `compute_elbow_distance_from_midline`：肘部到中线距离。
- `save_turn_reports`/`save_turn_detail_files`：输出分段统计与详细指标。
- `plot_angles`/`visualize_elbow_position`/`visualize_3d_keypoints`：各类可视化输出。

## 8. 应用场景

- **动作标准化**：自动分段后可对不同长度、不同风格的滑雪动作进行标准化分析。
- **技术对比**：支持融合前后、不同滑雪者、不同训练阶段的 turn 级别技术对比。
- **论文制图**：所有统计与可视化结果均为 csv/png，便于直接用于论文图表和数据分析。
