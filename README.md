# VideoPoseRecognization

基于 AIoT 的脑卒中患者步行平衡能力评价与智能干预系统。

## 项目概述

结合外骨骼辅助与视频姿态识别技术，自动评估患者步态参数，并与临床平衡量表（BBS）进行相关性分析。

## 功能

- **姿态估计**：MediaPipe Tasks API 提取 10 个下肢关键点（髋/膝/踝/足部）
- **步态分析**：Heel-strike 检测、关节角度时序、 cadence 提取
- **频域特征**：躯干轨迹 PSD 分析，用于静态平衡评估
- **临床相关**：行走视频特征与 BBS/FAC 等量表的相关/回归分析

## 核心脚本

| 脚本 | 功能 |
|------|------|
| `scripts/workflow/single_video_angles_events.py` | 单视频处理：提取角度/事件/可视化 |
| `scripts/batch_process_bbs.py` | 批量处理 BBS 数据集 |
| `scripts/analysis/evaluate_gait_event_consistency.py` | Heel-strike 检测算法 |
| `scripts/debug_hs.py` | Heel-strike 检测调试与可视化 |
| `scripts/analyze_balance_features.py` | 临床评分相关性分析 |

## 数据说明

| 视频类型 | 说明 | 髋关节 | 膝/踝关节 | 足部/Heel |
|----------|------|--------|-----------|-----------|
| BBS.MOV | 不穿设备，完整 BBS 评估 | ✅ | ✅ | ✅ |
| pre/training | 穿辅助设备 | ❌ | ✅ | ❌ |

## 下肢关键点（MediaPipe）

| 索引 | 左侧 | 右侧 |
|------|------|------|
| 23/24 | left_hip | right_hip |
| 25/26 | left_knee | right_knee |
| 27/28 | left_ankle | right_ankle |
| 29/30 | left_heel | right_heel |
| 31/32 | left_foot_index | right_foot_index |

## 分析方案

详见 [docs/video_analysis_plan.md](docs/video_analysis_plan.md)

## 环境

- Python 3.11+
- MediaPipe Tasks API
- OpenCV
- scipy（频域分析）
- scikit-learn（回归分析）

## 快速开始

```bash
# 处理单个视频
python scripts/run_test_mov.py

# 批量处理 BBS 数据集
python scripts/batch_process_bbs.py

# 调试 heel-strike 检测
python scripts/debug_hs.py
```
