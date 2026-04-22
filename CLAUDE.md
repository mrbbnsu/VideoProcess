# VideoPoseRecognization

基于 AIoT 的脑卒中患者步行平衡能力评价与智能干预系统。

## 项目结构

```
VideoPoseRecognization/
├── src/
│   ├── pose_estimation/   # 姿态估计模块
│   ├── gait_analysis/     # 步态分析模块
│   └── knowledge_base/    # 知识库 RAG
├── scripts/               # 训练/提取脚本
├── experiments/           # 测试代码（Test.mp4）
├── data/raw/Video/        # 原始视频数据
├── models/
│   ├── openpose_dlls/     # 旧版编译好的 OpenPose DLL（可能过时）
│   ├── openpose/          # OpenPose 源码（git clone 的）
│   ├── pose_iter_584000.caffemodel  # OpenPose Caffe 预训练模型
│   └── gait_*.pkl/.pth    # 训练好的步态模型
├── docs/
├── literature/            # 文献笔记
├── output/                # 调试输出图像
├── openpose/              # OpenPose 源码
└── requirements.txt
```

## 技术栈

- Python 3.8+
- 姿态估计: OpenPose (Caffe), MediaPipe
- 视频处理: OpenCV
- ML: scikit-learn (MLP), PyTorch
- 知识库 RAG: ChromaDB + requests

## OpenPose

### 源码
```
git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
```
**注意**: OpenPose 需要从源码编译（CMake + Visual Studio）。

### 旧版预编译 DLL（可能过时）
位于 `models/openpose_dlls/`，包含:
- openpose.dll 及依赖的 caffe/opencv 等 DLL
- pyopenpose.cp37-win_amd64.pyd (Python 3.7 binding)

### 模型文件 (models/)
| 文件 | 说明 |
|------|------|
| `pose_iter_584000.caffemodel` | OpenPose BODY_25 预训练权重 (100MB) |
| `pose_deploy.prototxt` | Caffe 网络定义 |
| `gait_mlp_model.pkl` | 步态分类 MLP 模型 |

## 下肢关键点（MediaPipe 10个）

| 索引 | 左侧 | 右侧 |
|------|------|------|
| 23/24 | left_hip | right_hip |
| 25/26 | left_knee | right_knee |
| 27/28 | left_ankle | right_ankle |
| 29/30 | left_heel | right_heel |
| 31/32 | left_foot_index | right_foot_index |

## OpenPose BODY_25 关键点（25个）

| 索引 | 名称 | 索引 | 名称 |
|------|------|------|------|
| 9 | right_hip | 12 | left_hip |
| 10 | right_knee | 13 | left_knee |
| 11 | right_ankle | 14 | left_ankle |
| 19 | left_big_toe | 22 | right_big_toe |
| 20 | left_small_toe | 23 | right_small_toe |
| 21 | left_heel | 24 | right_heel |

## 下一步

1. **编译 OpenPose** - Windows 上用 CMake + Visual Studio
2. **测试姿态估计** - 用 MediaPipe 或编译后的 OpenPose
3. **重建步态分析代码** - 从 bak/ 中参考或重新实现

## 知识库配置 (.env)

```bash
# 用户代理配置
HTTP_PROXY=http://proxy.xenodynamics.cc:11080
HTTPS_PROXY=http://proxy.xenodynamics.cc:11080
```
