"""查看关键点坐标，判断视角（侧面/正面）并提取步态特征。"""
import cv2
import math
from pathlib import Path
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options
from mediapipe import Image, ImageFormat

video_path = Path(__file__).parent.parent / "data/Rec/Test.MOV"
model_path = Path(__file__).parent.parent / "models/pose_landmarker_lite.task"
out_dir = Path(__file__).parent.parent / "output"
out_dir.mkdir(exist_ok=True)

cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Resolution: {w}x{h}, FPS: {fps:.2f}, Duration: {total/fps:.1f}s")

base_opts = base_options.BaseOptions(model_asset_path=str(model_path))
options = vision.PoseLandmarkerOptions(base_options=base_opts, running_mode=vision.RunningMode.VIDEO)
landmarker = vision.PoseLandmarker.create_from_options(options)

# 采样 10 帧看关键点分布
sample_frames = [14, 50, 100, 150, 200, 250, 300, 350, 380, 395]
lower_ids = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
name_map = {23:"L_Hip", 24:"R_Hip", 25:"L_Knee", 26:"R_Knee",
            27:"L_Ankle", 28:"R_Ankle", 29:"L_Heel", 30:"R_Heel",
            31:"L_FootIdx", 32:"R_FootIdx"}

for frame_idx in sample_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame_bgr = cap.read()
    if not ok:
        continue

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
    ts_ms = int((frame_idx / fps) * 1000)
    result = landmarker.detect_for_video(mp_image, ts_ms)

    if not result.pose_landmarks:
        print(f"Frame {frame_idx}: no detection")
        continue

    lms = result.pose_landmarks[0]
    print(f"\nFrame {frame_idx} ({ts_ms/1000:.2f}s):")
    print(f"  {'Joint':<12} {'x':>8} {'y':>8} {'vis':>6}")
    for i in lower_ids:
        if i < len(lms):
            lm = lms[i]
            print(f"  {name_map[i]:<12} {lm.x:>8.4f} {lm.y:>8.4f} {lm.visibility:>6.2f}")

    # 判断：x 值接近 → 正面视角（左右对称）; y 值接近 → 侧面视角（前后深度）
    # 髋部
    lh = lms[23] if len(lms) > 23 else None
    rh = lms[24] if len(lms) > 24 else None
    if lh and rh:
        dx = abs(lh.x - rh.x)
        dy = abs(lh.y - rh.y)
        print(f"  Hip delta: dx={dx:.4f}, dy={dy:.4f}  {'-> Front view (left-right symmetric)' if dx > 0.05 else '-> Side view?'}")

cap.release()
landmarker.close()
