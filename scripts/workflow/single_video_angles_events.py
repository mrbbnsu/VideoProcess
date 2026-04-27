import argparse
import csv
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from mediapipe.tasks.python import vision
# from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.core import base_options
from mediapipe import Image, ImageFormat

from scripts.analysis.evaluate_gait_event_consistency import detect_heel_strikes  # noqa: E402


LOWER_BODY_IDS = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

# Default model path - can be overridden via param
DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parent.parent.parent / "models" / "pose_landmarker_lite.task")


def angle_abc(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    bax = a[0] - b[0]
    bay = a[1] - b[1]
    bcx = c[0] - b[0]
    bcy = c[1] - b[1]

    n1 = math.hypot(bax, bay)
    n2 = math.hypot(bcx, bcy)
    if n1 < 1e-12 or n2 < 1e-12:
        return float("nan")

    cosv = (bax * bcx + bay * bcy) / (n1 * n2)
    cosv = max(-1.0, min(1.0, cosv))
    return math.degrees(math.acos(cosv))


def point(lms, idx: int, min_vis: float) -> tuple[float, float] | None:
    if idx >= len(lms):
        return None
    lm = lms[idx]
    if lm.visibility < min_vis:
        return None
    return (lm.x, lm.y)


def joint_angle(lms, a: int, b: int, c: int, min_vis: float) -> float:
    pa = point(lms, a, min_vis)
    pb = point(lms, b, min_vis)
    pc = point(lms, c, min_vis)
    if pa is None or pb is None or pc is None:
        return float("nan")
    return angle_abc(pa, pb, pc)


def finite_or_empty(v: float) -> str:
    return f"{v:.6f}" if math.isfinite(v) else ""


def joint_distance(lms, i: int, j: int, min_vis: float) -> float | None:
    """Euclidean distance between two landmarks (normalized 0-1 coordinates)."""
    if i >= len(lms) or j >= len(lms):
        return None
    li, lj = lms[i], lms[j]
    if li.visibility < min_vis or lj.visibility < min_vis:
        return None
    return math.hypot(li.x - lj.x, li.y - lj.y)


def safe_ratio(num: float, den: float, default: float = float("nan")) -> float:
    if not math.isfinite(num) or not math.isfinite(den):
        return default
    if abs(den) < 1e-12:
        return default
    return num / den


def parse_float_or_nan(v: str) -> float:
    if not v:
        return float("nan")
    try:
        return float(v)
    except ValueError:
        return float("nan")


def crop_frame_lr(frame_bgr, crop_left_ratio: float, crop_right_ratio: float):
    if crop_left_ratio <= 0 and crop_right_ratio <= 0:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    x0 = int(w * crop_left_ratio)
    x1 = w - int(w * crop_right_ratio)
    x0 = max(0, min(x0, w - 1))
    x1 = max(x0 + 1, min(x1, w))
    cropped = frame_bgr[:, x0:x1]
    return cropped.copy()


def estimate_pose_state(lms, min_vis: float, min_points: int) -> tuple[tuple[float, float], float] | None:
    pts = []
    for lm in lms:
        if lm.visibility >= min_vis:
            pts.append((lm.x, lm.y))

    if len(pts) < min_points:
        return None

    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)

    min_x = min(p[0] for p in pts)
    max_x = max(p[0] for p in pts)
    min_y = min(p[1] for p in pts)
    max_y = max(p[1] for p in pts)
    scale = max(max_x - min_x, max_y - min_y)
    if scale <= 1e-8:
        return None

    return (cx, cy), scale


def pass_single_person_lock(
    current_state: tuple[tuple[float, float], float] | None,
    last_state: tuple[tuple[float, float], float] | None,
    max_center_jump: float,
    max_scale_change: float,
) -> bool:
    if current_state is None:
        return False
    if last_state is None:
        return True

    (cx, cy), scale = current_state
    (px, py), pscale = last_state

    center_jump = math.hypot(cx - px, cy - py)
    if center_jump > max_center_jump:
        return False

    rel_scale_change = abs(scale - pscale) / max(pscale, 1e-8)
    if rel_scale_change > max_scale_change:
        return False

    return True


def lower_body_median_jump(curr_lms, prev_lms, min_vis: float) -> float | None:
    jumps = []
    for idx in LOWER_BODY_IDS:
        c = curr_lms[idx]
        p = prev_lms[idx]
        if c.visibility < min_vis or p.visibility < min_vis:
            continue
        jumps.append(math.hypot(c.x - p.x, c.y - p.y))

    if not jumps:
        return None

    jumps.sort()
    n = len(jumps)
    mid = n // 2
    if n % 2 == 1:
        return jumps[mid]
    return 0.5 * (jumps[mid - 1] + jumps[mid])


def run(
    video: str | Path,
    start_seconds: float = 0.0,
    seconds: float = 20.0,
    output_dir: str | Path = "output/single_video",
    min_vis: float = 0.3,
    smooth_window: int = 5,
    min_event_gap_ms: float = 500.0,
    peak_threshold: float = 0.01,
    export_segment_video: bool = False,
    crop_left_ratio: float = 0.0,
    crop_right_ratio: float = 0.0,
    single_person_lock: bool = False,
    lock_max_center_jump: float = 0.12,
    lock_max_scale_change: float = 0.35,
    lock_min_keypoints: int = 6,
    lock_max_lower_median_jump: float = 0.06,
    model_path: str = DEFAULT_MODEL_PATH,
) -> dict:
    """Extract lower-limb joint angles and heel-strike events from one video.

    Returns a dict with paths to generated files and statistics.
    """
    video_path = Path(video)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    if seconds <= 0:
        raise ValueError("seconds must be > 0")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = video_path.stem
    start_tag = str(int(start_seconds)) if float(start_seconds).is_integer() else str(start_seconds).replace('.', 'p')
    end_tag = str(int(start_seconds + seconds)) if float(start_seconds + seconds).is_integer() else str(start_seconds + seconds).replace('.', 'p')
    segment_tag = f"{start_tag}to{end_tag}s"
    angles_csv = out_dir / f"{stem}_angles_{segment_tag}.csv"
    events_csv = out_dir / f"{stem}_events_{segment_tag}.csv"
    plot_png = out_dir / f"{stem}_angles_events_{segment_tag}.png"
    segment_video = out_dir / f"{stem}_selected_{segment_tag}.mp4"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    max_frames = int(seconds * fps)

    if start_seconds > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start_seconds * 1000.0))

    writer = None

    # Create PoseLandmarker via Tasks API
    base_options_obj = base_options.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options_obj,
        running_mode=vision.RunningMode.VIDEO,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    rows: list[dict[str, str]] = []
    lheel_y: list[tuple[float, float]] = []  # (time, y_rel_hip)
    rheel_y: list[tuple[float, float]] = []  # (time, y_rel_hip)

    frame_idx = 0
    lock_rejected_frames = 0
    lock_accepted_frames = 0
    last_locked_state: tuple[tuple[float, float], float] | None = None
    last_locked_landmarks = None
    writer_size: tuple[int, int] | None = None

    while frame_idx < max_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_bgr = crop_frame_lr(frame_bgr, crop_left_ratio, crop_right_ratio)

        if export_segment_video and writer is None:
            h, w = frame_bgr.shape[:2]
            if (w % 2) == 1:
                w -= 1
            if (h % 2) == 1:
                h -= 1
            frame_bgr = frame_bgr[:h, :w]
            writer = cv2.VideoWriter(
                str(segment_video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (w, h),
            )
            writer_size = (w, h)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int((start_seconds + frame_idx / fps) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        accepted_landmarks = None
        if result.pose_landmarks:
            if single_person_lock:
                pose_state = estimate_pose_state(
                    result.pose_landmarks[0],
                    min_vis,
                    lock_min_keypoints,
                )
                accept_by_motion = pass_single_person_lock(
                    pose_state,
                    last_locked_state,
                    lock_max_center_jump,
                    lock_max_scale_change,
                )

                accept_by_lower_jump = True
                if last_locked_landmarks is not None:
                    med_jump = lower_body_median_jump(
                        result.pose_landmarks[0],
                        last_locked_landmarks,
                        min_vis,
                    )
                    if med_jump is not None and med_jump > lock_max_lower_median_jump:
                        accept_by_lower_jump = False

                if accept_by_motion and accept_by_lower_jump:
                    accepted_landmarks = result.pose_landmarks[0]
                    last_locked_state = pose_state
                    last_locked_landmarks = result.pose_landmarks[0]
                    lock_accepted_frames += 1
                else:
                    lock_rejected_frames += 1
            else:
                accepted_landmarks = result.pose_landmarks[0]

        if accepted_landmarks is not None:
            vision.drawing_utils.draw_landmarks(
                frame_bgr,
                accepted_landmarks,
                vision.PoseLandmarksConnections.POSE_LANDMARKS,
            )
        elif single_person_lock:
            cv2.putText(
                frame_bgr,
                "LOCK_REJECT",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        if writer is not None:
            if writer_size is not None:
                ww, wh = writer_size
                fh, fw = frame_bgr.shape[:2]
                if (fw, fh) != (ww, wh):
                    frame_bgr = cv2.resize(frame_bgr, (ww, wh), interpolation=cv2.INTER_LINEAR)
            writer.write(frame_bgr)

        t = start_seconds + (frame_idx / fps)
        out = {
            "frame_idx": str(frame_idx),
            "time_s": f"{t:.6f}",
            "left_hip_deg": "",
            "right_hip_deg": "",
            "left_knee_deg": "",
            "right_knee_deg": "",
            "left_ankle_deg": "",
            "right_ankle_deg": "",
            "left_heel_y": "",
            "right_heel_y": "",
            # distance columns (normalized by shoulder width)
            "shoulder_width": "",
            "knee_width_norm": "",
            "ankle_width_norm": "",
            "base_of_support_norm": "",
            "hip_width_norm": "",
        }

        if accepted_landmarks is not None:
            lms = accepted_landmarks

            l_hip = joint_angle(lms, 11, 23, 25, min_vis)
            r_hip = joint_angle(lms, 12, 24, 26, min_vis)
            l_knee = joint_angle(lms, 23, 25, 27, min_vis)
            r_knee = joint_angle(lms, 24, 26, 28, min_vis)
            l_ankle = joint_angle(lms, 25, 27, 31, min_vis)
            r_ankle = joint_angle(lms, 26, 28, 32, min_vis)

            out["left_hip_deg"] = finite_or_empty(l_hip)
            out["right_hip_deg"] = finite_or_empty(r_hip)
            out["left_knee_deg"] = finite_or_empty(l_knee)
            out["right_knee_deg"] = finite_or_empty(r_knee)
            out["left_ankle_deg"] = finite_or_empty(l_ankle)
            out["right_ankle_deg"] = finite_or_empty(r_ankle)

            lh = point(lms, 29, min_vis)
            rh = point(lms, 30, min_vis)
            l_hip_pt = point(lms, 23, min_vis)
            r_hip_pt = point(lms, 24, min_vis)
            hip_center_y = (l_hip_pt[1] + r_hip_pt[1]) / 2 if (l_hip_pt and r_hip_pt) else None

            l_rel = lh[1] - hip_center_y if (lh and hip_center_y is not None) else None
            r_rel = rh[1] - hip_center_y if (rh and hip_center_y is not None) else None

            if l_rel is not None:
                out["left_heel_y"] = f"{l_rel:.6f}"
                lheel_y.append((t, l_rel))
            if r_rel is not None:
                out["right_heel_y"] = f"{r_rel:.6f}"
                rheel_y.append((t, r_rel))

            # Distance metrics normalized by shoulder width
            sw = joint_distance(lms, 11, 12, min_vis)  # shoulder width (anchor)
            if sw is not None and sw > 1e-8:
                out["shoulder_width"] = f"{sw:.6f}"

                # Hip width (left_hip to right_hip)
                hw = joint_distance(lms, 23, 24, min_vis)
                if hw is not None:
                    out["hip_width_norm"] = f"{safe_ratio(hw, sw):.6f}"

                # Knee width (left_knee to right_knee)
                kw = joint_distance(lms, 25, 26, min_vis)
                if kw is not None:
                    out["knee_width_norm"] = f"{safe_ratio(kw, sw):.6f}"

                # Ankle width (left_ankle to right_ankle)
                aw = joint_distance(lms, 27, 28, min_vis)
                if aw is not None:
                    out["ankle_width_norm"] = f"{safe_ratio(aw, sw):.6f}"

                # Step length: heel-to-heel distance (left heel idx=29, right heel idx=30)
                if lh is not None and rh is not None:
                    step_dist = math.hypot(lh[0] - rh[0], lh[1] - rh[1])
                    out["base_of_support_norm"] = f"{safe_ratio(step_dist, sw):.6f}"

        rows.append(out)
        frame_idx += 1

    cap.release()
    landmarker.close()
    if writer is not None:
        writer.release()

    with angles_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_idx",
                "time_s",
                "left_hip_deg",
                "right_hip_deg",
                "left_knee_deg",
                "right_knee_deg",
                "left_ankle_deg",
                "right_ankle_deg",
                "left_heel_y",
                "right_heel_y",
                "shoulder_width",
                "hip_width_norm",
                "knee_width_norm",
                "ankle_width_norm",
                "base_of_support_norm",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    left_events = detect_heel_strikes(
        [t for t, _ in lheel_y], [y for _, y in lheel_y],
        smooth_window, min_event_gap_ms, peak_threshold
    )
    right_events = detect_heel_strikes(
        [t for t, _ in rheel_y], [y for _, y in rheel_y],
        smooth_window, min_event_gap_ms, peak_threshold
    )

    with events_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["side", "event_time_s"])
        for t in left_events:
            writer.writerow(["left", f"{t:.6f}"])
        for t in right_events:
            writer.writerow(["right", f"{t:.6f}"])

    valid_t = [float(r["time_s"]) for r in rows]
    l_hip = [parse_float_or_nan(r["left_hip_deg"]) for r in rows]
    r_hip = [parse_float_or_nan(r["right_hip_deg"]) for r in rows]
    l_knee = [parse_float_or_nan(r["left_knee_deg"]) for r in rows]
    r_knee = [parse_float_or_nan(r["right_knee_deg"]) for r in rows]

    # Distance & symmetry metrics
    knee_width_norm = [parse_float_or_nan(r["knee_width_norm"]) for r in rows]
    ankle_width_norm = [parse_float_or_nan(r["ankle_width_norm"]) for r in rows]
    base_of_support_norm = [parse_float_or_nan(r["base_of_support_norm"]) for r in rows]
    hip_width_norm = [parse_float_or_nan(r["hip_width_norm"]) for r in rows]

    # Symmetry ratios: mean(min/max) (1.0 = perfectly symmetric)
    def sym_ratio(a: list[float], b: list[float]) -> float:
        valid = [(x, y) for x, y in zip(a, b) if math.isfinite(x) and math.isfinite(y) and x > 1e-8 and y > 1e-8]
        if not valid:
            return float("nan")
        return sum(min(x, y) / max(x, y) for x, y in valid) / len(valid)

    knee_sym = sym_ratio(knee_width_norm, knee_width_norm)
    ankle_sym = sym_ratio(ankle_width_norm, ankle_width_norm)

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)

    axes[0].plot(valid_t, l_hip, label="Left hip", linewidth=1.4)
    axes[0].plot(valid_t, r_hip, label="Right hip", linewidth=1.4)
    axes[0].set_ylabel("deg")
    axes[0].set_title("Hip Angles")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(valid_t, l_knee, label="Left knee", linewidth=1.4)
    axes[1].plot(valid_t, r_knee, label="Right knee", linewidth=1.4)
    axes[1].set_ylabel("deg")
    axes[1].set_title("Knee Angles")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best")

    # Gait timing: left + right heel-strikes
    all_ev = sorted(left_events + right_events)
    for i, t in enumerate(all_ev):
        color = "tab:blue" if t in left_events else "tab:orange"
        axes[2].vlines(t, 0.0, 1.0, color=color, alpha=0.8, linewidth=1.1,
                       label="Left" if i == 0 else ("Right" if i == len(left_events) else None))

    axes[2].set_ylim(-0.1, 1.1)
    axes[2].set_yticks([])
    axes[2].set_title(f"Gait Timing (Left={len(left_events)}, Right={len(right_events)})")
    axes[2].set_xlabel("time (s)")
    axes[2].grid(axis="x", alpha=0.3)
    axes[2].legend(loc="best")

    axes[3].plot(valid_t, hip_width_norm, label="Hip width", linewidth=1.2, alpha=0.7)
    axes[3].plot(valid_t, knee_width_norm, label="Knee width", linewidth=1.2, alpha=0.7)
    axes[3].plot(valid_t, ankle_width_norm, label="Ankle width", linewidth=1.2, alpha=0.7)
    axes[3].plot(valid_t, base_of_support_norm, label="Base of support (heel-heel)", linewidth=1.2, alpha=0.7)
    axes[3].set_ylabel("normalized")
    axes[3].set_title("Limb Widths & Step Length (shoulder-width normalized)")
    axes[3].grid(alpha=0.3)
    axes[3].legend(loc="best")

    fig.suptitle(f"{video_path.name} | Segment {segment_tag}", fontsize=12)
    fig.tight_layout()
    fig.savefig(plot_png, dpi=160)
    plt.close(fig)

    return {
        "video": str(video_path),
        "start_seconds": start_seconds,
        "seconds": seconds,
        "fps": fps,
        "frames_processed": frame_idx,
        "crop_left_ratio": crop_left_ratio,
        "crop_right_ratio": crop_right_ratio,
        "lock_accepted_frames": lock_accepted_frames if single_person_lock else None,
        "lock_rejected_frames": lock_rejected_frames if single_person_lock else None,
        "left_events": len(left_events),
        "right_events": len(right_events),
        "total_heel_strikes": len(left_events) + len(right_events),
        "cadence": f"{(len(left_events) + len(right_events)) / seconds:.2f}" if seconds > 0 else None,
        "knee_width_symmetry": f"{knee_sym:.4f}" if math.isfinite(knee_sym) else None,
        "ankle_width_symmetry": f"{ankle_sym:.4f}" if math.isfinite(ankle_sym) else None,
        "angles_csv": str(angles_csv),
        "events_csv": str(events_csv),
        "segment_video": str(segment_video) if export_segment_video else None,
        "plot_png": str(plot_png),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract lower-limb joint angles and heel-strike events from one video.")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--start-seconds", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--seconds", type=float, default=20.0, help="How many seconds to process")
    parser.add_argument("--output-dir", default="output/single_video", help="Output directory")
    parser.add_argument("--min-vis", type=float, default=0.3, help="Minimum landmark visibility")
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--min-event-gap-ms", type=float, default=350.0)
    parser.add_argument("--peak-threshold", type=float, default=0.001)
    parser.add_argument("--export-segment-video", action="store_true", help="Export the selected segment as a standalone video clip")
    parser.add_argument("--crop-left-ratio", type=float, default=0.0, help="Crop this ratio from the left side before pose inference")
    parser.add_argument("--crop-right-ratio", type=float, default=0.0, help="Crop this ratio from the right side before pose inference")
    parser.add_argument("--single-person-lock", action="store_true", help="Enable continuity lock to reject likely identity switches")
    parser.add_argument("--lock-max-center-jump", type=float, default=0.12)
    parser.add_argument("--lock-max-scale-change", type=float, default=0.35)
    parser.add_argument("--lock-min-keypoints", type=int, default=6)
    parser.add_argument("--lock-max-lower-median-jump", type=float, default=0.06)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to PoseLandmarker .task model file")
    args = parser.parse_args()

    result = run(
        video=args.video,
        start_seconds=args.start_seconds,
        seconds=args.seconds,
        output_dir=args.output_dir,
        min_vis=args.min_vis,
        smooth_window=args.smooth_window,
        min_event_gap_ms=args.min_event_gap_ms,
        peak_threshold=args.peak_threshold,
        export_segment_video=args.export_segment_video,
        crop_left_ratio=args.crop_left_ratio,
        crop_right_ratio=args.crop_right_ratio,
        single_person_lock=args.single_person_lock,
        lock_max_center_jump=args.lock_max_center_jump,
        lock_max_scale_change=args.lock_max_scale_change,
        lock_min_keypoints=args.lock_min_keypoints,
        lock_max_lower_median_jump=args.lock_max_lower_median_jump,
        model_path=args.model_path,
    )

    print("Done")
    for k, v in result.items():
        print(f"{k}={v}")


if __name__ == "__main__":
    main()