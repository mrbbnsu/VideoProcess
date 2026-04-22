import argparse
import csv
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp

from evaluate_gait_event_consistency import detect_heel_strikes


LOWER_BODY_IDS = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract lower-limb joint angles and heel-strike events from one video."
    )
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--start-seconds", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--seconds", type=float, default=20.0, help="How many seconds to process")
    parser.add_argument("--output-dir", default="output/single_video", help="Output directory")
    parser.add_argument("--min-vis", type=float, default=0.3, help="Minimum landmark visibility")
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--min-event-gap-ms", type=float, default=350.0)
    parser.add_argument("--peak-threshold", type=float, default=0.001)
    parser.add_argument(
        "--export-segment-video",
        action="store_true",
        help="Export the selected segment as a standalone video clip",
    )
    parser.add_argument(
        "--crop-left-ratio",
        type=float,
        default=0.0,
        help="Crop this ratio from the left side before pose inference (e.g., 0.2)",
    )
    parser.add_argument(
        "--crop-right-ratio",
        type=float,
        default=0.0,
        help="Crop this ratio from the right side before pose inference (e.g., 0.2)",
    )
    parser.add_argument(
        "--single-person-lock",
        action="store_true",
        help="Enable continuity lock to reject likely identity switches",
    )
    parser.add_argument(
        "--lock-max-center-jump",
        type=float,
        default=0.12,
        help="Maximum normalized center jump per frame for lock acceptance",
    )
    parser.add_argument(
        "--lock-max-scale-change",
        type=float,
        default=0.35,
        help="Maximum relative body scale change per frame for lock acceptance",
    )
    parser.add_argument(
        "--lock-min-keypoints",
        type=int,
        default=6,
        help="Minimum visible keypoints required to evaluate lock continuity",
    )
    parser.add_argument(
        "--lock-max-lower-median-jump",
        type=float,
        default=0.06,
        help="Maximum median frame-to-frame lower-body jump for lock acceptance",
    )
    return parser.parse_args()


def get_mp_solutions():
    if hasattr(mp, "solutions"):
        return mp.solutions
    from mediapipe.python import solutions as mp_solutions  # type: ignore

    return mp_solutions


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
    # OpenCV VideoWriter is more stable with contiguous arrays.
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


def main() -> None:
    args = parse_args()

    if args.seconds <= 0:
        raise ValueError("--seconds must be > 0")
    if args.min_vis < 0 or args.min_vis > 1:
        raise ValueError("--min-vis must be in [0, 1]")
    if args.lock_max_center_jump < 0:
        raise ValueError("--lock-max-center-jump must be >= 0")
    if args.lock_max_scale_change < 0:
        raise ValueError("--lock-max-scale-change must be >= 0")
    if args.lock_max_lower_median_jump < 0:
        raise ValueError("--lock-max-lower-median-jump must be >= 0")
    if args.lock_min_keypoints < 1:
        raise ValueError("--lock-min-keypoints must be >= 1")
    if args.crop_left_ratio < 0 or args.crop_left_ratio >= 1:
        raise ValueError("--crop-left-ratio must be in [0, 1)")
    if args.crop_right_ratio < 0 or args.crop_right_ratio >= 1:
        raise ValueError("--crop-right-ratio must be in [0, 1)")
    if args.crop_left_ratio + args.crop_right_ratio >= 1:
        raise ValueError("--crop-left-ratio + --crop-right-ratio must be < 1")

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = video_path.stem
    start_tag = str(int(args.start_seconds)) if float(args.start_seconds).is_integer() else str(args.start_seconds).replace('.', 'p')
    seconds_tag = str(int(args.seconds)) if float(args.seconds).is_integer() else str(args.seconds).replace('.', 'p')
    segment_tag = f"{start_tag}to{str(int(args.start_seconds + args.seconds)) if float(args.start_seconds + args.seconds).is_integer() else str(args.start_seconds + args.seconds).replace('.', 'p')}s"
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

    max_frames = int(args.seconds * fps)

    if args.start_seconds > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, args.start_seconds * 1000.0))

    writer = None

    mp_solutions = get_mp_solutions()
    mp_pose = mp_solutions.pose
    mp_draw = mp_solutions.drawing_utils

    rows: list[dict[str, str]] = []
    times: list[float] = []
    lheel_y: list[float] = []
    rheel_y: list[float] = []

    frame_idx = 0
    lock_rejected_frames = 0
    lock_accepted_frames = 0
    last_locked_state: tuple[tuple[float, float], float] | None = None
    last_locked_landmarks = None
    writer_size: tuple[int, int] | None = None
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while frame_idx < max_frames:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_bgr = crop_frame_lr(frame_bgr, args.crop_left_ratio, args.crop_right_ratio)

            if args.export_segment_video and writer is None:
                h, w = frame_bgr.shape[:2]
                # Some codecs expect even dimensions.
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
            result = pose.process(frame_rgb)

            accepted_landmarks = None
            if result.pose_landmarks is not None:
                if args.single_person_lock:
                    pose_state = estimate_pose_state(
                        result.pose_landmarks.landmark,
                        args.min_vis,
                        args.lock_min_keypoints,
                    )
                    accept_by_motion = pass_single_person_lock(
                        pose_state,
                        last_locked_state,
                        args.lock_max_center_jump,
                        args.lock_max_scale_change,
                    )

                    accept_by_lower_jump = True
                    if last_locked_landmarks is not None:
                        med_jump = lower_body_median_jump(
                            result.pose_landmarks.landmark,
                            last_locked_landmarks,
                            args.min_vis,
                        )
                        if med_jump is not None and med_jump > args.lock_max_lower_median_jump:
                            accept_by_lower_jump = False

                    if accept_by_motion and accept_by_lower_jump:
                        accepted_landmarks = result.pose_landmarks
                        last_locked_state = pose_state
                        last_locked_landmarks = result.pose_landmarks.landmark
                        lock_accepted_frames += 1
                    else:
                        lock_rejected_frames += 1
                else:
                    accepted_landmarks = result.pose_landmarks

            if accepted_landmarks is not None:
                mp_draw.draw_landmarks(
                    frame_bgr,
                    accepted_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )
            elif args.single_person_lock:
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
                    ww, hh = writer_size
                    fh, fw = frame_bgr.shape[:2]
                    if (fw, fh) != (ww, hh):
                        frame_bgr = cv2.resize(frame_bgr, (ww, hh), interpolation=cv2.INTER_LINEAR)
                writer.write(frame_bgr)

            t = args.start_seconds + (frame_idx / fps)
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
            }

            if accepted_landmarks is not None:
                lms = accepted_landmarks.landmark

                # Hip: trunk-thigh angle (ipsilateral shoulder -> ipsilateral hip -> ipsilateral knee)
                l_hip = joint_angle(lms, 11, 23, 25, args.min_vis)
                r_hip = joint_angle(lms, 12, 24, 26, args.min_vis)
                l_knee = joint_angle(lms, 23, 25, 27, args.min_vis)
                r_knee = joint_angle(lms, 24, 26, 28, args.min_vis)
                l_ankle = joint_angle(lms, 25, 27, 31, args.min_vis)
                r_ankle = joint_angle(lms, 26, 28, 32, args.min_vis)

                out["left_hip_deg"] = finite_or_empty(l_hip)
                out["right_hip_deg"] = finite_or_empty(r_hip)
                out["left_knee_deg"] = finite_or_empty(l_knee)
                out["right_knee_deg"] = finite_or_empty(r_knee)
                out["left_ankle_deg"] = finite_or_empty(l_ankle)
                out["right_ankle_deg"] = finite_or_empty(r_ankle)

                lh = point(lms, 29, args.min_vis)
                rh = point(lms, 30, args.min_vis)
                if lh is not None:
                    out["left_heel_y"] = f"{lh[1]:.6f}"
                if rh is not None:
                    out["right_heel_y"] = f"{rh[1]:.6f}"

                if lh is not None and rh is not None:
                    times.append(t)
                    lheel_y.append(lh[1])
                    rheel_y.append(rh[1])

            rows.append(out)
            frame_idx += 1

    cap.release()
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
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    left_events = detect_heel_strikes(
        times,
        lheel_y,
        args.smooth_window,
        args.min_event_gap_ms,
        args.peak_threshold,
    )
    right_events = detect_heel_strikes(
        times,
        rheel_y,
        args.smooth_window,
        args.min_event_gap_ms,
        args.peak_threshold,
    )

    with events_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["side", "event_time_s"])
        for t in left_events:
            writer.writerow(["left", f"{t:.6f}"])
        for t in right_events:
            writer.writerow(["right", f"{t:.6f}"])

    # Multi-panel visualization: hip, knee, and gait timing.
    valid_t = [float(r["time_s"]) for r in rows]
    l_hip = [parse_float_or_nan(r["left_hip_deg"]) for r in rows]
    r_hip = [parse_float_or_nan(r["right_hip_deg"]) for r in rows]
    l_knee = [parse_float_or_nan(r["left_knee_deg"]) for r in rows]
    r_knee = [parse_float_or_nan(r["right_knee_deg"]) for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

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

    for i, t in enumerate(left_events):
        axes[2].vlines(t, 0.6, 1.0, color="tab:blue", alpha=0.8, linewidth=1.1, label="Left heel-strike" if i == 0 else None)
    for i, t in enumerate(right_events):
        axes[2].vlines(t, 0.0, 0.4, color="tab:orange", alpha=0.8, linewidth=1.1, label="Right heel-strike" if i == 0 else None)

    axes[2].set_ylim(-0.1, 1.1)
    axes[2].set_yticks([0.2, 0.8])
    axes[2].set_yticklabels(["Right", "Left"])
    axes[2].set_title("Gait Timing (Heel-Strike Events)")
    axes[2].set_xlabel("time (s)")
    axes[2].grid(axis="x", alpha=0.3)
    axes[2].legend(loc="best")

    fig.suptitle(f"{video_path.name} | Segment {segment_tag}", fontsize=12)
    fig.tight_layout()
    fig.savefig(plot_png, dpi=160)
    plt.close(fig)

    print("Done")
    print(f"video={video_path}")
    print(f"start_seconds={args.start_seconds}")
    print(f"seconds={args.seconds}")
    print(f"fps={fps:.6f}")
    print(f"frames_processed={frame_idx}")
    print(f"crop_left_ratio={args.crop_left_ratio}")
    print(f"crop_right_ratio={args.crop_right_ratio}")
    if args.single_person_lock:
        print(f"lock_accepted_frames={lock_accepted_frames}")
        print(f"lock_rejected_frames={lock_rejected_frames}")
    print(f"left_events={len(left_events)}")
    print(f"right_events={len(right_events)}")
    print(f"angles_csv={angles_csv}")
    print(f"events_csv={events_csv}")
    if args.export_segment_video:
        print(f"segment_video={segment_video}")
    print(f"plot_png={plot_png}")


if __name__ == "__main__":
    main()
