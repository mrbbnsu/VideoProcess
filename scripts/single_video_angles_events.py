import argparse
import csv
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp

from evaluate_gait_event_consistency import detect_heel_strikes


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
    return parser.parse_args()


def get_mp_solutions():
    if hasattr(mp, "solutions"):
        return mp.solutions
    from mediapipe import solutions as mp_solutions  # type: ignore

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


def main() -> None:
    args = parse_args()

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

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    max_frames = int(args.seconds * fps)

    if args.start_seconds > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, args.start_seconds * 1000.0))

    mp_solutions = get_mp_solutions()
    mp_pose = mp_solutions.pose

    rows: list[dict[str, str]] = []
    times: list[float] = []
    lheel_y: list[float] = []
    rheel_y: list[float] = []

    frame_idx = 0
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

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = pose.process(frame_rgb)

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

            if result.pose_landmarks is not None:
                lms = result.pose_landmarks.landmark

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

    # Quick visualization for sanity check.
    valid_t = [float(r["time_s"]) for r in rows if r["left_knee_deg"] and r["right_knee_deg"]]
    l_knee = [float(r["left_knee_deg"]) for r in rows if r["left_knee_deg"] and r["right_knee_deg"]]
    r_knee = [float(r["right_knee_deg"]) for r in rows if r["left_knee_deg"] and r["right_knee_deg"]]

    plt.figure(figsize=(12, 6))
    if valid_t:
        plt.plot(valid_t, l_knee, label="Left knee", linewidth=1.5)
        plt.plot(valid_t, r_knee, label="Right knee", linewidth=1.5)

    for i, t in enumerate(left_events):
        plt.axvline(t, color="tab:blue", alpha=0.18, linewidth=1, label="Left events" if i == 0 else None)
    for i, t in enumerate(right_events):
        plt.axvline(t, color="tab:orange", alpha=0.18, linewidth=1, label="Right events" if i == 0 else None)

    plt.title(f"{video_path.name} | Knee angles + detected events ({args.seconds:.1f}s)")
    plt.xlabel("time (s)")
    plt.ylabel("deg")
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(plot_png, dpi=160)
    plt.close()

    print("Done")
    print(f"video={video_path}")
    print(f"start_seconds={args.start_seconds}")
    print(f"seconds={args.seconds}")
    print(f"fps={fps:.6f}")
    print(f"frames_processed={frame_idx}")
    print(f"left_events={len(left_events)}")
    print(f"right_events={len(right_events)}")
    print(f"angles_csv={angles_csv}")
    print(f"events_csv={events_csv}")
    print(f"plot_png={plot_png}")


if __name__ == "__main__":
    main()
