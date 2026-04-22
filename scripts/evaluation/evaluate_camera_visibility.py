import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp


JOINT_LANDMARKS = {
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate lower-limb landmark visibility quality for videos in a folder."
    )
    parser.add_argument("--video-dir", required=True, help="Folder containing videos")
    parser.add_argument("--seconds", type=float, default=20.0, help="Seconds to evaluate from each video")
    parser.add_argument("--min-vis", type=float, default=0.5, help="Visibility threshold")
    parser.add_argument("--output-csv", default="output/reports/video_visibility_ranking.csv")
    parser.add_argument("--output-md", default="output/reports/video_visibility_ranking.md")
    return parser.parse_args()


def get_mp_solutions():
    if hasattr(mp, "solutions"):
        return mp.solutions
    from mediapipe import solutions as mp_solutions  # type: ignore

    return mp_solutions


def iter_videos(video_dir: Path) -> list[Path]:
    exts = {".mp4", ".mov", ".avi", ".mkv"}
    vids = [p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(vids, key=lambda p: p.name.lower())


def evaluate_video(video_path: Path, seconds: float, min_vis: float) -> dict[str, float | int | str]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    max_frames = int(seconds * fps)

    mp_solutions = get_mp_solutions()
    mp_pose = mp_solutions.pose

    counts_visible = {k: 0 for k in JOINT_LANDMARKS}
    processed = 0
    detected_frames = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while processed < max_frames:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = pose.process(frame_rgb)
            processed += 1

            if result.pose_landmarks is None:
                continue

            detected_frames += 1
            lms = result.pose_landmarks.landmark
            for name, idx in JOINT_LANDMARKS.items():
                if lms[idx].visibility >= min_vis:
                    counts_visible[name] += 1

    cap.release()

    if processed == 0:
        raise RuntimeError(f"No frames processed: {video_path}")

    rates = {f"{k}_rate": (v / processed) for k, v in counts_visible.items()}
    lower_mean = sum(rates.values()) / len(rates)

    row: dict[str, float | int | str] = {
        "video": video_path.name,
        "fps": fps,
        "frames_processed": processed,
        "pose_detected_rate": detected_frames / processed,
        **rates,
        "lower_body_mean_rate": lower_mean,
    }
    return row


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "video",
        "fps",
        "frames_processed",
        "pose_detected_rate",
        "left_hip_rate",
        "right_hip_rate",
        "left_knee_rate",
        "right_knee_rate",
        "left_ankle_rate",
        "right_ankle_rate",
        "lower_body_mean_rate",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_md(path: Path, rows: list[dict[str, float | int | str]], seconds: float, min_vis: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Video Visibility Ranking")
    lines.append("")
    lines.append(f"- seconds evaluated per video: {seconds}")
    lines.append(f"- visibility threshold (min_vis): {min_vis}")
    lines.append("")
    lines.append("| rank | video | pose_detected | left_hip | right_hip | left_knee | right_knee | left_ankle | right_ankle | lower_body_mean |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for i, r in enumerate(rows, start=1):
        lines.append(
            "| {rank} | {video} | {p:.3f} | {lh:.3f} | {rh:.3f} | {lk:.3f} | {rk:.3f} | {la:.3f} | {ra:.3f} | {mean:.3f} |".format(
                rank=i,
                video=r["video"],
                p=r["pose_detected_rate"],
                lh=r["left_hip_rate"],
                rh=r["right_hip_rate"],
                lk=r["left_knee_rate"],
                rk=r["right_knee_rate"],
                la=r["left_ankle_rate"],
                ra=r["right_ankle_rate"],
                mean=r["lower_body_mean_rate"],
            )
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    video_dir = Path(args.video_dir)
    if not video_dir.exists() or not video_dir.is_dir():
        raise NotADirectoryError(video_dir)

    videos = iter_videos(video_dir)
    if not videos:
        raise RuntimeError(f"No video files found in: {video_dir}")

    rows: list[dict[str, float | int | str]] = []
    for i, vp in enumerate(videos, start=1):
        print(f"[{i}/{len(videos)}] evaluating {vp.name} ...")
        row = evaluate_video(vp, args.seconds, args.min_vis)
        rows.append(row)

    rows.sort(key=lambda r: float(r["lower_body_mean_rate"]), reverse=True)

    out_csv = Path(args.output_csv)
    out_md = Path(args.output_md)
    write_csv(out_csv, rows)
    write_md(out_md, rows, args.seconds, args.min_vis)

    print("Done")
    print(f"videos={len(rows)}")
    print(f"output_csv={out_csv}")
    print(f"output_md={out_md}")


if __name__ == "__main__":
    main()
