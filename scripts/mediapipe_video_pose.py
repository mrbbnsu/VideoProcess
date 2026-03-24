import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MediaPipe Pose on a video and export annotated video + landmarks CSV."
    )
    parser.add_argument(
        "--input",
        required=False,
        default="",
        help="Input video path. If omitted, auto-pick first video under data/raw/Video.",
    )
    parser.add_argument(
        "--output-video",
        default="output/mediapipe/pose_annotated.mp4",
        help="Output annotated video path",
    )
    parser.add_argument(
        "--output-csv",
        default="output/mediapipe/pose_landmarks.csv",
        help="Output landmarks CSV path",
    )
    parser.add_argument(
        "--model-complexity",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="MediaPipe Pose model complexity",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.5,
        help="Minimum tracking confidence",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional frame limit for quick tests (0 means all frames)",
    )
    return parser.parse_args()


def build_csv_header() -> list[str]:
    header = ["frame_idx", "timestamp_ms"]
    for i in range(33):
        header.extend(
            [
                f"lm_{i}_x",
                f"lm_{i}_y",
                f"lm_{i}_z",
                f"lm_{i}_visibility",
            ]
        )
    return header


def resolve_input_video(input_arg: str) -> Path:
    if input_arg:
        return Path(input_arg)

    search_roots = [Path("data/raw/Video"), Path("Data/Video")]
    exts = ("*.mp4", "*.avi", "*.mov", "*.mkv")

    candidates: list[Path] = []
    for root in search_roots:
        if root.exists():
            for ext in exts:
                candidates.extend(sorted(root.rglob(ext)))

    if not candidates:
        raise FileNotFoundError(
            "No input video found. Pass --input, or put a video under data/raw/Video or Data/Video."
        )

    picked = candidates[0]
    print(f"No --input provided, auto-selected: {picked}")
    return picked


def get_mp_solutions():
    if hasattr(mp, "solutions"):
        return mp.solutions

    # Compatibility for newer mediapipe builds exposing solutions under mediapipe.python.
    from mediapipe.python import solutions as mp_solutions  # type: ignore

    return mp_solutions


def main() -> None:
    args = parse_args()

    input_path = resolve_input_video(args.input)
    output_video_path = Path(args.output_video)
    output_csv_path = Path(args.output_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    mp_solutions = get_mp_solutions()
    mp_pose = mp_solutions.pose
    mp_draw = mp_solutions.drawing_utils

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as pose, output_csv_path.open("w", newline="", encoding="utf-8") as f_csv:
        csv_writer = csv.writer(f_csv)
        csv_writer.writerow(build_csv_header())

        frame_idx = 0
        detected_frames = 0

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = pose.process(frame_rgb)

            row = [frame_idx, int((frame_idx / fps) * 1000)]

            if result.pose_landmarks:
                detected_frames += 1
                mp_draw.draw_landmarks(
                    frame_bgr,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )

                for lm in result.pose_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z, lm.visibility])
            else:
                for _ in range(33):
                    row.extend(["", "", "", ""])

            csv_writer.writerow(row)
            writer.write(frame_bgr)

            frame_idx += 1
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break

    cap.release()
    writer.release()

    print("MediaPipe pose processing completed.")
    print(f"Input: {input_path}")
    print(f"Frames processed: {frame_idx}")
    print(f"Frames with pose detected: {detected_frames}")
    print(f"Annotated video: {output_video_path}")
    print(f"Landmarks CSV: {output_csv_path}")


if __name__ == "__main__":
    main()
