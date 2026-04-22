import argparse
import csv
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import mediapipe as mp

from export_video_timestamps import read_mp4_mvhd_creation_time_utc

LOWER_BODY_IDS = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]


@dataclass
class FramePose:
    frame_idx: int
    rel_time_sec: float
    abs_time_utc: datetime
    detected: bool
    landmarks: dict[int, tuple[float, float, float, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test sync between T1_1 and T1_2 using MediaPipe pose timestamps."
    )
    parser.add_argument("--video-a", default="D:\\AtWork\\Algrithrom\\VideoPoseRecognization\\data\\Subject\\qxf4.MOV")
    parser.add_argument("--video-b", default="D:\\AtWork\\Algrithrom\\VideoPoseRecognization\\data\\Subject\\qxf5.MOV")
    parser.add_argument(
        "--output-csv",
        default="output/sync/T1_1_T1_2_sync.csv",
        help="Matched frame pairs with lower-body landmarks",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional frame cap per video (0 means all)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Process every Nth frame (1 means every frame)",
    )
    parser.add_argument(
        "--max-delta-ms",
        type=float,
        default=20.0,
        help="Maximum allowed time gap to consider a pair synced",
    )
    parser.add_argument(
        "--compensate-start-offset",
        action="store_true",
        help="Shift video B timeline by media start-time difference before matching.",
    )
    parser.add_argument(
        "--extra-offset-ms-b",
        type=float,
        default=0.0,
        help="Extra manual/event-based offset for video B in milliseconds (applied during matching).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Log progress every N processed frames per video (0 disables progress logs).",
    )
    return parser.parse_args()


def get_mp_solutions():
    if hasattr(mp, "solutions"):
        return mp.solutions
    from mediapipe.python import solutions as mp_solutions  # type: ignore

    return mp_solutions


def extract_pose_timeline(
    video_path: Path,
    max_frames: int,
    stride: int,
    progress_every: int,
) -> tuple[list[FramePose], float, datetime]:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    start_time_utc = read_mp4_mvhd_creation_time_utc(video_path)
    if start_time_utc is None:
        raise RuntimeError(f"Cannot read media creation time (mvhd) from: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 0.0

    mp_solutions = get_mp_solutions()
    mp_pose = mp_solutions.pose

    frames: list[FramePose] = []
    frame_idx = 0
    start_ts = time.perf_counter()
    total_estimate = max_frames if max_frames > 0 else None

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if stride > 1 and (frame_idx % stride != 0):
                frame_idx += 1
                continue

            rel_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            abs_time_utc = start_time_utc + timedelta(seconds=rel_time_sec)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = pose.process(frame_rgb)

            landmarks: dict[int, tuple[float, float, float, float]] = {}
            detected = result.pose_landmarks is not None
            if detected:
                for idx in LOWER_BODY_IDS:
                    lm = result.pose_landmarks.landmark[idx]
                    landmarks[idx] = (lm.x, lm.y, lm.z, lm.visibility)

            frames.append(
                FramePose(
                    frame_idx=frame_idx,
                    rel_time_sec=rel_time_sec,
                    abs_time_utc=abs_time_utc,
                    detected=detected,
                    landmarks=landmarks,
                )
            )

            if progress_every > 0 and len(frames) % progress_every == 0:
                elapsed = time.perf_counter() - start_ts
                fps_proc = len(frames) / elapsed if elapsed > 0 else 0.0
                if total_estimate is not None and fps_proc > 0:
                    remaining = max(0.0, (total_estimate - len(frames)) / fps_proc)
                    print(
                        f"[{video_path.name}] processed={len(frames)}/{total_estimate} "
                        f"(raw_frame_idx={frame_idx}) elapsed={elapsed:.1f}s "
                        f"speed={fps_proc:.2f} fps eta={remaining:.1f}s"
                    )
                else:
                    print(
                        f"[{video_path.name}] processed={len(frames)} "
                        f"(raw_frame_idx={frame_idx}) elapsed={elapsed:.1f}s "
                        f"speed={fps_proc:.2f} fps"
                    )

            frame_idx += 1
            if max_frames > 0 and len(frames) >= max_frames:
                break

    cap.release()
    return frames, fps, start_time_utc


def nearest_sync_pairs(
    timeline_a: list[FramePose],
    timeline_b: list[FramePose],
    max_delta_ms: float,
    offset_sec_b: float = 0.0,
) -> list[tuple[FramePose, FramePose, float]]:
    pairs: list[tuple[FramePose, FramePose, float]] = []
    j = 0

    for a in timeline_a:
        while j + 1 < len(timeline_b):
            curr = abs(
                (timeline_b[j].abs_time_utc + timedelta(seconds=offset_sec_b) - a.abs_time_utc).total_seconds()
            )
            nxt = abs(
                (timeline_b[j + 1].abs_time_utc + timedelta(seconds=offset_sec_b) - a.abs_time_utc).total_seconds()
            )
            if nxt <= curr:
                j += 1
            else:
                break

        b = timeline_b[j]
        delta_ms = (
            (b.abs_time_utc + timedelta(seconds=offset_sec_b) - a.abs_time_utc).total_seconds() * 1000.0
        )
        if abs(delta_ms) <= max_delta_ms:
            pairs.append((a, b, delta_ms))

    return pairs


def build_csv_header() -> list[str]:
    header = [
        "a_frame_idx",
        "a_abs_utc",
        "b_frame_idx",
        "b_abs_utc",
        "delta_ms_b_minus_a",
        "a_detected",
        "b_detected",
    ]

    for prefix in ["a", "b"]:
        for idx in LOWER_BODY_IDS:
            header.extend(
                [
                    f"{prefix}_lm_{idx}_x",
                    f"{prefix}_lm_{idx}_y",
                    f"{prefix}_lm_{idx}_z",
                    f"{prefix}_lm_{idx}_vis",
                ]
            )

    return header


def lm_values(frame: FramePose, idx: int) -> list[float | str]:
    if idx in frame.landmarks:
        x, y, z, v = frame.landmarks[idx]
        return [x, y, z, v]
    return ["", "", "", ""]


def write_pairs_csv(output_csv: Path, pairs: list[tuple[FramePose, FramePose, float]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(build_csv_header())

        for a, b, delta_ms in pairs:
            row = [
                a.frame_idx,
                a.abs_time_utc.isoformat(),
                b.frame_idx,
                b.abs_time_utc.isoformat(),
                delta_ms,
                a.detected,
                b.detected,
            ]

            for idx in LOWER_BODY_IDS:
                row.extend(lm_values(a, idx))
            for idx in LOWER_BODY_IDS:
                row.extend(lm_values(b, idx))

            writer.writerow(row)


def main() -> None:
    args = parse_args()

    video_a = Path(args.video_a)
    video_b = Path(args.video_b)

    timeline_a, fps_a, start_a = extract_pose_timeline(
        video_a,
        args.max_frames,
        args.stride,
        args.progress_every,
    )
    timeline_b, fps_b, start_b = extract_pose_timeline(
        video_b,
        args.max_frames,
        args.stride,
        args.progress_every,
    )

    offset_sec_b = 0.0
    if args.compensate_start_offset:
        # Shift B to A timeline so matching focuses on relative frame timing.
        offset_sec_b = (start_a - start_b).total_seconds()

    # Apply additional fine offset (e.g., from gait-event alignment).
    offset_sec_b += args.extra_offset_ms_b / 1000.0

    pairs = nearest_sync_pairs(
        timeline_a,
        timeline_b,
        args.max_delta_ms,
        offset_sec_b=offset_sec_b,
    )

    output_csv = Path(args.output_csv)
    write_pairs_csv(output_csv, pairs)

    print("Sync test completed.")
    print(f"Video A: {video_a}")
    print(f"Video B: {video_b}")
    print(f"A start UTC (media mvhd): {start_a.isoformat()}")
    print(f"B start UTC (media mvhd): {start_b.isoformat()}")
    print(f"B offset compensation (sec): {offset_sec_b:.6f}")
    print(f"A fps: {fps_a:.6f}")
    print(f"B fps: {fps_b:.6f}")
    print(f"A frames used: {len(timeline_a)}")
    print(f"B frames used: {len(timeline_b)}")
    print(f"Pairs within +/-{args.max_delta_ms} ms: {len(pairs)}")
    if pairs:
        deltas = [p[2] for p in pairs]
        mean_abs = sum(abs(d) for d in deltas) / len(deltas)
        print(f"Mean abs delta (ms): {mean_abs:.3f}")
    print(f"Output CSV: {output_csv}")


if __name__ == "__main__":
    main()
