import argparse
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
import struct

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-frame timestamps from a video."
    )
    parser.add_argument(
        "--input",
        default="",
        help="Input video path. If omitted, auto-pick first video under data/raw/Video.",
    )
    parser.add_argument(
        "--output-csv",
        default="output/timing/video_timestamps.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional frame limit for quick checks (0 means all frames)",
    )
    parser.add_argument(
        "--start-time-utc",
        default="",
        help="Optional absolute start time in ISO-8601, e.g. 2026-03-24T10:00:00.000Z",
    )
    return parser.parse_args()


def resolve_input_video(input_arg: str) -> Path:
    if input_arg:
        return Path(input_arg)

    search_roots = [Path("data/raw/Video"), Path("Data/Video")]
    patterns = ("*.mp4", "*.avi", "*.mov", "*.mkv")

    candidates: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for pat in patterns:
            candidates.extend(sorted(root.rglob(pat)))

    if not candidates:
        raise FileNotFoundError(
            "No input video found. Pass --input, or put videos under data/raw/Video or Data/Video."
        )

    picked = candidates[0]
    print(f"No --input provided, auto-selected: {picked}")
    return picked


def parse_start_time_utc(text: str) -> datetime | None:
    if not text:
        return None

    normalized = text.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _read_atom_header(f):
    header = f.read(8)
    if len(header) < 8:
        return None, None, None

    size, atom_type = struct.unpack(">I4s", header)
    header_size = 8

    if size == 1:
        ext = f.read(8)
        if len(ext) < 8:
            return None, None, None
        size = struct.unpack(">Q", ext)[0]
        header_size = 16
    elif size == 0:
        cur = f.tell()
        f.seek(0, 2)
        end = f.tell()
        f.seek(cur)
        size = end - (cur - 8)

    return size, atom_type.decode("ascii", errors="replace"), header_size


def read_mp4_mvhd_creation_time_utc(video_path: Path) -> datetime | None:
    # ISO BMFF epoch starts at 1904-01-01 UTC.
    mp4_epoch = datetime(1904, 1, 1, tzinfo=timezone.utc)

    with video_path.open("rb") as f:
        f.seek(0, 2)
        file_end = f.tell()
        f.seek(0)

        while f.tell() < file_end:
            atom_start = f.tell()
            size, atom_type, header_size = _read_atom_header(f)
            if size is None or size < header_size:
                break

            atom_end = atom_start + size
            if atom_type == "moov":
                while f.tell() < atom_end:
                    child_start = f.tell()
                    child_size, child_type, child_header_size = _read_atom_header(f)
                    if child_size is None or child_size < child_header_size:
                        break

                    child_end = child_start + child_size
                    if child_type == "mvhd":
                        version_flags = f.read(4)
                        if len(version_flags) < 4:
                            return None
                        version = version_flags[0]

                        if version == 0:
                            creation_raw = f.read(4)
                            if len(creation_raw) < 4:
                                return None
                            creation_seconds = struct.unpack(">I", creation_raw)[0]
                        elif version == 1:
                            creation_raw = f.read(8)
                            if len(creation_raw) < 8:
                                return None
                            creation_seconds = struct.unpack(">Q", creation_raw)[0]
                        else:
                            return None

                        return mp4_epoch + timedelta(seconds=creation_seconds)

                    f.seek(child_end)

                return None

            f.seek(atom_end)

    return None


def main() -> None:
    args = parse_args()

    input_path = resolve_input_video(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    start_time_utc = parse_start_time_utc(args.start_time_utc)
    start_time_source = "manual"
    if start_time_utc is None:
        start_time_utc = read_mp4_mvhd_creation_time_utc(input_path)
        start_time_source = "media_mvhd" if start_time_utc is not None else "none"

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 0.0

    header = [
        "frame_idx",
        "time_sec_from_pts",
        "time_sec_from_fps",
        "delta_ms_pts_vs_fps",
    ]
    if start_time_utc is not None:
        header.append("absolute_utc_iso")

    frame_idx = 0

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        while True:
            ok, _ = cap.read()
            if not ok:
                break

            t_pts_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            t_fps_sec = (frame_idx / fps) if fps > 0 else ""

            if isinstance(t_fps_sec, float):
                delta_ms = (t_pts_sec - t_fps_sec) * 1000.0
            else:
                delta_ms = ""

            row = [frame_idx, t_pts_sec, t_fps_sec, delta_ms]

            if start_time_utc is not None:
                abs_dt = start_time_utc + timedelta(seconds=t_pts_sec)
                row.append(abs_dt.isoformat())

            writer.writerow(row)

            frame_idx += 1
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break

    cap.release()

    print("Export completed.")
    print(f"Input video: {input_path}")
    print(f"Frames exported: {frame_idx}")
    if fps > 0:
        print(f"Nominal FPS: {fps:.6f}")
    else:
        print("Nominal FPS: unavailable")
    print(f"Absolute start time source: {start_time_source}")
    if start_time_utc is not None:
        print(f"Absolute start time UTC: {start_time_utc.isoformat()}")
    else:
        print("Absolute start time UTC: unavailable")
    print(f"CSV: {output_csv}")


if __name__ == "__main__":
    main()
