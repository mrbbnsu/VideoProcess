import argparse
import csv
import subprocess
import sys
from pathlib import Path


VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single_video_angles_events.py by one row in VFile_parsed.csv."
    )
    parser.add_argument(
        "--parsed-csv",
        default="data/Accessment/VFile_parsed.csv",
        help="Path to VFile_parsed.csv",
    )
    parser.add_argument(
        "--line-number",
        type=int,
        required=True,
        help="1-based data row number in CSV (header excluded)",
    )
    parser.add_argument(
        "--line-number-kind",
        choices=["file", "data"],
        default="data",
        help="file: count with header; data: first data row is 1",
    )
    parser.add_argument(
        "--video-root",
        default="data/Accessment",
        help="Root directory to search video files",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=10.0,
        help="Duration to process",
    )
    parser.add_argument(
        "--output-dir",
        default="output/export",
        help="Output directory for extracted artifacts",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to run child script",
    )
    parser.add_argument(
        "--single-person-lock",
        dest="single_person_lock",
        action="store_true",
        help="Enable single person lock",
    )
    parser.add_argument(
        "--no-single-person-lock",
        dest="single_person_lock",
        action="store_false",
        help="Disable single person lock",
    )
    parser.set_defaults(single_person_lock=True)
    parser.add_argument(
        "--export-segment-video",
        dest="export_segment_video",
        action="store_true",
        help="Export selected segment video",
    )
    parser.add_argument(
        "--no-export-segment-video",
        dest="export_segment_video",
        action="store_false",
        help="Do not export selected segment video",
    )
    parser.set_defaults(export_segment_video=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved command only",
    )
    return parser.parse_args()


def parse_start_seconds(value: str) -> float:
    s = (value or "").strip()
    if not s:
        return 0.0
    try:
        return float(s)
    except ValueError:
        pass

    s = s.replace(":", "_")
    parts = [p for p in s.split("_") if p]
    if len(parts) >= 2:
        try:
            return float(parts[0]) * 60.0 + float(parts[1])
        except ValueError:
            return 0.0
    return 0.0


def clean_header_key(key: str) -> str:
    # Normalize invisible BOM/NBSP and surrounding whitespace in CSV headers.
    return (key or "").replace("\ufeff", "").replace("\xa0", " ").strip()


def row_get(row: dict[str, str], key: str, default: str = "") -> str:
    if key in row:
        return row.get(key, default)
    wanted = clean_header_key(key)
    for k, v in row.items():
        if clean_header_key(k) == wanted:
            return v
    return default


def read_target_row(path: Path, line_number: int, kind: str) -> tuple[dict[str, str], int]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if kind == "file":
        # DictReader starts from line 2 in the file.
        data_index = line_number - 2
    else:
        data_index = line_number - 1

    if data_index < 0 or data_index >= len(rows):
        raise IndexError(
            f"line out of range: line_number={line_number}, kind={kind}, data_rows={len(rows)}"
        )

    file_line = data_index + 2
    return rows[data_index], file_line


def resolve_video(video_name: str, video_root: Path) -> Path:
    name = (video_name or "").strip()
    if not name:
        raise ValueError("video_name is empty in selected row")

    direct = Path(name)
    if direct.exists():
        return direct.resolve()

    root_joined = video_root / name
    if root_joined.exists():
        return root_joined.resolve()

    stem = Path(name).stem
    candidates: list[Path] = []

    if video_root.exists():
        for ext in VIDEO_EXTS:
            candidates.extend(video_root.rglob(f"{stem}{ext}"))

    if not candidates:
        raise FileNotFoundError(
            f"cannot find video for '{name}' under {video_root.as_posix()}"
        )

    candidates = sorted(set(p.resolve() for p in candidates))
    return candidates[0]


def to_crop_ratios(subject_pos_side: str, subject_crop_ratio: str) -> tuple[float, float]:
    side = (subject_pos_side or "").strip().lower()
    try:
        keep_ratio = float(subject_crop_ratio) if subject_crop_ratio else 0.0
    except ValueError:
        keep_ratio = 0.0

    keep_ratio = max(0.0, min(keep_ratio, 1.0))
    crop_ratio = 1.0 - keep_ratio
    if side in {"right", "r", "右", "右侧"}:
        # Keep right keep_ratio of width -> crop left (1 - keep_ratio).
        return crop_ratio, 0.0
    if side in {"left", "l", "左", "左侧"}:
        # Keep left keep_ratio of width -> crop right (1 - keep_ratio).
        return 0.0, crop_ratio
    return 0.0, 0.0


def main() -> None:
    args = parse_args()

    parsed_csv = Path(args.parsed_csv)
    if not parsed_csv.exists():
        raise FileNotFoundError(parsed_csv)
    if args.seconds <= 0:
        raise ValueError("--seconds must be > 0")

    row, file_line = read_target_row(parsed_csv, args.line_number, args.line_number_kind)

    project_root = Path(__file__).resolve().parent.parent
    video_root = Path(args.video_root)
    if not video_root.is_absolute():
        video_root = (project_root / video_root).resolve()

    video_path = resolve_video(row_get(row, "video_name", ""), video_root)
    start_seconds = parse_start_seconds(row_get(row, "start_seconds", "") or row_get(row, "start_time", ""))
    crop_left, crop_right = to_crop_ratios(
        row_get(row, "subject_pos_side", ""), row_get(row, "subject_crop_ratio", "")
    )

    cmd = [
        args.python,
        str((project_root / "scripts" / "single_video_angles_events.py").resolve()),
        "--video",
        str(video_path),
        "--start-seconds",
        str(start_seconds),
        "--seconds",
        str(args.seconds),
        "--output-dir",
        str((project_root / args.output_dir).resolve()),
        "--crop-left-ratio",
        str(crop_left),
        "--crop-right-ratio",
        str(crop_right),
    ]
    if args.single_person_lock:
        cmd.append("--single-person-lock")
    if args.export_segment_video:
        cmd.append("--export-segment-video")

    print("Resolved row")
    print(f"- parsed_csv={parsed_csv.as_posix()}")
    print(f"- selected_file_line={file_line}")
    print(f"- video_name={row_get(row, 'video_name', '')}")
    print(f"- resolved_video={video_path.as_posix()}")
    print(f"- start_seconds={start_seconds}")
    print(f"- subject_pos_side={row_get(row, 'subject_pos_side', '')}")
    print(f"- subject_crop_ratio={row_get(row, 'subject_crop_ratio', '')}")
    print(f"- crop_left_ratio={crop_left}")
    print(f"- crop_right_ratio={crop_right}")
    print("RUN:")
    print(" ".join(cmd))

    if args.dry_run:
        return

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"child script failed with exit code {result.returncode}")


if __name__ == "__main__":
    main()
