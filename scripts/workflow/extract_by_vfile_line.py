import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.workflow.single_video_angles_events import run


VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV")

# Default configuration
DEFAULT_PARSED_CSV = str(PROJECT_ROOT / "data/Accessment/VFile_parsed.csv")
DEFAULT_VIDEO_ROOT = str(PROJECT_ROOT / "data/Accessment")
DEFAULT_SECONDS = 10.0
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "output/export")
line_numbers: list[int] = [1, 2, 3, 4, 5]


def parse_start_seconds(value: str) -> float:
    s = (value or "").strip()
    if not s:
        return 0.0
    if "_" in s or ":" in s:
        s = s.replace(":", "_")
        parts = [p for p in s.split("_") if p]
        if len(parts) >= 2:
            try:
                return float(parts[0]) * 60.0 + float(parts[1])
            except ValueError:
                return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def clean_header_key(key: str) -> str:
    return (key or "").replace("﻿", "").replace("\xa0", " ").strip()


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
        data_index = line_number - 2
    else:
        data_index = line_number - 1

    if data_index < 0 or data_index >= len(rows):
        raise IndexError(f"line out of range: line_number={line_number}, kind={kind}, data_rows={len(rows)}")

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
        raise FileNotFoundError(f"cannot find video for '{name}' under {video_root.as_posix()}")

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
        return crop_ratio, 0.0
    if side in {"left", "l", "左", "左侧"}:
        return 0.0, crop_ratio
    return 0.0, 0.0


def main() -> None:
    parsed_csv = Path(DEFAULT_PARSED_CSV)
    seconds = DEFAULT_SECONDS
    output_dir = DEFAULT_OUTPUT_DIR

    if not parsed_csv.exists():
        raise FileNotFoundError(parsed_csv)
    if seconds <= 0:
        raise ValueError("seconds must be > 0")

    video_root = Path(DEFAULT_VIDEO_ROOT)
    if not video_root.is_absolute():
        video_root = (PROJECT_ROOT / video_root).resolve()

    for idx, line_number in enumerate(line_numbers):
        print(f"\n{'='*60}")
        print(f"Processing line {line_number} ({idx + 1}/{len(line_numbers)})")
        print(f"{'='*60}")

        row, file_line = read_target_row(parsed_csv, line_number, "data")

        video_path = resolve_video(row_get(row, "video_name", ""), video_root)
        start_seconds = parse_start_seconds(row_get(row, "start_seconds", "") or row_get(row, "start_time", ""))
        crop_left, crop_right = to_crop_ratios(
            row_get(row, "subject_pos_side", ""), row_get(row, "subject_crop_ratio", "")
        )

        print(f"- parsed_csv={parsed_csv.as_posix()}")
        print(f"- selected_file_line={file_line}")
        print(f"- video_name={row_get(row, 'video_name', '')}")
        print(f"- resolved_video={video_path.as_posix()}")
        print(f"- start_seconds={start_seconds}")
        print(f"- subject_pos_side={row_get(row, 'subject_pos_side', '')}")
        print(f"- subject_crop_ratio={row_get(row, 'subject_crop_ratio', '')}")
        print(f"- crop_left_ratio={crop_left}")
        print(f"- crop_right_ratio={crop_right}")

        result = run(
            video=video_path,
            start_seconds=start_seconds,
            seconds=seconds,
            output_dir=output_dir,
            crop_left_ratio=crop_left,
            crop_right_ratio=crop_right,
            single_person_lock=True,
            export_segment_video=True,
        )

        print("Result:")
        for k, v in result.items():
            print(f"  {k}={v}")


if __name__ == "__main__":
    main()