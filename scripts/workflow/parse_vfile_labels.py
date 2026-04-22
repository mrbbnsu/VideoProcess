import argparse
import csv
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse VFile.csv into structured labels for modeling."
    )
    parser.add_argument(
        "--input-csv",
        default="data/Accessment/VFile.csv",
        help="Raw label CSV path",
    )
    parser.add_argument(
        "--output-csv",
        default="data/Accessment/VFile_parsed.csv",
        help="Parsed structured CSV path",
    )
    return parser.parse_args()


def clean_header_key(v: str) -> str:
    return (v or "").replace("\ufeff", "").replace("\xa0", " ").strip().lower()

def read_text_with_fallback(path: Path) -> tuple[str, str]:
    encodings = ["utf-8-sig", "gb18030", "gbk", "gb2312", "utf-16"]
    for enc in encodings:
        try:
            return path.read_text(encoding=enc), enc
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="latin-1"), "latin-1"


def extract_score(text: str, key: str) -> int | None:
    # Supports patterns like FAC 3 / FAC3 / FAC：3 / FAC=3
    pattern = re.compile(rf"{re.escape(key)}\s*[:：=]?\s*(\d+)")
    m = pattern.search(text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def parse_bool_yno(v: str) -> int | None:
    s = (v or "").strip().lower()
    if s == "yes":
        return 1
    if s == "no":
        return 0
    return None


def parse_start_seconds(v: str) -> float | None:
    s = (v or "").strip()
    if not s:
        return None
    s = s.replace(":", "_")
    parts = [p for p in s.split("_") if p != ""]
    if len(parts) == 1:
        try:
            return float(parts[0])
        except ValueError:
            return None
    if len(parts) >= 2:
        try:
            mm = float(parts[0])
            ss = float(parts[1])
            return mm * 60.0 + ss
        except ValueError:
            return None
    return None


def normalize_side(v: str) -> str:
    s = (v or "").strip().lower()
    if s in {"left", "l", "左", "左侧"}:
        return "Left"
    if s in {"right", "r", "右", "右侧"}:
        return "Right"
    return ""


def main() -> None:
    args = parse_args()
    in_path = Path(args.input_csv)
    out_path = Path(args.output_csv)

    if not in_path.exists():
        raise FileNotFoundError(in_path)

    text, used_enc = read_text_with_fallback(in_path)

    rows: list[list[str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Raw file has no header, fields are comma-separated.
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        rows.append(parts)

    parsed: list[dict[str, str | int | float | None]] = []

    # If file already contains a header (e.g., manually enriched parsed file), reuse it.
    header = rows[0] if rows else []
    has_header = "video_name" in header or "subject_id" in header

    data_rows = rows[1:] if has_header else rows
    normalized_header = [clean_header_key(name) for name in header]
    col_index = {clean_header_key(name): i for i, name in enumerate(header)} if has_header else {}
    subject_pos_indices = []
    if has_header:
        subject_pos_indices = [
            i
            for i, name in enumerate(header)
            if clean_header_key(name) in {"subject_pos", "subject_pos_side"}
        ]

    for parts in data_rows:
        def pick(idx: int, key: str = "") -> str:
            if has_header and key:
                pos = col_index.get(clean_header_key(key))
                if pos is None or pos >= len(parts):
                    return ""
                return parts[pos]
            if idx >= len(parts):
                return ""
            return parts[idx]

        subject_id = parts[0]
        subject_name = parts[1]
        assessment_text = parts[2]
        video_name = parts[3]
        exo_yn = parts[4]

        if has_header:
            subject_id = pick(0, "subject_id")
            subject_name = pick(1, "subject_name")
            assessment_text = pick(2, "assessment_text")
            video_name = pick(3, "video_name")
            exo_yn = pick(4, "exo_yes_no")

        affected_side_raw = pick(10, "Affected Side")
        start_time_raw = pick(11, "start_time")
        subject_pos_side = pick(12, "subject_pos_side") or pick(12, "Subject_Pos")
        subject_crop_ratio_raw = (
            pick(13, "subject_crop_ratio")
            or pick(13, "Subject_Crop")
            or pick(13, "subject_crop")
        )

        # Handle duplicated Subject_Pos in manually edited CSV header.
        if has_header and subject_pos_indices:
            if subject_pos_indices[0] < len(parts):
                subject_pos_side = parts[subject_pos_indices[0]]
        crop_indices = []
        if has_header:
            crop_indices = [
                i
                for i, name in enumerate(header)
                if clean_header_key(name) in {"subject_crop", "subject_crop_ratio"}
            ]
        if has_header and crop_indices:
            if crop_indices[0] < len(parts):
                subject_crop_ratio_raw = parts[crop_indices[0]]

        try:
            subject_crop_ratio = float(subject_crop_ratio_raw) if subject_crop_ratio_raw else None
        except ValueError:
            subject_crop_ratio = None

        fac = extract_score(assessment_text, "FAC")
        bbs = extract_score(assessment_text, "BBS")
        tis = extract_score(assessment_text, "TIS")
        fma_le = extract_score(assessment_text, "FMA-LE")
        exo_on = parse_bool_yno(exo_yn)

        parsed.append(
            {
                "subject_id": subject_id,
                "subject_name": subject_name,
                "video_name": video_name,
                "exo_yes_no": exo_yn,
                "exo_on": exo_on,
                "FAC": fac,
                "BBS": bbs,
                "TIS": tis,
                "FMA_LE": fma_le,
                "assessment_text": assessment_text,
                "affected_side": normalize_side(affected_side_raw),
                "start_time": start_time_raw,
                "start_seconds": parse_start_seconds(start_time_raw),
                "subject_pos_side": normalize_side(subject_pos_side),
                "subject_crop_ratio": subject_crop_ratio,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "subject_id",
        "subject_name",
        "video_name",
        "exo_yes_no",
        "exo_on",
        "FAC",
        "BBS",
        "TIS",
        "FMA_LE",
        "assessment_text",
        "affected_side",
        "start_time",
        "start_seconds",
        "subject_pos_side",
        "subject_crop_ratio",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(parsed)

    # Basic summary for quick QC.
    n = len(parsed)
    n_exo_known = sum(1 for r in parsed if r["exo_on"] is not None)
    n_fac = sum(1 for r in parsed if r["FAC"] is not None)
    n_bbs = sum(1 for r in parsed if r["BBS"] is not None)
    n_tis = sum(1 for r in parsed if r["TIS"] is not None)
    n_fma = sum(1 for r in parsed if r["FMA_LE"] is not None)

    print("Done")
    print(f"input={in_path}")
    print(f"output={out_path}")
    print(f"detected_encoding={used_enc}")
    print(f"rows={n}")
    print(f"exo_on_known={n_exo_known}")
    print(f"FAC_found={n_fac}, BBS_found={n_bbs}, TIS_found={n_tis}, FMA_LE_found={n_fma}")


if __name__ == "__main__":
    main()
