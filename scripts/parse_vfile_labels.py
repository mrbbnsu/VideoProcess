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

    parsed: list[dict[str, str | int | None]] = []
    for parts in rows:
        subject_id = parts[0]
        subject_name = parts[1]
        assessment_text = parts[2]
        video_name = parts[3]
        exo_yn = parts[4]

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
