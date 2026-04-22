import argparse
import csv
import math
from pathlib import Path

LOWER_IDS = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze frame-to-frame lower-body landmark jumps."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--jump-threshold", type=float, default=0.08)
    return parser.parse_args()


def to_float(v: str) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def main() -> None:
    args = parse_args()
    in_path = Path(args.input_csv)
    out_path = Path(args.output_csv)

    with in_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    out_rows: list[dict] = []

    prev = None
    for idx, row in enumerate(rows):
        frame_idx = int(row.get("frame_idx", idx)) if str(row.get("frame_idx", "")).isdigit() else idx
        t_ms = row.get("timestamp_ms", "")

        rec = {
            "frame_idx": frame_idx,
            "timestamp_ms": t_ms,
            "valid_pairs": 0,
            "mean_jump_xy": "",
            "max_jump_xy": "",
            "max_jump_joint": "",
            "is_jump_outlier": 0,
        }

        if prev is not None:
            jumps = []
            max_jump = -1.0
            max_joint = ""

            for lm_id in LOWER_IDS:
                x = to_float(row.get(f"lm_{lm_id}_x", ""))
                y = to_float(row.get(f"lm_{lm_id}_y", ""))
                px = to_float(prev.get(f"lm_{lm_id}_x", ""))
                py = to_float(prev.get(f"lm_{lm_id}_y", ""))

                if x is None or y is None or px is None or py is None:
                    continue

                d = math.hypot(x - px, y - py)
                jumps.append(d)
                if d > max_jump:
                    max_jump = d
                    max_joint = str(lm_id)

            rec["valid_pairs"] = len(jumps)
            if jumps:
                mean_jump = sum(jumps) / len(jumps)
                rec["mean_jump_xy"] = f"{mean_jump:.6f}"
                rec["max_jump_xy"] = f"{max_jump:.6f}"
                rec["max_jump_joint"] = max_joint
                rec["is_jump_outlier"] = 1 if max_jump >= args.jump_threshold else 0

        out_rows.append(rec)
        prev = row

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_idx",
                "timestamp_ms",
                "valid_pairs",
                "mean_jump_xy",
                "max_jump_xy",
                "max_jump_joint",
                "is_jump_outlier",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    valid = [r for r in out_rows if r["max_jump_xy"] != ""]
    outliers = [r for r in valid if r["is_jump_outlier"] == 1]
    max_jump_all = max([float(r["max_jump_xy"]) for r in valid], default=0.0)

    print("Done")
    print(f"input_csv={in_path}")
    print(f"output_csv={out_path}")
    print(f"rows={len(out_rows)}")
    print(f"valid_rows={len(valid)}")
    print(f"outlier_rows={len(outliers)}")
    print(f"max_jump_all={max_jump_all:.6f}")


if __name__ == "__main__":
    main()
