import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


LEFT_POINTS = {
    "left_ankle_y (27)": "lm_27_y",
    "left_heel_y (29)": "lm_29_y",
    "left_foot_index_y (31)": "lm_31_y",
}

RIGHT_POINTS = {
    "right_ankle_y (28)": "lm_28_y",
    "right_heel_y (30)": "lm_30_y",
    "right_foot_index_y (32)": "lm_32_y",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot vertical (y) trajectories of left/right foot landmarks in two panels."
    )
    parser.add_argument("--input-csv", required=True, help="Landmarks CSV from mediapipe_video_pose.py")
    parser.add_argument("--output-png", required=True, help="Output figure path")
    parser.add_argument(
        "--stabilize-by-hips",
        action="store_true",
        help="Subtract per-frame hip-center y (avg of lm_23_y and lm_24_y) from foot y",
    )
    return parser.parse_args()


def to_float(v: str) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def mean_or_none(a: float | None, b: float | None) -> float | None:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return 0.5 * (a + b)


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()

    in_csv = Path(args.input_csv)
    out_png = Path(args.output_png)

    if not in_csv.exists():
        raise FileNotFoundError(in_csv)

    rows = read_rows(in_csv)
    if not rows:
        raise RuntimeError("No rows in input CSV")

    t = []
    left_series = {k: [] for k in LEFT_POINTS}
    right_series = {k: [] for k in RIGHT_POINTS}

    for i, row in enumerate(rows):
        ts = to_float(row.get("timestamp_ms", ""))
        t.append(ts / 1000.0 if ts is not None else float(i))

        hip_l = to_float(row.get("lm_23_y", ""))
        hip_r = to_float(row.get("lm_24_y", ""))
        hip_center_y = mean_or_none(hip_l, hip_r)

        for label, col in LEFT_POINTS.items():
            y = to_float(row.get(col, ""))
            if args.stabilize_by_hips and y is not None and hip_center_y is not None:
                y = y - hip_center_y
            left_series[label].append(y)

        for label, col in RIGHT_POINTS.items():
            y = to_float(row.get(col, ""))
            if args.stabilize_by_hips and y is not None and hip_center_y is not None:
                y = y - hip_center_y
            right_series[label].append(y)

    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    for label, y in left_series.items():
        axes[0].plot(t, y, linewidth=1.3, label=label)
    if args.stabilize_by_hips:
        axes[0].set_title("Left Foot Vertical Position (y - hip_center_y)")
        axes[0].set_ylabel("relative y")
    else:
        axes[0].set_title("Left Foot Vertical Position (y)")
        axes[0].set_ylabel("normalized y")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best")

    for label, y in right_series.items():
        axes[1].plot(t, y, linewidth=1.3, label=label)
    if args.stabilize_by_hips:
        axes[1].set_title("Right Foot Vertical Position (y - hip_center_y)")
        axes[1].set_ylabel("relative y")
    else:
        axes[1].set_title("Right Foot Vertical Position (y)")
        axes[1].set_ylabel("normalized y")
    axes[1].set_xlabel("time (s)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    print("Done")
    print(f"input_csv={in_csv}")
    print(f"rows={len(rows)}")
    print(f"stabilize_by_hips={args.stabilize_by_hips}")
    print(f"output_png={out_png}")


if __name__ == "__main__":
    main()
