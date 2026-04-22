import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot A/B same-side joint angle comparison from angle CSV."
    )
    parser.add_argument("--angle-csv", required=True)
    parser.add_argument("--output-image", required=True)
    return parser.parse_args()


def to_float_array(rows: list[dict], key: str) -> np.ndarray:
    vals = []
    for r in rows:
        v = r.get(key, "")
        vals.append(float(v) if v else float("nan"))
    return np.array(vals, dtype=float)


def main() -> None:
    args = parse_args()

    angle_csv = Path(args.angle_csv)
    out_img = Path(args.output_image)
    out_img.parent.mkdir(parents=True, exist_ok=True)

    with angle_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError("No rows in angle CSV")

    t = to_float_array(rows, "time_s")

    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex=True)

    layout = [
        ("hip", "left", axes[0, 0]),
        ("hip", "right", axes[0, 1]),
        ("knee", "left", axes[1, 0]),
        ("knee", "right", axes[1, 1]),
        ("ankle", "left", axes[2, 0]),
        ("ankle", "right", axes[2, 1]),
    ]

    for joint, side, ax in layout:
        a_col = f"a_{side}_{joint}_deg"
        b_col = f"b_{side}_{joint}_deg"

        a = to_float_array(rows, a_col)
        b = to_float_array(rows, b_col)

        ax.plot(t, a, label=f"A {side} {joint}", linewidth=1.3)
        ax.plot(t, b, label=f"B {side} {joint}", linewidth=1.3)

        # Quick consistency summary in title
        valid = np.isfinite(a) & np.isfinite(b)
        if np.any(valid):
            mae = float(np.mean(np.abs(a[valid] - b[valid])))
            ax.set_title(f"{side.capitalize()} {joint.capitalize()} | A vs B (MAE={mae:.2f} deg)")
        else:
            ax.set_title(f"{side.capitalize()} {joint.capitalize()} | A vs B")

        ax.set_ylabel("deg")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    axes[2, 0].set_xlabel("time (s)")
    axes[2, 1].set_xlabel("time (s)")

    fig.suptitle("Cross-Camera Consistency: A/B Same-Side Joint Angles", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_img, dpi=170)
    plt.close(fig)

    print("Done")
    print(f"angle_csv={angle_csv}")
    print(f"output_image={out_img}")
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
