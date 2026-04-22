import argparse
import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evaluate_gait_event_consistency import detect_heel_strikes


@dataclass
class Point2D:
    x: float
    y: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-frame bilateral hip/knee/ankle angles, export CSV, plots, and per-cycle stats."
    )
    parser.add_argument("--sync-csv", required=True, help="Synced keypoints CSV from test_t1_sync.py")
    parser.add_argument("--output-dir", default="output/angles", help="Output directory")
    parser.add_argument("--min-vis", type=float, default=0.3, help="Minimum visibility threshold")
    parser.add_argument("--cycle-side", choices=["left", "right"], default="left", help="Heel side used for cycle boundaries")
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--min-event-gap-ms", type=float, default=120.0)
    parser.add_argument("--peak-threshold", type=float, default=0.0003)
    return parser.parse_args()


def parse_iso(ts: str) -> float:
    return datetime.fromisoformat(ts).timestamp()


def get_point(row: dict, prefix: str, lm_id: int, min_vis: float) -> Point2D | None:
    xk = f"{prefix}_lm_{lm_id}_x"
    yk = f"{prefix}_lm_{lm_id}_y"
    vk = f"{prefix}_lm_{lm_id}_vis"

    if not row.get(xk) or not row.get(yk) or not row.get(vk):
        return None

    vis = float(row[vk])
    if vis < min_vis:
        return None

    return Point2D(float(row[xk]), float(row[yk]))


def angle_abc(a: Point2D, b: Point2D, c: Point2D) -> float:
    bax = a.x - b.x
    bay = a.y - b.y
    bcx = c.x - b.x
    bcy = c.y - b.y

    n1 = math.hypot(bax, bay)
    n2 = math.hypot(bcx, bcy)
    if n1 < 1e-12 or n2 < 1e-12:
        return float("nan")

    cosv = (bax * bcx + bay * bcy) / (n1 * n2)
    cosv = max(-1.0, min(1.0, cosv))
    return math.degrees(math.acos(cosv))


def safe_joint_angle(row: dict, prefix: str, a_id: int, b_id: int, c_id: int, min_vis: float) -> float:
    a = get_point(row, prefix, a_id, min_vis)
    b = get_point(row, prefix, b_id, min_vis)
    c = get_point(row, prefix, c_id, min_vis)
    if a is None or b is None or c is None:
        return float("nan")
    return angle_abc(a, b, c)


def compute_angles_for_row(row: dict, min_vis: float) -> dict:
    out = {}
    for prefix in ["a", "b"]:
        # Hip: trunk-thigh angle (ipsilateral shoulder -> ipsilateral hip -> ipsilateral knee)
        out[f"{prefix}_left_hip_deg"] = safe_joint_angle(row, prefix, 11, 23, 25, min_vis)
        out[f"{prefix}_right_hip_deg"] = safe_joint_angle(row, prefix, 12, 24, 26, min_vis)

        # Knee: hip-knee-ankle
        out[f"{prefix}_left_knee_deg"] = safe_joint_angle(row, prefix, 23, 25, 27, min_vis)
        out[f"{prefix}_right_knee_deg"] = safe_joint_angle(row, prefix, 24, 26, 28, min_vis)

        # Ankle: knee-ankle-foot_index
        out[f"{prefix}_left_ankle_deg"] = safe_joint_angle(row, prefix, 25, 27, 31, min_vis)
        out[f"{prefix}_right_ankle_deg"] = safe_joint_angle(row, prefix, 26, 28, 32, min_vis)

    return out


def write_angle_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("No rows to write")
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def to_np(values: list[float]) -> np.ndarray:
    return np.array(values, dtype=float)


def parse_float_or_nan(value: str) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def plot_lr_curves(angle_rows: list[dict], out_png: Path, source_prefix: str) -> None:
    t = to_np([float(r["time_s"]) for r in angle_rows])

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    joints = ["hip", "knee", "ankle"]

    for ax, joint in zip(axes, joints):
        l = to_np([parse_float_or_nan(r.get(f"{source_prefix}_left_{joint}_deg", "")) for r in angle_rows])
        rr = to_np([parse_float_or_nan(r.get(f"{source_prefix}_right_{joint}_deg", "")) for r in angle_rows])

        ax.plot(t, l, label=f"{source_prefix.upper()} Left {joint}", linewidth=1.4)
        ax.plot(t, rr, label=f"{source_prefix.upper()} Right {joint}", linewidth=1.4)
        ax.set_ylabel("deg")
        ax.set_title(f"{source_prefix.upper()} {joint.capitalize()} Angle: Left vs Right")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def collect_cycle_events(raw_rows: list[dict], cycle_side: str, smooth_window: int, min_event_gap_ms: float, peak_threshold: float) -> list[float]:
    times = []
    heel_y = []

    heel_lm = 29 if cycle_side == "left" else 30

    for row in raw_rows:
        if row.get("a_detected") != "True":
            continue
        if not row.get("a_abs_utc") or not row.get(f"a_lm_{heel_lm}_y"):
            continue

        times.append(parse_iso(row["a_abs_utc"]))
        heel_y.append(float(row[f"a_lm_{heel_lm}_y"]))

    return detect_heel_strikes(times, heel_y, smooth_window, min_event_gap_ms, peak_threshold)


def finite_stats(arr: np.ndarray) -> tuple[float, float, float, float, float, int] | None:
    x = arr[np.isfinite(arr)]
    if x.size == 0:
        return None
    mean = float(np.mean(x))
    mn = float(np.min(x))
    mx = float(np.max(x))
    rom = float(mx - mn)
    std = float(np.std(x))
    return mean, mn, mx, rom, std, int(x.size)


def write_cycle_stats_csv(path: Path, angle_rows: list[dict], cycle_events_abs: list[float], t0_abs: float) -> int:
    if len(cycle_events_abs) < 2:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "cycle_idx",
                "start_s",
                "end_s",
                "duration_s",
                "source",
                "side",
                "joint",
                "mean_deg",
                "min_deg",
                "max_deg",
                "rom_deg",
                "std_deg",
                "n",
            ])
        return 0

    t = to_np([float(r["time_s"]) for r in angle_rows])

    headers = [
        "cycle_idx",
        "start_s",
        "end_s",
        "duration_s",
        "source",
        "side",
        "joint",
        "mean_deg",
        "min_deg",
        "max_deg",
        "rom_deg",
        "std_deg",
        "n",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        cycle_count = 0
        for i in range(len(cycle_events_abs) - 1):
            s = cycle_events_abs[i] - t0_abs
            e = cycle_events_abs[i + 1] - t0_abs
            if e <= s:
                continue

            mask = (t >= s) & (t < e)
            if not np.any(mask):
                continue

            for source in ["a", "b"]:
                for side in ["left", "right"]:
                    for joint in ["hip", "knee", "ankle"]:
                        col = f"{source}_{side}_{joint}_deg"
                        vals = to_np([parse_float_or_nan(r.get(col, "")) for r in angle_rows])[mask]
                        st = finite_stats(vals)
                        if st is None:
                            continue
                        mean, mn, mx, rom, std, n = st
                        writer.writerow(
                            [
                                i,
                                f"{s:.6f}",
                                f"{e:.6f}",
                                f"{(e - s):.6f}",
                                source,
                                side,
                                joint,
                                f"{mean:.6f}",
                                f"{mn:.6f}",
                                f"{mx:.6f}",
                                f"{rom:.6f}",
                                f"{std:.6f}",
                                n,
                            ]
                        )
            cycle_count += 1

    return cycle_count


def main() -> None:
    args = parse_args()

    sync_csv = Path(args.sync_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with sync_csv.open("r", encoding="utf-8", newline="") as f:
        raw_rows = list(csv.DictReader(f))

    usable_rows = []
    t0_abs = None
    for row in raw_rows:
        if row.get("a_detected") != "True" or row.get("b_detected") != "True":
            continue
        if not row.get("a_abs_utc"):
            continue

        ta_abs = parse_iso(row["a_abs_utc"])
        if t0_abs is None:
            t0_abs = ta_abs
        t_rel = ta_abs - t0_abs

        angle_map = compute_angles_for_row(row, args.min_vis)
        out = {
            "a_frame_idx": row["a_frame_idx"],
            "b_frame_idx": row["b_frame_idx"],
            "a_abs_utc": row["a_abs_utc"],
            "b_abs_utc": row["b_abs_utc"],
            "time_s": f"{t_rel:.6f}",
            "delta_ms_b_minus_a": row["delta_ms_b_minus_a"],
        }
        for k, v in angle_map.items():
            out[k] = f"{v:.6f}" if math.isfinite(v) else ""
        usable_rows.append(out)

    if not usable_rows:
        raise RuntimeError("No usable synced rows found")

    angle_csv = out_dir / f"{sync_csv.stem}_angles.csv"
    write_angle_csv(angle_csv, usable_rows)

    plot_a = out_dir / f"{sync_csv.stem}_A_left_right_curves.png"
    plot_b = out_dir / f"{sync_csv.stem}_B_left_right_curves.png"
    plot_lr_curves(usable_rows, plot_a, "a")
    plot_lr_curves(usable_rows, plot_b, "b")

    cycle_events = collect_cycle_events(
        raw_rows,
        args.cycle_side,
        args.smooth_window,
        args.min_event_gap_ms,
        args.peak_threshold,
    )

    cycle_stats_csv = out_dir / f"{sync_csv.stem}_cycle_stats.csv"
    cycle_count = write_cycle_stats_csv(cycle_stats_csv, usable_rows, cycle_events, t0_abs if t0_abs is not None else 0.0)

    print("Done")
    print(f"sync_csv={sync_csv}")
    print(f"usable_rows={len(usable_rows)}")
    print(f"angle_csv={angle_csv}")
    print(f"plot_a={plot_a}")
    print(f"plot_b={plot_b}")
    print(f"cycle_events={len(cycle_events)}")
    print(f"cycles_written={cycle_count}")
    print(f"cycle_stats_csv={cycle_stats_csv}")


if __name__ == "__main__":
    main()
