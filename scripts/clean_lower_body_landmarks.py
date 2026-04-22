import argparse
import csv
from pathlib import Path

LOWER_IDS = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep only lower-body landmarks and suppress jump noise."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--median-k", type=int, default=5, help="Odd median window size")
    parser.add_argument("--max-step", type=float, default=0.05, help="Max per-frame step in normalized coords")
    parser.add_argument("--min-vis", type=float, default=0.5, help="Visibility threshold for valid points")
    return parser.parse_args()


def median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def to_float(v: str) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def median_filter(series: list[float | None], k: int) -> list[float | None]:
    if k < 3 or k % 2 == 0:
        return series[:]
    h = k // 2
    out: list[float | None] = []
    n = len(series)
    for i in range(n):
        lo = max(0, i - h)
        hi = min(n, i + h + 1)
        win = [x for x in series[lo:hi] if x is not None]
        out.append(median(win) if win else None)
    return out


def clamp_steps(series: list[float | None], max_step: float) -> list[float | None]:
    out: list[float | None] = []
    prev: float | None = None
    for x in series:
        if x is None:
            out.append(None)
            continue
        if prev is None:
            out.append(x)
            prev = x
            continue
        d = x - prev
        if d > max_step:
            x = prev + max_step
        elif d < -max_step:
            x = prev - max_step
        out.append(x)
        prev = x
    return out


def main() -> None:
    args = parse_args()
    in_path = Path(args.input_csv)
    out_path = Path(args.output_csv)

    with in_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    # Build per-landmark coordinate series, dropping low-visibility points.
    xs: dict[int, list[float | None]] = {i: [] for i in LOWER_IDS}
    ys: dict[int, list[float | None]] = {i: [] for i in LOWER_IDS}
    zs: dict[int, list[float | None]] = {i: [] for i in LOWER_IDS}
    vs: dict[int, list[float | None]] = {i: [] for i in LOWER_IDS}

    for row in rows:
        for i in LOWER_IDS:
            xv = to_float(row.get(f"lm_{i}_x", ""))
            yv = to_float(row.get(f"lm_{i}_y", ""))
            zv = to_float(row.get(f"lm_{i}_z", ""))
            vv = to_float(row.get(f"lm_{i}_visibility", ""))
            valid = (
                xv is not None
                and yv is not None
                and zv is not None
                and vv is not None
                and vv >= args.min_vis
            )
            if valid:
                xs[i].append(xv)
                ys[i].append(yv)
                zs[i].append(zv)
                vs[i].append(vv)
            else:
                xs[i].append(None)
                ys[i].append(None)
                zs[i].append(None)
                vs[i].append(vv)

    # Filter and clamp only x/y/z; keep raw visibility.
    for i in LOWER_IDS:
        xs[i] = clamp_steps(median_filter(xs[i], args.median_k), args.max_step)
        ys[i] = clamp_steps(median_filter(ys[i], args.median_k), args.max_step)
        zs[i] = clamp_steps(median_filter(zs[i], args.median_k), args.max_step)

    fields = ["frame_idx", "timestamp_ms"]
    for i in LOWER_IDS:
        fields.extend([f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z", f"lm_{i}_visibility"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for idx, row in enumerate(rows):
            out = {
                "frame_idx": row.get("frame_idx", idx),
                "timestamp_ms": row.get("timestamp_ms", ""),
            }
            for i in LOWER_IDS:
                x = xs[i][idx]
                y = ys[i][idx]
                z = zs[i][idx]
                v = vs[i][idx]
                out[f"lm_{i}_x"] = "" if x is None else f"{x:.6f}"
                out[f"lm_{i}_y"] = "" if y is None else f"{y:.6f}"
                out[f"lm_{i}_z"] = "" if z is None else f"{z:.6f}"
                out[f"lm_{i}_visibility"] = "" if v is None else f"{v:.6f}"
            w.writerow(out)

    print("Done")
    print(f"input_csv={in_path}")
    print(f"output_csv={out_path}")
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
