import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate gait-event consistency (heel-strike timing difference) from synced keypoint CSV."
    )
    parser.add_argument("--sync-csv", required=True, help="Output CSV from test_t1_sync.py")
    parser.add_argument(
        "--output-json",
        default="output/sync/gait_event_consistency.json",
        help="Where to write metric summary",
    )
    parser.add_argument(
        "--max-match-ms",
        type=float,
        default=120.0,
        help="Maximum time difference allowed when matching events",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=3,
        help="Odd moving-average window for y smoothing",
    )
    parser.add_argument(
        "--min-event-gap-ms",
        type=float,
        default=120.0,
        help="Minimum gap between consecutive events of the same foot",
    )
    parser.add_argument(
        "--peak-threshold",
        type=float,
        default=0.0003,
        help="Minimum peak amplitude above local baseline in normalized image y",
    )
    parser.add_argument(
        "--write-aligned-csv",
        default="",
        help="Optional path to write CSV with gait-event aligned B timestamps and deltas.",
    )
    return parser.parse_args()


def parse_iso(ts: str) -> float:
    return datetime.fromisoformat(ts).timestamp()


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or window % 2 == 0 or len(values) < window:
        return values[:]

    half = window // 2
    out = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def detect_heel_strikes(
    times: list[float],
    y_values: list[float],
    smooth_window: int,
    min_event_gap_ms: float,
    peak_threshold: float,
) -> list[float]:
    if len(times) < 3:
        return []

    y = moving_average(y_values, smooth_window)

    # Adaptive floor for low-motion sequences: combine absolute and relative threshold.
    signal_range = max(y) - min(y)
    adaptive_threshold = max(peak_threshold, signal_range * 0.015)

    # Heel-strike proxy: heel point reaches a local maximum in image y (lowest in world).
    peak_indices: list[int] = []
    last_peak_time = None
    min_gap_sec = min_event_gap_ms / 1000.0

    for i in range(1, len(y) - 1):
        if not (y[i] >= y[i - 1] and y[i] >= y[i + 1]):
            continue

        lo = max(0, i - 4)
        hi = min(len(y), i + 5)
        local_min = min(y[lo:hi])
        if (y[i] - local_min) < adaptive_threshold:
            continue

        t = times[i]
        if last_peak_time is not None and (t - last_peak_time) < min_gap_sec:
            continue

        peak_indices.append(i)
        last_peak_time = t

    return [times[i] for i in peak_indices]


def match_events(a_times: list[float], b_times: list[float], max_match_ms: float) -> list[float]:
    if not a_times or not b_times:
        return []

    max_sec = max_match_ms / 1000.0
    deltas_ms: list[float] = []

    j = 0
    used_b = set()

    for ta in a_times:
        while j + 1 < len(b_times) and abs(b_times[j + 1] - ta) <= abs(b_times[j] - ta):
            j += 1

        candidates = [j]
        if j - 1 >= 0:
            candidates.append(j - 1)
        if j + 1 < len(b_times):
            candidates.append(j + 1)

        best_idx = None
        best_abs = None
        for idx in candidates:
            if idx in used_b:
                continue
            d = b_times[idx] - ta
            ad = abs(d)
            if ad <= max_sec and (best_abs is None or ad < best_abs):
                best_abs = ad
                best_idx = idx

        if best_idx is not None:
            used_b.add(best_idx)
            deltas_ms.append((b_times[best_idx] - ta) * 1000.0)

    return deltas_ms


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]

    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    w = pos - lo
    return sorted_values[lo] * (1.0 - w) + sorted_values[hi] * w


def median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def summarize(deltas_ms: list[float]) -> dict:
    if not deltas_ms:
        return {
            "matched_events": 0,
            "mean_ms": None,
            "mean_abs_ms": None,
            "p95_abs_ms": None,
            "max_abs_ms": None,
        }

    abs_vals = [abs(v) for v in deltas_ms]
    abs_sorted = sorted(abs_vals)

    return {
        "matched_events": len(deltas_ms),
        "mean_ms": sum(deltas_ms) / len(deltas_ms),
        "mean_abs_ms": sum(abs_vals) / len(abs_vals),
        "p95_abs_ms": percentile(abs_sorted, 0.95),
        "max_abs_ms": max(abs_vals),
    }


def shift_iso_time(iso_ts: str, offset_ms: float) -> str:
    dt = datetime.fromisoformat(iso_ts)
    shifted = dt.timestamp() - (offset_ms / 1000.0)
    return datetime.fromtimestamp(shifted, tz=dt.tzinfo).isoformat()


def main() -> None:
    args = parse_args()

    sync_csv = Path(args.sync_csv)
    output_json = Path(args.output_json)

    if not sync_csv.exists():
        raise FileNotFoundError(f"Sync CSV not found: {sync_csv}")

    raw_rows: list[dict] = []
    times: list[float] = []
    a_lheel_y: list[float] = []
    a_rheel_y: list[float] = []
    b_lheel_y: list[float] = []
    b_rheel_y: list[float] = []

    with sync_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_rows.append(row)
            if row.get("a_detected") != "True" or row.get("b_detected") != "True":
                continue
            if not row.get("a_abs_utc"):
                continue

            need_keys = ["a_lm_29_y", "a_lm_30_y", "b_lm_29_y", "b_lm_30_y"]
            if any(not row.get(k) for k in need_keys):
                continue

            times.append(parse_iso(row["a_abs_utc"]))
            a_lheel_y.append(float(row["a_lm_29_y"]))
            a_rheel_y.append(float(row["a_lm_30_y"]))
            b_lheel_y.append(float(row["b_lm_29_y"]))
            b_rheel_y.append(float(row["b_lm_30_y"]))

    if len(times) < 3:
        raise RuntimeError("Not enough valid synced rows to evaluate gait events")

    a_l_events = detect_heel_strikes(times, a_lheel_y, args.smooth_window, args.min_event_gap_ms, args.peak_threshold)
    a_r_events = detect_heel_strikes(times, a_rheel_y, args.smooth_window, args.min_event_gap_ms, args.peak_threshold)
    b_l_events = detect_heel_strikes(times, b_lheel_y, args.smooth_window, args.min_event_gap_ms, args.peak_threshold)
    b_r_events = detect_heel_strikes(times, b_rheel_y, args.smooth_window, args.min_event_gap_ms, args.peak_threshold)

    left_deltas = match_events(a_l_events, b_l_events, args.max_match_ms)
    right_deltas = match_events(a_r_events, b_r_events, args.max_match_ms)

    all_deltas = left_deltas + right_deltas
    event_offset_ms = median(all_deltas)
    if event_offset_ms is None:
        event_offset_ms = 0.0

    # Apply global event-based offset to B-A deltas for aligned evaluation.
    left_deltas_aligned = [d - event_offset_ms for d in left_deltas]
    right_deltas_aligned = [d - event_offset_ms for d in right_deltas]
    all_deltas_aligned = [d - event_offset_ms for d in all_deltas]

    result = {
        "sync_csv": str(sync_csv),
        "params": {
            "max_match_ms": args.max_match_ms,
            "smooth_window": args.smooth_window,
            "min_event_gap_ms": args.min_event_gap_ms,
            "peak_threshold": args.peak_threshold,
        },
        "counts": {
            "valid_rows": len(times),
            "a_left_events": len(a_l_events),
            "a_right_events": len(a_r_events),
            "b_left_events": len(b_l_events),
            "b_right_events": len(b_r_events),
        },
        "event_alignment": {
            "estimated_global_offset_ms": event_offset_ms,
            "note": "B timeline is shifted by -offset for alignment.",
        },
        "left_heel_strike_delta_ms": summarize(left_deltas),
        "right_heel_strike_delta_ms": summarize(right_deltas),
        "overall_heel_strike_delta_ms": summarize(all_deltas),
        "left_heel_strike_delta_ms_aligned": summarize(left_deltas_aligned),
        "right_heel_strike_delta_ms_aligned": summarize(right_deltas_aligned),
        "overall_heel_strike_delta_ms_aligned": summarize(all_deltas_aligned),
    }

    if args.write_aligned_csv:
        out_csv = Path(args.write_aligned_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = []
        if raw_rows:
            fieldnames = list(raw_rows[0].keys())
        if "b_abs_utc_aligned" not in fieldnames:
            fieldnames.append("b_abs_utc_aligned")
        if "delta_ms_b_minus_a_aligned" not in fieldnames:
            fieldnames.append("delta_ms_b_minus_a_aligned")

        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in raw_rows:
                out_row = dict(row)
                delta_raw = out_row.get("delta_ms_b_minus_a", "")
                if delta_raw != "":
                    try:
                        out_row["delta_ms_b_minus_a_aligned"] = float(delta_raw) - event_offset_ms
                    except ValueError:
                        out_row["delta_ms_b_minus_a_aligned"] = ""
                else:
                    out_row["delta_ms_b_minus_a_aligned"] = ""

                b_abs = out_row.get("b_abs_utc", "")
                if b_abs:
                    try:
                        out_row["b_abs_utc_aligned"] = shift_iso_time(b_abs, event_offset_ms)
                    except ValueError:
                        out_row["b_abs_utc_aligned"] = ""
                else:
                    out_row["b_abs_utc_aligned"] = ""

                writer.writerow(out_row)

        result["aligned_csv"] = str(out_csv)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("Gait event consistency computed.")
    print(f"valid_rows={len(times)}")
    print(f"left_events A/B={len(a_l_events)}/{len(b_l_events)} matched={len(left_deltas)}")
    print(f"right_events A/B={len(a_r_events)}/{len(b_r_events)} matched={len(right_deltas)}")
    print(f"estimated_event_offset_ms={event_offset_ms}")
    overall = result["overall_heel_strike_delta_ms"]
    overall_aligned = result["overall_heel_strike_delta_ms_aligned"]
    print(f"overall_matched={overall['matched_events']}")
    print(f"overall_mean_abs_ms(raw)={overall['mean_abs_ms']}")
    print(f"overall_p95_abs_ms(raw)={overall['p95_abs_ms']}")
    print(f"overall_mean_abs_ms(aligned)={overall_aligned['mean_abs_ms']}")
    print(f"overall_p95_abs_ms(aligned)={overall_aligned['p95_abs_ms']}")
    print(f"output_json={output_json}")


if __name__ == "__main__":
    main()
