import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from evaluate_gait_event_consistency import detect_heel_strikes, median


@dataclass
class MatchedEvent:
    side: str
    a_time: float
    b_time: float
    delta_ms_raw: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print per-event timing/frame gaps before and after global event alignment."
    )
    parser.add_argument("--sync-csv", required=True)
    parser.add_argument("--max-match-ms", type=float, default=120.0)
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--min-event-gap-ms", type=float, default=120.0)
    parser.add_argument("--peak-threshold", type=float, default=0.0003)
    return parser.parse_args()


def parse_iso(ts: str) -> float:
    return datetime.fromisoformat(ts).timestamp()


def infer_fps(rows: list[dict], prefix: str) -> float:
    frame_key = f"{prefix}_frame_idx"
    time_key = f"{prefix}_abs_utc"

    samples = []
    prev_f = None
    prev_t = None
    for r in rows:
        if not r.get(frame_key) or not r.get(time_key):
            continue
        f = int(r[frame_key])
        t = parse_iso(r[time_key])
        if prev_f is not None and prev_t is not None:
            df = f - prev_f
            dt = t - prev_t
            if df > 0 and dt > 0:
                samples.append(df / dt)
        prev_f = f
        prev_t = t

    if not samples:
        return 0.0

    samples.sort()
    mid = len(samples) // 2
    if len(samples) % 2 == 1:
        return samples[mid]
    return (samples[mid - 1] + samples[mid]) / 2.0


def match_with_pairs(a_times: list[float], b_times: list[float], side: str, max_match_ms: float) -> list[MatchedEvent]:
    if not a_times or not b_times:
        return []

    max_sec = max_match_ms / 1000.0
    out: list[MatchedEvent] = []

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
            tb = b_times[best_idx]
            out.append(MatchedEvent(side=side, a_time=ta, b_time=tb, delta_ms_raw=(tb - ta) * 1000.0))

    return out


def main() -> None:
    args = parse_args()

    sync_csv = Path(args.sync_csv)
    if not sync_csv.exists():
        raise FileNotFoundError(sync_csv)

    raw_rows = []
    times = []
    a_lheel_y = []
    a_rheel_y = []
    b_lheel_y = []
    b_rheel_y = []

    with sync_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_rows.append(row)
            if row.get("a_detected") != "True" or row.get("b_detected") != "True":
                continue
            need_keys = ["a_abs_utc", "a_lm_29_y", "a_lm_30_y", "b_lm_29_y", "b_lm_30_y"]
            if any(not row.get(k) for k in need_keys):
                continue

            times.append(parse_iso(row["a_abs_utc"]))
            a_lheel_y.append(float(row["a_lm_29_y"]))
            a_rheel_y.append(float(row["a_lm_30_y"]))
            b_lheel_y.append(float(row["b_lm_29_y"]))
            b_rheel_y.append(float(row["b_lm_30_y"]))

    if len(times) < 3:
        raise RuntimeError("Not enough valid rows")

    fps_a = infer_fps(raw_rows, "a")
    fps_b = infer_fps(raw_rows, "b")

    a_l_events = detect_heel_strikes(times, a_lheel_y, args.smooth_window, args.min_event_gap_ms, args.peak_threshold)
    a_r_events = detect_heel_strikes(times, a_rheel_y, args.smooth_window, args.min_event_gap_ms, args.peak_threshold)
    b_l_events = detect_heel_strikes(times, b_lheel_y, args.smooth_window, args.min_event_gap_ms, args.peak_threshold)
    b_r_events = detect_heel_strikes(times, b_rheel_y, args.smooth_window, args.min_event_gap_ms, args.peak_threshold)

    pairs = []
    pairs.extend(match_with_pairs(a_l_events, b_l_events, side="L", max_match_ms=args.max_match_ms))
    pairs.extend(match_with_pairs(a_r_events, b_r_events, side="R", max_match_ms=args.max_match_ms))
    pairs.sort(key=lambda p: p.a_time)

    offset_ms = median([p.delta_ms_raw for p in pairs])
    if offset_ms is None:
        offset_ms = 0.0

    print(f"sync_csv={sync_csv}")
    print(f"fps_a={fps_a:.6f}, fps_b={fps_b:.6f}")
    print(f"matched_events={len(pairs)}")
    print(f"estimated_global_event_offset_ms={offset_ms:.6f}")
    print("-")
    print("idx side | delta_raw_ms | delta_raw_frame@A delta_raw_frame@B | delta_aligned_ms | delta_aligned_frame@A delta_aligned_frame@B")

    for i, p in enumerate(pairs):
        raw_ms = p.delta_ms_raw
        ali_ms = raw_ms - offset_ms

        raw_fa = raw_ms / 1000.0 * fps_a if fps_a > 0 else 0.0
        raw_fb = raw_ms / 1000.0 * fps_b if fps_b > 0 else 0.0
        ali_fa = ali_ms / 1000.0 * fps_a if fps_a > 0 else 0.0
        ali_fb = ali_ms / 1000.0 * fps_b if fps_b > 0 else 0.0

        print(
            f"{i:02d}  {p.side}   | {raw_ms:9.3f} | {raw_fa:8.3f} {raw_fb:8.3f} |"
            f" {ali_ms:9.3f} | {ali_fa:8.3f} {ali_fb:8.3f}"
        )


if __name__ == "__main__":
    main()
