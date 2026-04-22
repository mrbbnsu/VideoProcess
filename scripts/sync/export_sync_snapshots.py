import argparse
import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

LOWER_BODY_IDS = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downsample synced pairs to 30fps and export 5fps side-by-side snapshots."
    )
    parser.add_argument("--video-a", required=True)
    parser.add_argument("--video-b", required=True)
    parser.add_argument("--sync-csv", required=True)
    parser.add_argument("--output-dir", default="output/sync/snapshots_5fps")
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--snapshot-fps", type=float, default=5.0)
    parser.add_argument("--max-output", type=int, default=0, help="0 means no limit")
    parser.add_argument("--target-height", type=int, default=360, help="Output pair image height (without title area)")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality (1-100)")
    return parser.parse_args()


def parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def read_rows(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def downsample_by_time(rows: list[dict], fps: float) -> list[dict]:
    if not rows:
        return []

    min_interval = 1.0 / fps
    kept: list[dict] = []

    last_t = None
    for row in rows:
        t = parse_iso(row["a_abs_utc"]).timestamp()
        if last_t is None or (t - last_t) >= min_interval - 1e-9:
            kept.append(row)
            last_t = t

    return kept


def sample_for_snapshots(rows: list[dict], fps: float) -> list[dict]:
    if not rows:
        return []

    min_interval = 1.0 / fps
    sampled: list[dict] = []

    last_t = None
    for row in rows:
        t = parse_iso(row["a_abs_utc"]).timestamp()
        if last_t is None or (t - last_t) >= min_interval - 1e-9:
            sampled.append(row)
            last_t = t

    return sampled


def draw_landmarks(frame: np.ndarray, row: dict, prefix: str) -> np.ndarray:
    h, w = frame.shape[:2]
    out = frame.copy()

    points: list[tuple[int, int]] = []
    for lm_id in LOWER_BODY_IDS:
        x_key = f"{prefix}_lm_{lm_id}_x"
        y_key = f"{prefix}_lm_{lm_id}_y"
        x_val = row.get(x_key, "")
        y_val = row.get(y_key, "")
        if not x_val or not y_val:
            points.append((-1, -1))
            continue

        x = int(float(x_val) * w)
        y = int(float(y_val) * h)
        points.append((x, y))
        cv2.circle(out, (x, y), 4, (0, 255, 255), -1)

    chains = [(0, 2), (2, 4), (4, 6), (6, 8), (1, 3), (3, 5), (5, 7), (7, 9), (0, 1)]
    for a, b in chains:
        xa, ya = points[a]
        xb, yb = points[b]
        if xa >= 0 and xb >= 0:
            cv2.line(out, (xa, ya), (xb, yb), (0, 200, 0), 2)

    return out


def get_frame(cap: cv2.VideoCapture, frame_idx: int, tag: str) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {tag}")
    return frame


def make_pair_image(
    a_img: np.ndarray,
    b_img: np.ndarray,
    row: dict,
    pair_idx: int,
    target_h: int,
) -> np.ndarray:

    def resize_keep_h(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        tw = int((target_h / h) * w)
        return cv2.resize(img, (tw, target_h), interpolation=cv2.INTER_AREA)

    a_img = resize_keep_h(a_img)
    b_img = resize_keep_h(b_img)

    gap = 12
    title_h = 64
    canvas_h = title_h + target_h
    canvas_w = a_img.shape[1] + gap + b_img.shape[1]
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    canvas[title_h:, : a_img.shape[1]] = a_img
    x2 = a_img.shape[1] + gap
    canvas[title_h:, x2 : x2 + b_img.shape[1]] = b_img

    delta = float(row["delta_ms_b_minus_a"])
    text = f"pair={pair_idx}  delta={delta:.3f} ms  A={row['a_frame_idx']}  B={row['b_frame_idx']}"
    cv2.putText(canvas, text, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"A {row['a_abs_utc']}", (8, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"B {row['b_abs_utc']}", (x2 + 8, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1, cv2.LINE_AA)

    return canvas


def main() -> None:
    args = parse_args()

    video_a = Path(args.video_a)
    video_b = Path(args.video_b)
    sync_csv = Path(args.sync_csv)
    out_dir = Path(args.output_dir)

    rows = read_rows(sync_csv)
    rows_30 = downsample_by_time(rows, args.target_fps)
    rows_5 = sample_for_snapshots(rows_30, args.snapshot_fps)

    if args.max_output > 0:
        rows_5 = rows_5[: args.max_output]

    out_dir.mkdir(parents=True, exist_ok=True)

    cap_a = cv2.VideoCapture(str(video_a))
    cap_b = cv2.VideoCapture(str(video_b))
    if not cap_a.isOpened() or not cap_b.isOpened():
        raise RuntimeError("Cannot open one or both input videos")

    saved = 0
    for i, row in enumerate(rows_5):
        a_idx = int(row["a_frame_idx"])
        b_idx = int(row["b_frame_idx"])

        frame_a = get_frame(cap_a, a_idx, "video_a")
        frame_b = get_frame(cap_b, b_idx, "video_b")

        frame_a = draw_landmarks(frame_a, row, "a")
        frame_b = draw_landmarks(frame_b, row, "b")

        combined = make_pair_image(frame_a, frame_b, row, i, args.target_height)
        out_path = out_dir / f"sync_{i:04d}.jpg"
        quality = max(1, min(100, int(args.jpeg_quality)))
        ok = cv2.imwrite(str(out_path), combined, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError(f"Failed to save {out_path}")
        saved += 1

    cap_a.release()
    cap_b.release()

    print("Done")
    print(f"source_rows={len(rows)}")
    print(f"rows_after_{args.target_fps:.1f}fps_downsample={len(rows_30)}")
    print(f"rows_after_{args.snapshot_fps:.1f}fps_sampling={len(rows_5)}")
    print(f"saved_images={saved}")
    print(f"target_height={args.target_height}")
    print(f"jpeg_quality={max(1, min(100, int(args.jpeg_quality)))}")
    print(f"output_dir={out_dir}")


if __name__ == "__main__":
    main()
