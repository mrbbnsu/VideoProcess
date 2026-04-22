import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

LOWER_BODY_IDS = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a side-by-side visualization for synced frame pairs and landmarks."
    )
    parser.add_argument("--video-a", required=True, help="Path to video A")
    parser.add_argument("--video-b", required=True, help="Path to video B")
    parser.add_argument("--sync-csv", required=True, help="Sync pairs CSV from test_t1_sync.py")
    parser.add_argument(
        "--output-image",
        default="output/sync/sync_pairs_preview.jpg",
        help="Output combined image path",
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=6,
        help="How many synced pairs to draw",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting row index in sync CSV",
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=320,
        help="Rendered height for each frame tile",
    )
    return parser.parse_args()


def read_sync_rows(csv_path: Path, start_index: int, num_pairs: int) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if idx < start_index:
                continue
            rows.append(row)
            if len(rows) >= num_pairs:
                break
    return rows


def get_frame(video_path: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")
    return frame


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

    # Draw simple lower-body chains for readability.
    chains = [
        (0, 2),
        (2, 4),
        (4, 6),
        (6, 8),
        (1, 3),
        (3, 5),
        (5, 7),
        (7, 9),
        (0, 1),
    ]
    for a, b in chains:
        xa, ya = points[a]
        xb, yb = points[b]
        if xa >= 0 and xb >= 0:
            cv2.line(out, (xa, ya), (xb, yb), (0, 200, 0), 2)

    return out


def resize_keep_aspect(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    target_w = int((target_h / h) * w)
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)


def put_label(img: np.ndarray, text: str, x: int, y: int) -> None:
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def main() -> None:
    args = parse_args()

    video_a = Path(args.video_a)
    video_b = Path(args.video_b)
    sync_csv = Path(args.sync_csv)
    out_path = Path(args.output_image)

    rows = read_sync_rows(sync_csv, args.start_index, args.num_pairs)
    if not rows:
        raise RuntimeError("No rows found in sync CSV for the requested range")

    margin = 16
    gap = 12
    title_h = 42

    rendered_rows = []
    max_row_w = 0

    for i, row in enumerate(rows):
        a_idx = int(row["a_frame_idx"])
        b_idx = int(row["b_frame_idx"])
        delta_ms = float(row["delta_ms_b_minus_a"])

        frame_a = get_frame(video_a, a_idx)
        frame_b = get_frame(video_b, b_idx)

        frame_a = draw_landmarks(frame_a, row, "a")
        frame_b = draw_landmarks(frame_b, row, "b")

        frame_a = resize_keep_aspect(frame_a, args.target_height)
        frame_b = resize_keep_aspect(frame_b, args.target_height)

        row_h = args.target_height + 36
        row_w = frame_a.shape[1] + gap + frame_b.shape[1]
        row_canvas = np.zeros((row_h, row_w, 3), dtype=np.uint8)

        row_canvas[36 : 36 + frame_a.shape[0], 0 : frame_a.shape[1]] = frame_a
        x2 = frame_a.shape[1] + gap
        row_canvas[36 : 36 + frame_b.shape[0], x2 : x2 + frame_b.shape[1]] = frame_b

        put_label(row_canvas, f"Pair {args.start_index + i} | delta={delta_ms:.3f} ms", 2, 22)
        put_label(row_canvas, f"A frame={a_idx}", 6, 54)
        put_label(row_canvas, f"B frame={b_idx}", x2 + 6, 54)

        rendered_rows.append(row_canvas)
        max_row_w = max(max_row_w, row_w)

    total_h = title_h + margin + len(rendered_rows) * args.target_height + len(rendered_rows) * 36 + (len(rendered_rows) - 1) * margin + margin
    total_w = max_row_w + margin * 2
    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)

    put_label(canvas, "Synced Pairs Preview (A | B)", margin, 28)

    y = title_h + margin
    for row_img in rendered_rows:
        x = margin
        canvas[y : y + row_img.shape[0], x : x + row_img.shape[1]] = row_img
        y += row_img.shape[0] + margin

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), canvas)
    if not ok:
        raise RuntimeError(f"Failed to save image: {out_path}")

    print(f"Saved: {out_path}")
    print(f"Rows: {len(rendered_rows)}")


if __name__ == "__main__":
    main()
