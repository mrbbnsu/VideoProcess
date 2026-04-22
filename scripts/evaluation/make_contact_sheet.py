import argparse
import math
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a contact sheet from images in a folder.")
    parser.add_argument("--input-dir", required=True, help="Folder containing images")
    parser.add_argument("--output-image", required=True, help="Output contact sheet image")
    parser.add_argument("--cols", type=int, default=4, help="Number of columns")
    parser.add_argument("--tile-width", type=int, default=420, help="Tile width")
    parser.add_argument("--tile-height", type=int, default=260, help="Tile height")
    parser.add_argument("--padding", type=int, default=12, help="Padding between tiles")
    parser.add_argument("--label", action="store_true", help="Draw file name label")
    return parser.parse_args()


def list_images(input_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def fit_to_tile(img: np.ndarray, tile_w: int, tile_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(tile_w / w, tile_h / h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    y0 = (tile_h - nh) // 2
    x0 = (tile_w - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_image = Path(args.output_image)
    images = list_images(input_dir)
    if not images:
        raise RuntimeError(f"No images found in: {input_dir}")

    cols = max(1, args.cols)
    rows = math.ceil(len(images) / cols)

    pad = args.padding
    tile_w = args.tile_width
    tile_h = args.tile_height

    sheet_w = pad + cols * tile_w + (cols - 1) * pad + pad
    sheet_h = pad + rows * tile_h + (rows - 1) * pad + pad
    sheet = np.full((sheet_h, sheet_w, 3), 16, dtype=np.uint8)

    for i, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        tile = fit_to_tile(img, tile_w, tile_h)

        r = i // cols
        c = i % cols
        x = pad + c * (tile_w + pad)
        y = pad + r * (tile_h + pad)

        sheet[y : y + tile_h, x : x + tile_w] = tile

        if args.label:
            text = img_path.name
            cv2.putText(sheet, text, (x + 8, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

    output_image.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_image), sheet)
    if not ok:
        raise RuntimeError(f"Failed to save: {output_image}")

    print(f"Saved: {output_image}")
    print(f"Images: {len(images)}")
    print(f"Grid: {rows}x{cols}")


if __name__ == "__main__":
    main()
