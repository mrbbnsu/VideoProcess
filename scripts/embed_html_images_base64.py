import argparse
import base64
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed local image assets in an HTML file as base64 data URIs."
    )
    parser.add_argument("--input-html", required=True)
    parser.add_argument("--output-html", required=True)
    return parser.parse_args()


def mime_from_suffix(suffix: str) -> str:
    s = suffix.lower()
    if s == ".png":
        return "image/png"
    if s in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if s == ".gif":
        return "image/gif"
    if s == ".webp":
        return "image/webp"
    if s == ".svg":
        return "image/svg+xml"
    return "application/octet-stream"


def embed_images(html_text: str, base_dir: Path) -> tuple[str, int]:
    pattern = re.compile(r'src="([^"]+)"')
    replaced = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal replaced
        src = match.group(1)

        if src.startswith("data:") or src.startswith("http://") or src.startswith("https://"):
            return match.group(0)

        img_path = (base_dir / src).resolve()
        if not img_path.exists() or not img_path.is_file():
            return match.group(0)

        data = img_path.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        mime = mime_from_suffix(img_path.suffix)
        replaced += 1
        return f'src="data:{mime};base64,{b64}"'

    out = pattern.sub(repl, html_text)
    return out, replaced


def main() -> None:
    args = parse_args()
    input_html = Path(args.input_html).resolve()
    output_html = Path(args.output_html).resolve()

    if not input_html.exists():
        raise FileNotFoundError(input_html)

    html = input_html.read_text(encoding="utf-8")
    embedded_html, n = embed_images(html, input_html.parent)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(embedded_html, encoding="utf-8")

    print("Done")
    print(f"input_html={input_html}")
    print(f"output_html={output_html}")
    print(f"embedded_images={n}")
    print(f"output_size_bytes={output_html.stat().st_size}")


if __name__ == "__main__":
    main()
