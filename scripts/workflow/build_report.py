import argparse
import csv
import html
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full IC2-style pipeline and generate report bundle (md/html/embedded html)."
    )
    parser.add_argument("--video-a", required=True)
    parser.add_argument("--video-b", required=True)
    parser.add_argument("--pair-name", default="IC2-1_IC2-2")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument(
        "--compensate-start-offset",
        action="store_true",
        help="Enable start-time offset compensation when running test_t1_sync.py",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable to use")
    return parser.parse_args()


def run_cmd(cmd: list[str]) -> None:
    print("RUN:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def try_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_mae(rows: list[dict], a_col: str, b_col: str) -> float | None:
    diffs = []
    for row in rows:
        av = row.get(a_col, "")
        bv = row.get(b_col, "")
        if not av or not bv:
            continue
        af = try_float(av)
        bf = try_float(bv)
        if af is None or bf is None:
            continue
        diffs.append(abs(af - bf))
    if not diffs:
        return None
    return sum(diffs) / len(diffs)


def fmt_num(v: float | None, digits: int = 3) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}"


def file_size(path: Path) -> int:
    if not path.exists():
        return 0
    return path.stat().st_size


def rel_to_reports(path: Path, reports_dir: Path) -> str:
    return Path(os.path.relpath(path, start=reports_dir)).as_posix()


def write_result_bundle_md(
    out_path: Path,
    pair_name: str,
    sync_csv: Path,
    snapshots_dir: Path,
    angle_csv: Path,
    cycle_csv: Path,
    plot_a: Path,
    plot_b: Path,
    plot_ab: Path,
) -> None:
    lines = []
    lines.append(f"# {pair_name} Result Bundle")
    lines.append("")
    lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("Project: VideoPoseRecognization")
    lines.append("")
    lines.append("## 1) Sync")
    lines.append("")
    lines.append(f"- Sync CSV: {sync_csv.as_posix()}")
    lines.append(f"  - size: {file_size(sync_csv)} bytes")
    lines.append("")
    lines.append("## 2) Snapshots")
    lines.append("")
    lines.append(f"- Snapshot dir: {snapshots_dir.as_posix()}")
    if snapshots_dir.exists():
        n = len(list(snapshots_dir.glob("*.jpg")))
        lines.append(f"  - jpg count: {n}")
    lines.append("")
    lines.append("## 3) Angles and Cycle Stats")
    lines.append("")
    lines.append(f"- Angles CSV: {angle_csv.as_posix()} ({file_size(angle_csv)} bytes)")
    lines.append(f"- Cycle stats CSV: {cycle_csv.as_posix()} ({file_size(cycle_csv)} bytes)")
    lines.append("")
    lines.append("## 4) Plot Images")
    lines.append("")
    lines.append(f"- A left/right: {plot_a.as_posix()} ({file_size(plot_a)} bytes)")
    lines.append(f"- B left/right: {plot_b.as_posix()} ({file_size(plot_b)} bytes)")
    lines.append(f"- AB same-side: {plot_ab.as_posix()} ({file_size(plot_ab)} bytes)")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_visual_report_md(
    out_path: Path,
    pair_name: str,
    sync_rows: list[dict],
    angle_rows: list[dict],
    cycle_rows: list[dict],
    reports_dir: Path,
    plot_a: Path,
    plot_b: Path,
    plot_ab: Path,
    snapshots_dir: Path,
    sync_csv: Path,
    angle_csv: Path,
    cycle_csv: Path,
) -> None:
    avg_abs_delta = None
    if sync_rows:
        vals = []
        for r in sync_rows:
            x = r.get("delta_ms_b_minus_a", "")
            if x:
                xf = try_float(x)
                if xf is not None:
                    vals.append(abs(xf))
        if vals:
            avg_abs_delta = sum(vals) / len(vals)

    mae_l_hip = compute_mae(angle_rows, "a_left_hip_deg", "b_left_hip_deg")
    mae_r_hip = compute_mae(angle_rows, "a_right_hip_deg", "b_right_hip_deg")
    mae_l_knee = compute_mae(angle_rows, "a_left_knee_deg", "b_left_knee_deg")
    mae_r_knee = compute_mae(angle_rows, "a_right_knee_deg", "b_right_knee_deg")
    mae_l_ankle = compute_mae(angle_rows, "a_left_ankle_deg", "b_left_ankle_deg")
    mae_r_ankle = compute_mae(angle_rows, "a_right_ankle_deg", "b_right_ankle_deg")

    snap_samples = []
    if snapshots_dir.exists():
        snap_samples = sorted(snapshots_dir.glob("sync_*.jpg"))[:6]

    lines = []
    lines.append(f"# {pair_name} Visual Report")
    lines.append("")
    lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Sync pairs: {len(sync_rows)}")
    lines.append(f"- Avg absolute delta (ms): {fmt_num(avg_abs_delta)}")
    lines.append(f"- Angle rows: {len(angle_rows)}")
    lines.append(f"- Cycles written: {len(cycle_rows)}")
    lines.append("")
    lines.append("## Consistency Metrics (A/B same-side MAE, deg)")
    lines.append("")
    lines.append("| Metric | MAE |")
    lines.append("|---|---:|")
    lines.append(f"| Left Hip | {fmt_num(mae_l_hip)} |")
    lines.append(f"| Right Hip | {fmt_num(mae_r_hip)} |")
    lines.append(f"| Left Knee | {fmt_num(mae_l_knee)} |")
    lines.append(f"| Right Knee | {fmt_num(mae_r_knee)} |")
    lines.append(f"| Left Ankle | {fmt_num(mae_l_ankle)} |")
    lines.append(f"| Right Ankle | {fmt_num(mae_r_ankle)} |")
    lines.append("")
    lines.append("## Angle Plots")
    lines.append("")
    lines.append(f"![AB same side]({rel_to_reports(plot_ab, reports_dir)})")
    lines.append("")
    lines.append(f"![A left right]({rel_to_reports(plot_a, reports_dir)})")
    lines.append("")
    lines.append(f"![B left right]({rel_to_reports(plot_b, reports_dir)})")
    lines.append("")
    lines.append("## Snapshot Samples")
    lines.append("")
    if snap_samples:
        for s in snap_samples:
            lines.append(f"### {s.stem}")
            lines.append("")
            lines.append(f"![{s.stem}]({rel_to_reports(s, reports_dir)})")
            lines.append("")
    else:
        lines.append("No snapshots found.")
        lines.append("")

    lines.append("## Source Files")
    lines.append("")
    lines.append(f"- {rel_to_reports(sync_csv, reports_dir)}")
    lines.append(f"- {rel_to_reports(angle_csv, reports_dir)}")
    lines.append(f"- {rel_to_reports(cycle_csv, reports_dir)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def md_to_html(md_text: str, title: str) -> str:
    escaped_title = html.escape(title, quote=True)
    html_lines = [
        "<!doctype html>",
        "<html lang=\"en\">",
        "<head>",
        "  <meta charset=\"utf-8\" />",
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />",
        f"  <title>{escaped_title}</title>",
        "  <style>",
        "    body { font-family: Segoe UI, Arial, sans-serif; max-width: 1080px; margin: 0 auto; padding: 20px; line-height: 1.5; }",
        "    img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px; }",
        "    table { border-collapse: collapse; width: 100%; margin: 12px 0; }",
        "    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "    th { background: #f3f6fa; }",
        "    code { background: #f3f6fa; padding: 2px 6px; border-radius: 4px; }",
        "  </style>",
        "</head>",
        "<body>",
    ]

    for line in md_text.splitlines():
        if line.startswith("# "):
            html_lines.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            html_lines.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("![") and "](" in line and line.endswith(")"):
            alt = line[2: line.index("](")]
            src = line[line.index("](") + 2 : -1]
            safe_src = html.escape(src, quote=True)
            safe_alt = html.escape(alt, quote=True)
            html_lines.append(f"<img src=\"{safe_src}\" alt=\"{safe_alt}\" />")
        elif line.startswith("|") and line.endswith("|"):
            # Very small markdown-table support: rows are converted to <tr> cells.
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(c.replace("-", "").replace(":", "").strip() == "" for c in cells):
                continue
            if "<table>" not in html_lines:
                html_lines.append("<table>")
            tag = "th" if "| Metric | MAE |" in line else "td"
            row = "".join([f"<{tag}>{html.escape(c)}</{tag}>" for c in cells])
            html_lines.append(f"<tr>{row}</tr>")
        elif line.startswith("- "):
            html_lines.append(f"<li>{html.escape(line[2:])}</li>")
        elif line.strip() == "":
            html_lines.append("<p></p>")
        else:
            html_lines.append(f"<p>{html.escape(line)}</p>")

    if "<table>" in html_lines:
        html_lines.append("</table>")
    html_lines.append("</body>")
    html_lines.append("</html>")
    return "\n".join(html_lines)


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent
    scripts_dir = project_root / "scripts"

    pair_name = args.pair_name
    sync_csv = project_root / "output" / "sync" / f"{pair_name}_sync.csv"
    snapshots_dir = project_root / "output" / "sync" / f"{pair_name}_continuous_30fps"
    angle_dir = project_root / "output" / "angles" / "IC2_rematched"
    reports_dir = project_root / "output" / "reports"

    angle_csv = angle_dir / f"{pair_name}_sync_angles.csv"
    cycle_csv = angle_dir / f"{pair_name}_sync_cycle_stats.csv"
    plot_a = angle_dir / f"{pair_name}_sync_A_left_right_curves.png"
    plot_b = angle_dir / f"{pair_name}_sync_B_left_right_curves.png"
    plot_ab = angle_dir / f"{pair_name}_sync_AB_same_side_curves.png"

    visual_md = reports_dir / f"{pair_name}_visual_report.md"
    visual_html = reports_dir / f"{pair_name}_visual_report.html"
    visual_embedded_html = reports_dir / f"{pair_name}_visual_report_embedded.html"
    result_bundle_md = reports_dir / f"{pair_name}_result_bundle.md"

    sync_cmd = [
        args.python,
        str(scripts_dir / "test_t1_sync.py"),
        "--video-a",
        args.video_a,
        "--video-b",
        args.video_b,
        "--output-csv",
        str(sync_csv),
        "--max-frames",
        str(args.max_frames),
    ]
    if args.compensate_start_offset:
        sync_cmd.append("--compensate-start-offset")

    run_cmd(sync_cmd)

    run_cmd(
        [
            args.python,
            str(scripts_dir / "export_sync_snapshots.py"),
            "--video-a",
            args.video_a,
            "--video-b",
            args.video_b,
            "--sync-csv",
            str(sync_csv),
            "--output-dir",
            str(snapshots_dir),
            "--target-fps",
            "30",
            "--snapshot-fps",
            "5",
            "--max-output",
            "0",
        ]
    )

    run_cmd(
        [
            args.python,
            str(scripts_dir / "compute_joint_angles_and_plots.py"),
            "--sync-csv",
            str(sync_csv),
            "--output-dir",
            str(angle_dir),
        ]
    )

    run_cmd(
        [
            args.python,
            str(scripts_dir / "plot_ab_same_side_comparison.py"),
            "--angle-csv",
            str(angle_csv),
            "--output-image",
            str(plot_ab),
        ]
    )

    sync_rows = read_csv_rows(sync_csv)
    angle_rows = read_csv_rows(angle_csv)
    cycle_rows = read_csv_rows(cycle_csv)

    write_result_bundle_md(
        result_bundle_md,
        pair_name,
        sync_csv,
        snapshots_dir,
        angle_csv,
        cycle_csv,
        plot_a,
        plot_b,
        plot_ab,
    )

    write_visual_report_md(
        visual_md,
        pair_name,
        sync_rows,
        angle_rows,
        cycle_rows,
        reports_dir,
        plot_a,
        plot_b,
        plot_ab,
        snapshots_dir,
        sync_csv,
        angle_csv,
        cycle_csv,
    )

    md_text = visual_md.read_text(encoding="utf-8")
    html_text = md_to_html(md_text, f"{pair_name} Visual Report")
    visual_html.parent.mkdir(parents=True, exist_ok=True)
    visual_html.write_text(html_text, encoding="utf-8")

    run_cmd(
        [
            args.python,
            str(scripts_dir / "embed_html_images_base64.py"),
            "--input-html",
            str(visual_html),
            "--output-html",
            str(visual_embedded_html),
        ]
    )

    print("Done")
    print(f"result_bundle_md={result_bundle_md}")
    print(f"visual_report_md={visual_md}")
    print(f"visual_report_html={visual_html}")
    print(f"visual_report_embedded_html={visual_embedded_html}")


if __name__ == "__main__":
    main()
