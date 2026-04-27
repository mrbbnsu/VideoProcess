"""Batch process BBS pre/training videos."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.workflow.single_video_angles_events import run

base = PROJECT_ROOT / "data/BBS/HealthSubjectTest"

trials = ["T001", "T002", "T003"]
subjects = ["S001", "S002", "S003"]

for subj in subjects:
    for trial in trials:
        pre = base / f"{subj}_{trial}_pre.MOV"
        training = base / f"{subj}_{trial}_training.MOV"
        for video in [pre, training]:
            if not video.exists():
                print(f"[SKIP] {video.name} not found")
                continue
            stem = video.stem
            out_dir = PROJECT_ROOT / "output/BBSOutput" / f"{subj}" / stem
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n>>> Processing {video.name} ...")
            result = run(
                video=str(video),
                start_seconds=0.0,
                seconds=60.0,
                output_dir=str(out_dir),
                export_segment_video=True,
                single_person_lock=True,
                min_event_gap_ms=500.0,
                peak_threshold=0.005,
                smooth_window=9,
            )
            print(f"  Done: {result.get('angles_csv', '')}")
            for k, v in result.items():
                if k != 'angles_csv':
                    print(f"    {k}={v}")
        # break

print("\nAll done.")
