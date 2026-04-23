"""处理 Test.MOV 侧面视角步态视频。"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.workflow.single_video_angles_events import run

result = run(
    video=str(PROJECT_ROOT / "data/Rec/Test.MOV"),
    start_seconds=0.0,
    seconds=6.5,
    output_dir=str(PROJECT_ROOT / "output/Test_MOV"),
    export_segment_video=True,
    single_person_lock=True,
)

print("Done:")
for k, v in result.items():
    print(f"  {k}={v}")
