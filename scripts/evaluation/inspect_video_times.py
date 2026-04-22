from pathlib import Path
from datetime import datetime, timezone

from export_video_timestamps import read_mp4_mvhd_creation_time_utc


def fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def inspect(path: Path) -> None:
    print(f"=== {path} ===")
    if not path.exists():
        print("missing")
        return

    st = path.stat()
    print(f"size_bytes={st.st_size}")
    print(f"fs_ctime_utc={fmt_ts(st.st_ctime)}")
    print(f"fs_mtime_utc={fmt_ts(st.st_mtime)}")

    mvhd = read_mp4_mvhd_creation_time_utc(path)
    print(f"mvhd_creation_utc={mvhd.isoformat() if mvhd else 'None'}")


if __name__ == "__main__":
    inspect(Path("data/raw/Video/V1-1.MOV"))
    inspect(Path("data/raw/Video/V1-2.MOV"))
