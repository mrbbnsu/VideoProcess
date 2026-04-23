"""Debug heel y trajectory and heel-strike detection for Test.MOV."""
import csv
import matplotlib.pyplot as plt
from pathlib import Path

csv_path = Path(__file__).parent.parent / "output/Test_MOV/Test_angles_0to6p5s.csv"
events_path = Path(__file__).parent.parent / "output/Test_MOV/Test_events_0to6p5s.csv"

rows = []
with csv_path.open() as f:
    for row in csv.DictReader(f):
        rows.append(row)

left_ev, right_ev = [], []
with events_path.open() as f:
    for row in csv.reader(f):
        if len(row) < 2:
            continue
        side, t = row[0].strip(), row[1].strip()
        try:
            if side == "left":
                left_ev.append(float(t))
            elif side == "right":
                right_ev.append(float(t))
        except ValueError:
            continue

times = [float(r["time_s"]) for r in rows]
lheel = [float(r["left_heel_y"]) if r["left_heel_y"] else float("nan") for r in rows]
rheel = [float(r["right_heel_y"]) if r["right_heel_y"] else float("nan") for r in rows]

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

# Top: raw heel y trajectories
axes[0].plot(times, lheel, label="Left heel y (hip-norm)", linewidth=1.2, alpha=0.8)
axes[0].plot(times, rheel, label="Right heel y (hip-norm)", linewidth=1.2, alpha=0.8)
for t in left_ev:
    axes[0].axvline(t, color="tab:blue", linestyle="--", alpha=0.8, linewidth=1.2)
for t in right_ev:
    axes[0].axvline(t, color="tab:orange", linestyle="--", alpha=0.8, linewidth=1.2)
axes[0].set_ylabel("heel y (hip-norm)")
axes[0].set_title("Heel Y Trajectory + Detected Heel-Strike Moments (vertical lines)")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Middle: left and right overlaid with HS marked
axes[1].plot(times, lheel, label="Left heel", linewidth=1.5, color="tab:blue")
axes[1].plot(times, rheel, label="Right heel", linewidth=1.5, color="tab:orange")
for i, t in enumerate(left_ev):
    axes[1].axvline(t, color="blue", linestyle=":", alpha=0.9, label="Left HS" if i == 0 else "")
for i, t in enumerate(right_ev):
    axes[1].axvline(t, color="red", linestyle=":", alpha=0.9, label="Right HS" if i == 0 else "")
axes[1].set_ylabel("heel y (hip-norm)")
axes[1].set_title("Heel Y + HS Markers (bottom of each peak = heel-strike)")
axes[1].legend()
axes[1].grid(alpha=0.3)

# Bottom: L-R difference
diff = [l - r for l, r in zip(lheel, rheel)]
axes[2].plot(times, diff, label="L - R heel y", linewidth=1.2, color="purple")
axes[2].axhline(0, color="gray", linestyle="-", alpha=0.5)
axes[2].set_ylabel("L - R (hip-norm)")
axes[2].set_xlabel("time (s)")
axes[2].set_title("Left vs Right Heel Height Difference")
axes[2].legend()
axes[2].grid(alpha=0.3)

fig.tight_layout()
out_png = csv_path.parent / "heel_y_debug.png"
fig.savefig(out_png, dpi=150)
plt.close(fig)
print(f"Saved to {out_png}")
print(f"Left HS ({len(left_ev)}): {[f'{t:.3f}' for t in left_ev]}")
print(f"Right HS ({len(right_ev)}): {[f'{t:.3f}' for t in right_ev]}")
