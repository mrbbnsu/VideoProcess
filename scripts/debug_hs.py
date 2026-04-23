"""Debug heel-strike detection matching."""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.evaluate_gait_event_consistency import detect_heel_strikes
import csv

csv_path = PROJECT_ROOT / "output/Test_MOV/Test_angles_0to6p5s.csv"
rows = list(csv.DictReader(open(csv_path)))
lheel_y = [(float(r['time_s']), float(r['left_heel_y'])) for r in rows if r['left_heel_y']]
rheel_y = [(float(r['time_s']), float(r['right_heel_y'])) for r in rows if r['right_heel_y']]

l_ev = detect_heel_strikes([t for t,_ in lheel_y], [y for _,y in lheel_y], 5, 800.0, 0.01)
r_ev = detect_heel_strikes([t for t,_ in rheel_y], [y for _,y in rheel_y], 5, 800.0, 0.01)
print(f"LEFT ({len(l_ev)}): {[f'{t:.3f}' for t in l_ev]}")
print(f"RIGHT ({len(r_ev)}): {[f'{t:.3f}' for t in r_ev]}")

used_r = set()
matches = []
for lt in l_ev:
    for ri, rt in enumerate(r_ev):
        if ri in used_r:
            continue
        if abs(lt - rt) <= 0.25:
            matches.append((lt+rt)/2.0)
            used_r.add(ri)
            print(f"  LEFT {lt:.3f} <-> RIGHT {rt:.3f} (diff={abs(lt-rt)*1000:.0f}ms)")
            break
print(f"\nMATCHED ({len(matches)}): {[f'{t:.3f}' for t in matches]}")

import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

l_times = [t for t, _ in lheel_y]
l_vals = [y for _, y in lheel_y]
r_times = [t for t, _ in rheel_y]
r_vals = [y for _, y in rheel_y]

# Top: raw trajectories
axes[0].plot(l_times, l_vals, label="Left heel y", linewidth=1.2, alpha=0.8)
axes[0].plot(r_times, r_vals, label="Right heel y", linewidth=1.2, alpha=0.8)
for t in l_ev:
    axes[0].axvline(t, color="tab:blue", linestyle="--", alpha=0.7, linewidth=1.2)
for t in r_ev:
    axes[0].axvline(t, color="tab:orange", linestyle="--", alpha=0.7, linewidth=1.2)
axes[0].set_ylabel("heel y (hip-norm)")
axes[0].set_title("Heel Y Trajectory + Detected Heel-Strikes (dashed = L blue / R orange)")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Middle: events overlay with colors per match status
axes[1].plot(l_times, l_vals, label="Left heel", linewidth=1.5, color="tab:blue")
axes[1].plot(r_times, r_vals, label="Right heel", linewidth=1.5, color="tab:orange")
matched_l = set()
matched_r = set()
for mi, m in enumerate(matches):
    axes[1].axvline(m, color="green", linestyle="-", alpha=0.9, linewidth=2,
                    label=f"Match {mi+1}" if mi == 0 else "")
for lt in l_ev:
    axes[1].axvline(lt, color="blue", linestyle=":", alpha=0.9)
for rt in r_ev:
    axes[1].axvline(rt, color="red", linestyle=":", alpha=0.9)
axes[1].set_ylabel("heel y (hip-norm)")
axes[1].set_title("L/R Events (blue/red dotted) + Matched Combined (green solid)")
axes[1].legend()
axes[1].grid(alpha=0.3)

# Bottom: difference
diff = []
common_t = sorted(set(l_times) & set(r_times))
l_map = dict(lheel_y)
r_map = dict(rheel_y)
for t in common_t:
    if t in l_map and t in r_map:
        diff.append((t, l_map[t] - r_map[t]))
if diff:
    d_times, d_vals = zip(*diff)
    axes[2].plot(d_times, d_vals, label="L - R", linewidth=1.2, color="purple")
axes[2].axhline(0, color="gray", linestyle="-", alpha=0.5)
axes[2].set_ylabel("L - R (hip-norm)")
axes[2].set_xlabel("time (s)")
axes[2].set_title("Left vs Right Heel Height Difference")
axes[2].legend()
axes[2].grid(alpha=0.3)

fig.tight_layout()
plt.show()
