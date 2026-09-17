"""fig_speed_accuracy.png: cost against trials per evaluation, and cost against accuracy.
Reads curve_{py,r}.csv (SWEEP=1 runs of bench) and bench_{py,r}.csv (the parameter sets)."""
import csv
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

M = {   # method as bench names it -> legend label, colour, marker, linestyle
    "ours 1e-12 (density + 7 grads)": ("ours, density + 7 gradients", "#1b9e77", "D", "-"),
    "ours 1e-8":                      ("ours, tol 1e-8",              "#66c2a5", "o", ":"),
    "hddm-wfpt err=1e-8":             ("hddm-wfpt (HSSM blackbox)",   "#000000", "s", "-"),
    "rtdists precision=3":            ("rtdists, precision 3",        "#d95f02", "^", "-"),
    "rtdists precision=5":            ("rtdists, precision 5",        "#d95f02", "X", ":"),
    "rtdists precision=8":            ("rtdists, precision 8",        "#a63603", "P", ":"),
    "WienR 5e-3 (EMC2 default)":      ("WienR, 5e-3 (EMC2 default)",  "#e6ab02", "*", ":"),
    "WienR 1e-4":                     ("WienR, 1e-4",                 "#e6ab02", "h", "--"),
    "WienR 1e-8":                     ("WienR, 1e-8",                 "#e6ab02", "*", "-"),
    "WienR 1e-12":                    ("WienR, 1e-12",                "#e78ac3", "P", "-"),
    "WienR 1e-12 + 5 grads":          ("WienR, 1e-12 + 5 gradients",  "#e78ac3", "d", ":"),
}
OURS_ACC = 1.4e-11        # ours at 1e-12 vs the 30-digit mpmath reference (bench.py anchor)

def read(path):
    lines = [l for l in open(path) if l.strip() not in ("", '""')]   # appended files start blank
    return list(csv.DictReader(lines))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

for path, lang in (("curve_py.csv", "Python"), ("../R/curve_r.csv", "R")):
    curve = defaultdict(lambda: ([], []))
    for row in read(path):
        curve[row["method"]][0].append(int(float(row["n"])))
        curve[row["method"]][1].append(float(row["ms"]))
    for m, (n, ms) in curve.items():
        label, c, mk, ls = M[m]
        if m.startswith("ours") and lang == "R":
            label, mk, ls = "ours, same in R", "v", "--"
        ax1.plot(n, ms, ls, color=c, marker=mk, markersize=5, linewidth=1.6, label=label)

bench, nset = defaultdict(lambda: ([], [])), {}
for lang, path in (("Python", "bench_py.csv"), ("R", "../R/bench_r.csv")):
    rows = read(path)
    nset[lang] = max(int(float(r["set"])) for r in rows)
    for r in rows:
        bench[r["method"]][0].append(float(r["us_per_trial"]))
        bench[r["method"]][1].append(float(r["rel_diff"]))

shown = set(ax1.get_legend_handles_labels()[1])
for m, (us, rel) in bench.items():
    if m not in M:
        continue
    label, c, mk, _ = M[m]
    us, rel = np.array(us), np.array(rel)
    x, y = (OURS_ACC if m.startswith("ours 1e-12") else np.median(rel)), np.median(us)
    ax2.plot([x, max(x, rel.max())], [y, y], "-", color=c, linewidth=1.2, alpha=.5)
    ax2.plot([x, x], np.percentile(us, [25, 75]), "-", color=c, linewidth=1.2, alpha=.5)
    ax2.plot([x], [y], marker=mk, color=c, markersize=8, linestyle="none",
             label=None if label in shown else label)

n_lab = (f"{nset['Python']} parameter sets" if nset["Python"] == nset["R"]
         else f"{nset['Python']} Python / {nset['R']} R sets so far")
ax1.set(xscale="log", yscale="log")
ax1.set_xlabel("trials per evaluation", style="italic")
ax1.set_ylabel("wall time per evaluation (ms)", style="italic")
ax1.set_title("Cost of one full-DDM likelihood evaluation", style="italic", fontsize=11)
ax2.set(xscale="log", yscale="log")
ax2.set_xlabel("relative error: median, bar to worst case (cheap and exact = bottom left)", style="italic")
ax2.set_ylabel("microseconds per trial", style="italic")
ax2.set_title(f"Cost against accuracy, {n_lab} (Tran et al. priors)", style="italic", fontsize=11)
for ax in (ax1, ax2):
    ax.grid(True, which="major", color="0.85", linewidth=0.6)
    ax.grid(True, which="minor", color="0.93", linewidth=0.4)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color("0.4")

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
fig.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=4,
           frameon=False, prop={"style": "italic", "size": 9})
fig.tight_layout()
fig.savefig("fig_speed_accuracy.png", dpi=200, bbox_inches="tight")
print("wrote fig_speed_accuracy.png")
