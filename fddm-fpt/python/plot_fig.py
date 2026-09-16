"""fig_speed_accuracy.png: cost against trials per evaluation, and cost against accuracy.
Reads curve_{py,r}.csv (sweep) and bench_{py,r}.csv (30 published parameter sets)."""
import csv
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# label -> colour, marker, linestyle; curve name; bench name
M = [
    ("ours, density + 7 gradients",   "#1b9e77", "D", "-",  "ours, density + 7 gradients",     "ours 1e-12 (density + 7 grads)"),
    ("ours, same in R",               "#1b9e77", "v", "--", "ours, density + 7 gradients (R)", None),
    ("ours, tol 1e-8",                "#66c2a5", "o", ":",  None,                              "ours 1e-8"),
    ("hddm-wfpt (HSSM blackbox)",     "#000000", "s", "-",  "hddm-wfpt (HSSM blackbox)",       "hddm-wfpt err=1e-8"),
    ("rtdists, precision 3",          "#d95f02", "^", "-",  "rtdists, precision 3",            "rtdists precision=3"),
    ("rtdists, precision 5",          "#d95f02", "X", ":",  "rtdists, precision 5",            "rtdists precision=5"),
    ("rtdists, precision 8",          "#a63603", "P", ":",  None,                              "rtdists precision=8"),
    ("WienR, 5e-3 (EMC2 default)",    "#e6ab02", "*", ":",  "WienR, 5e-3 (EMC2 default)",      "WienR 5e-3 (EMC2 default)"),
    ("WienR, 1e-4",                   "#e6ab02", "h", "--", None,                              "WienR 1e-4"),
    ("WienR, 1e-8",                   "#e6ab02", "*", "-",  "WienR, 1e-8",                     "WienR 1e-8"),
    ("WienR, 1e-12",                  "#e78ac3", "P", "-",  "WienR, 1e-12",                    "WienR 1e-12"),
    ("WienR, 1e-12 + 5 gradients",    "#e78ac3", "d", ":",  "WienR, 1e-12 + 5 gradients",      "WienR 1e-12 + 5 grads"),
]
OURS_ACC = 5.6e-15        # ours at 1e-12 vs the 30-digit mpmath reference (bench.py anchor)

def read(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))

curve = defaultdict(lambda: ([], []))
for row in read("curve_py.csv") + read("../R/curve_r.csv"):
    curve[row["method"]][0].append(int(float(row["n"])))
    curve[row["method"]][1].append(float(row["ms"]))

bench = defaultdict(lambda: ([], []))
for row in read("bench_py.csv") + read("../R/bench_r.csv"):
    if "NA" in (row["us_per_trial"], row["rel_diff"]) or "" in (row["us_per_trial"], row["rel_diff"]):
        continue
    bench[row["method"]][0].append(float(row["us_per_trial"]))
    bench[row["method"]][1].append(float(row["rel_diff"]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

for label, c, mk, ls, cname, bname in M:
    if cname in curve:
        n, ms = curve[cname]
        ax1.plot(n, ms, ls, color=c, marker=mk, markersize=5, linewidth=1.6, label=label)
    if bname in bench:
        us, rel = (np.array(v) for v in bench[bname])
        x = OURS_ACC if bname.startswith("ours 1e-12") else np.median(rel)
        ax2.plot([x, max(x, rel.max())], [np.median(us)]*2, "-", color=c, linewidth=1.2, alpha=.5)
        ax2.plot([np.median(us)*0 + x], [np.median(us)], marker=mk, color=c, markersize=8,
                 linestyle="none", label=label if cname not in curve else None)
        ax2.plot([x, x], np.percentile(us, [25, 75]), "-", color=c, linewidth=1.2, alpha=.5)

ax1.set(xscale="log", yscale="log")
ax1.set_xlabel("trials per evaluation", style="italic")
ax1.set_ylabel("wall time per evaluation (ms)", style="italic")
ax1.set_title("Cost of one full-DDM likelihood evaluation", style="italic", fontsize=11)

ax2.set(xscale="log", yscale="log")
ax2.set_xlabel("relative error: median, bar to worst case (cheap and exact = bottom left)", style="italic")
ax2.set_ylabel("microseconds per trial", style="italic")
ax2.set_title("Cost against accuracy, 30 published parameter sets", style="italic", fontsize=11)

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
