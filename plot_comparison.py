import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# --- OpenEvolve data (from run1.log, in evaluation order) ---
oe_data = [
    (1,  59.813, "discard"),
    (2,  None,   "crash"),
    (3,  59.853, "discard"),
    (4,  59.525, "keep"),
    (5,  59.796, "discard"),
    (6,  59.615, "keep"),
    (7,  60.028, "discard"),
    (8,  59.864, "discard"),
    (9,  59.539, "keep"),
    (10, 59.504, "discard"),
    (11, 60.104, "discard"),
    (12, 59.110, "keep"),
    (13, 59.670, "keep"),
    (14, 59.138, "keep"),
    (15, 59.428, "keep"),
    (16, 59.772, "discard"),
    (17, 60.147, "discard"),
    (18, 60.107, "discard"),
    (19, 59.687, "discard"),
    (20, 59.515, "keep"),
    (21, 59.386, "keep"),
    (22, 59.686, "keep"),
    (23, 60.639, "discard"),
    (24, 60.039, "discard"),
    (25, 59.759, "discard"),
]

# --- Advisor data (from results.tsv) ---
adv_data = [
    (0,  66.64,  "keep"),
    (1,  35.82,  "keep"),
    (2,  39.27,  "discard"),
    (3,  177.13, "discard"),
    (4,  34.53,  "keep"),
    (5,  None,   "crash"),
    (6,  36.58,  "discard"),
    (7,  34.95,  "discard"),
    (8,  None,   "crash"),
    (9,  35.16,  "discard"),
    (10, 33.80,  "keep"),
    (11, 34.04,  "discard"),
    (12, 50.84,  "discard"),
    (13, None,   "crash"),
    (14, 59.92,  "discard"),
    (15, 36.50,  "discard"),
    (16, 34.37,  "discard"),
    (17, 34.22,  "discard"),
    (18, 34.04,  "discard"),
    (19, 34.21,  "discard"),
    (20, 35.56,  "discard"),
    (21, 34.46,  "discard"),
    (22, 34.31,  "discard"),
    (23, 34.18,  "discard"),
    (24, 34.07,  "discard"),
    (25, 33.75,  "keep"),
]

FLOOR = -300.0

def neg_clamp(v):
    if v is None:
        return None
    neg = -v
    return max(neg, FLOOR)

# Separate series
oe_keep_x, oe_keep_y   = [], []
oe_disc_x, oe_disc_y   = [], []
oe_crash_x, oe_crash_y = [], []

for it, us, st in oe_data:
    if st == "crash":
        oe_crash_x.append(it)
        oe_crash_y.append(FLOOR)
    elif st == "keep":
        oe_keep_x.append(it)
        oe_keep_y.append(neg_clamp(us))
    else:
        oe_disc_x.append(it)
        oe_disc_y.append(neg_clamp(us))

adv_keep_x, adv_keep_y   = [], []
adv_disc_x, adv_disc_y   = [], []
adv_crash_x, adv_crash_y = [], []

for it, us, st in adv_data:
    if st == "crash":
        adv_crash_x.append(it)
        adv_crash_y.append(FLOOR)
    elif st == "keep":
        adv_keep_x.append(it)
        adv_keep_y.append(neg_clamp(us))
    else:
        adv_disc_x.append(it)
        adv_disc_y.append(neg_clamp(us))

# Running best lines
def running_best(data):
    xs, ys = [], []
    best = None
    for it, us, st in data:
        if us is not None and (best is None or us < best):
            best = us
        if best is not None:
            xs.append(it)
            ys.append(-best)
    return xs, ys

oe_best_x, oe_best_y   = running_best(oe_data)
adv_best_x, adv_best_y = running_best(adv_data)

oe_best_val  = -oe_best_y[-1]   if oe_best_y  else None
adv_best_val = -adv_best_y[-1]  if adv_best_y else None

# --- Plot ---
fig, ax = plt.subplots(figsize=(14, 6))

# Colors
OE_KEEP_COLOR  = "#1f77b4"   # dark blue
OE_DISC_COLOR  = "#aec7e8"   # light blue
ADV_KEEP_COLOR = "#d62728"   # dark red
ADV_DISC_COLOR = "#f7a8a8"   # light pink
BEST_OE_COLOR  = "#2ca02c"   # green
BEST_ADV_COLOR = "#17becf"   # teal/cyan
CRASH_COLOR    = "#FFD700"   # gold

# OpenEvolve scatter
ax.scatter(oe_disc_x,  oe_disc_y,  color=OE_DISC_COLOR,  s=40, zorder=3, label="openevolve_discard")
ax.scatter(oe_keep_x,  oe_keep_y,  color=OE_KEEP_COLOR,  s=55, zorder=4, label="openevolve_keep")
ax.plot(oe_keep_x,     oe_keep_y,  color=OE_KEEP_COLOR,  linewidth=1.2, zorder=3, alpha=0.6)

# Advisor scatter
ax.scatter(adv_disc_x, adv_disc_y, color=ADV_DISC_COLOR, s=40, zorder=3, label="advisor_discard")
ax.scatter(adv_keep_x, adv_keep_y, color=ADV_KEEP_COLOR, s=55, zorder=4, label="advisor_keep")
ax.plot(adv_keep_x,    adv_keep_y, color=ADV_KEEP_COLOR, linewidth=1.2, zorder=3, alpha=0.6)

# Best lines
ax.plot(oe_best_x,  oe_best_y,  color=BEST_OE_COLOR,  linewidth=2.0, zorder=5, label="openevolve_best")
ax.plot(adv_best_x, adv_best_y, color=BEST_ADV_COLOR, linewidth=2.0, zorder=5, label="advisor_best")

# Crashes (combine both)
crash_x = oe_crash_x + adv_crash_x
crash_y = oe_crash_y + adv_crash_y
ax.scatter(crash_x, crash_y, color=CRASH_COLOR, marker="x", s=100, linewidths=2, zorder=6,
           label=f"crash ({len(crash_x)})")

# Axes
ax.set_xlabel("Iteration #", fontsize=12)
ax.set_ylabel("Negative Latency (µs)", fontsize=12)
ax.set_title("openevolve vs advisor — vectoradd", fontsize=14, fontweight="bold")
ax.set_xlim(-0.5, 25.5)
ax.set_ylim(FLOOR - 10, -20)
ax.grid(True, color="#cccccc", linewidth=0.8, linestyle="-")
ax.set_axisbelow(True)

# Annotation box
ann_text = (
    f"OpenEvolve best: {oe_best_val:.2f} µs\n"
    f"Advisor best:    {adv_best_val:.2f} µs"
)
ax.annotate(
    ann_text,
    xy=(0.02, 0.97),
    xycoords="axes fraction",
    va="top", ha="left",
    fontsize=10, fontweight="bold", color="#1f3a6e",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#1f77b4", linewidth=1.2),
)

# Footer note
fig.text(0.99, 0.01, "(outliers > 300 µs shown at floor)",
         ha="right", va="bottom", fontsize=8, color="#888888")

ax.legend(loc="upper right", fontsize=9, framealpha=1.0, edgecolor="#aaaaaa")

plt.tight_layout()
plt.savefig("openevolvevsadvisor.png", dpi=150, bbox_inches="tight")
print("Saved openevolvevsadvisor.png")
