"""
Generate openevolvevsadvisor.png comparing the advisor run (from TSV) vs
OpenEvolve run1 (data from log).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import csv

# ── Advisor data (from TSV) ──────────────────────────────────────────────────
TSV = "/workspace/vectoradd-advisor/vectoradd/runs/20260608_210059_vectoradd_starting_point/results.tsv"
adv_iters, adv_times, adv_kinds = [], [], []
with open(TSV) as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        it = int(row["agent_iteration"])
        t  = float(row["time_us"])
        st = row["status"]
        adv_iters.append(it)
        adv_times.append(t)
        adv_kinds.append(st)

# ── OpenEvolve data (parsed from log, shifted to start at 0) ─────────────────
# None = crash
oe_raw = [
    (0,  59.813),
    (1,  None),     # crash
    (2,  59.853),
    (3,  59.525),
    (4,  59.796),
    (5,  59.615),
    (6,  60.028),
    (7,  59.864),
    (8,  59.539),
    (9,  59.504),
    (10, 60.104),
    (11, 59.110),
    (12, 59.670),
    (13, 59.138),
    (14, 59.428),
    (15, 59.772),
    (16, 60.147),
    (17, 60.107),
    (18, 59.687),
    (19, 59.515),
    (20, 59.386),
    (21, 59.686),
    (22, 60.639),
    (23, 60.039),
    (24, 59.759),
]
oe_iters, oe_times, oe_kinds = [], [], []
best_so_far = float("inf")
for it, t in oe_raw:
    oe_iters.append(it)
    oe_times.append(t if t is not None else 0.0)
    if t is None:
        oe_kinds.append("crash")
    elif t < best_so_far:
        best_so_far = t
        oe_kinds.append("keep")
    else:
        oe_kinds.append("discard")

# ── Compute best-over-time step lines ────────────────────────────────────────
def best_step(iters, times, kinds):
    bx, by = [], []
    best = float("inf")
    for it, t, k in sorted(zip(iters, times, kinds)):
        if k == "keep" and t > 0:
            best = t
        if best < float("inf"):
            bx.append(it)
            by.append(best)
    return bx, by

adv_bx, adv_by = best_step(adv_iters, adv_times, adv_kinds)
oe_bx,  oe_by  = best_step(oe_iters,  oe_times,  oe_kinds)

adv_best = min(t for t, k in zip(adv_times, adv_kinds) if k == "keep")
oe_best  = min(oe_by) if oe_by else float("inf")

# ── Y-axis range (negative latency) — clip extreme outliers to floor ─────────
CLIP_US = 300.0
all_valid = [t for t in adv_times + oe_times if 0 < t <= CLIP_US]
y_hi = -(min(all_valid) * 0.82)
y_lo = -(CLIP_US * 1.08)

def ny(t):
    if t <= 0:
        return y_lo
    return max(-t, y_lo)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

# OpenEvolve series — blue tones
oe_kx = [it for it, k in zip(oe_iters, oe_kinds) if k == "keep"]
oe_ky = [ny(oe_times[i]) for i, k in enumerate(oe_kinds) if k == "keep"]
oe_dx = [it for it, k in zip(oe_iters, oe_kinds) if k == "discard"]
oe_dy = [ny(oe_times[i]) for i, k in enumerate(oe_kinds) if k == "discard"]
oe_cx = [it for it, k in zip(oe_iters, oe_kinds) if k == "crash"]
if oe_kx:
    ax.scatter(oe_kx, oe_ky, c="#3b82f6", s=70, zorder=5,
               edgecolors="white", linewidths=0.5, label="openevolve keep")
if oe_dx:
    ax.scatter(oe_dx, oe_dy, c="#93c5fd", s=40, zorder=4,
               edgecolors="white", linewidths=0.3, alpha=0.8, label="openevolve discard")
if oe_bx:
    ax.step(oe_bx, [-t for t in oe_by], where="post",
            color="#3b82f6", linewidth=2, label="openevolve best", zorder=6)

# Advisor series — green tones
adv_kx = [it for it, k in zip(adv_iters, adv_kinds) if k == "keep"]
adv_ky = [ny(adv_times[i]) for i, k in enumerate(adv_kinds) if k == "keep"]
adv_dx = [it for it, k in zip(adv_iters, adv_kinds) if k == "discard"]
adv_dy = [ny(adv_times[i]) for i, k in enumerate(adv_kinds) if k == "discard"]
adv_cx = [it for it, k in zip(adv_iters, adv_kinds) if k == "crash"]
if adv_kx:
    ax.scatter(adv_kx, adv_ky, c="#22c55e", s=70, zorder=5,
               edgecolors="white", linewidths=0.5, label="advisor keep")
if adv_dx:
    ax.scatter(adv_dx, adv_dy, c="#ef4444", s=40, zorder=4,
               edgecolors="white", linewidths=0.3, alpha=0.7, label="advisor discard")
if adv_bx:
    ax.step(adv_bx, [-t for t in adv_by], where="post",
            color="#22c55e", linewidth=2, label="advisor best", zorder=6)

# Crashes (both OE and advisor)
all_cx = oe_cx + adv_cx
if all_cx:
    ax.scatter(all_cx, [y_lo] * len(all_cx), c="#fbbf24", s=40, zorder=3,
               marker="x", linewidths=1.5, label=f"crash ({len(all_cx)})", alpha=0.8)

ax.set_ylim(y_lo * 1.05, y_hi)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
ax.set_xlabel("Iteration #", fontsize=12)
ax.set_ylabel("Negative Latency (-μs)", fontsize=12)
ax.set_title("openevolve vs advisor — vectoradd", fontsize=14, fontweight="bold")
ax.legend(loc="upper right", framealpha=0.9)
ax.grid(True, alpha=0.3)

# Best-time annotation
ax.annotate(
    f"OpenEvolve best: {oe_best:.2f} μs\nAdvisor best: {adv_best:.2f} μs",
    xy=(0.02, 0.98), xycoords="axes fraction",
    va="top", fontsize=11, fontweight="bold", color="#1e3a5f",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#3b82f6", alpha=0.9),
)

# Clip note
ax.annotate(
    f"(outliers > {CLIP_US:.0f} μs shown at floor)",
    xy=(0.98, 0.02), xycoords="axes fraction",
    ha="right", va="bottom", fontsize=9, color="#6b7280",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#d1d5db", alpha=0.8),
)

fig.tight_layout()
out = "/workspace/vectoradd-advisor/openevolvevsadvisor.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved {out}")
