import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind
import os

# ── File paths ────────────────────────────────────────────────────────────────
desktop = os.path.join(os.path.expanduser("~"), "desktop")
data_dir = os.path.join(desktop, "dementia")
ptinfo  = pd.read_csv(os.path.join(data_dir, "ptinfo.csv"))

# ── Filter settings ───────────────────────────────────────────────────────────
CDR_THRESHOLD  = 0.5
MMSE_THRESHOLD = 25

# ── Clean columns ─────────────────────────────────────────────────────────────
ptinfo.columns = ptinfo.columns.str.strip()

required = ["Subject ID", "CDR", "MMSE", "nWBV"]
missing = [c for c in required if c not in ptinfo.columns]
if missing:
    raise ValueError(f"ptinfo.csv is missing required columns: {missing}")

ptinfo = ptinfo.dropna(subset=required).copy()

# ── Keep one nWBV value per subject (first visit only) ───────────────────────
if "Visit" in ptinfo.columns:
    ptinfo_first = (
        ptinfo.sort_values(["Subject ID", "Visit"])
              .drop_duplicates(subset="Subject ID", keep="first")
              .set_index("Subject ID")
    )
elif "MR Delay" in ptinfo.columns:
    ptinfo_first = (
        ptinfo.sort_values(["Subject ID", "MR Delay"])
              .drop_duplicates(subset="Subject ID", keep="first")
              .set_index("Subject ID")
    )
else:
    ptinfo_first = (
        ptinfo.drop_duplicates(subset="Subject ID", keep="first")
              .set_index("Subject ID")
    )

# ── Build groups directly from ptinfo ────────────────────────────────────────
control_df = ptinfo_first[
    (ptinfo_first["CDR"] == 0) &
    (ptinfo_first["MMSE"] >= MMSE_THRESHOLD)
].copy()

dementia_df = ptinfo_first[
    (ptinfo_first["CDR"] > CDR_THRESHOLD) &
    (ptinfo_first["MMSE"] < MMSE_THRESHOLD)
].copy()

ctrl_nwbv = control_df["nWBV"].dropna()
dem_nwbv  = dementia_df["nWBV"].dropna()

print(f"Control  n={len(ctrl_nwbv)}")
print(f"Demented n={len(dem_nwbv)}")

print("\nDemented CDR counts:")
print(dementia_df["CDR"].value_counts().sort_index())

print("\nGroup summaries:")
print(f"Control  — mean={ctrl_nwbv.mean():.4f}, median={ctrl_nwbv.median():.4f}, sd={ctrl_nwbv.std():.4f}")
print(f"Demented — mean={dem_nwbv.mean():.4f}, median={dem_nwbv.median():.4f}, sd={dem_nwbv.std():.4f}")

# ── Welch t-test ──────────────────────────────────────────────────────────────
t_stat, p_val = ttest_ind(ctrl_nwbv, dem_nwbv, equal_var=False)
print(f"\nWelch t-test: t={t_stat:.3f}, p={p_val:.4f}")

# ── Plot: boxplot + jitter ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

data = [ctrl_nwbv.values, dem_nwbv.values]
positions = [1, 2]

bp = ax.boxplot(
    data,
    positions=positions,
    widths=0.5,
    patch_artist=True,
    showfliers=False,
    medianprops=dict(color="black", linewidth=2),
    boxprops=dict(linewidth=1.5),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5)
)

# Fill colors
box_colors = ["#B0B0B0", "#F1948A"]
edge_colors = ["#222222", "#C0392B"]

for patch, fc, ec in zip(bp["boxes"], box_colors, edge_colors):
    patch.set_facecolor(fc)
    patch.set_edgecolor(ec)
    patch.set_alpha(0.75)

for whisker, ec in zip(bp["whiskers"], [edge_colors[0], edge_colors[0], edge_colors[1], edge_colors[1]]):
    whisker.set_color(ec)

for cap, ec in zip(bp["caps"], [edge_colors[0], edge_colors[0], edge_colors[1], edge_colors[1]]):
    cap.set_color(ec)

# Jittered raw points
rng = np.random.default_rng(42)
jitter_width = 0.10

x_ctrl = positions[0] + rng.uniform(-jitter_width, jitter_width, len(ctrl_nwbv))
x_dem  = positions[1] + rng.uniform(-jitter_width, jitter_width, len(dem_nwbv))

ax.scatter(
    x_ctrl, ctrl_nwbv,
    s=38, alpha=0.75,
    color="#888888", edgecolors="#222222", linewidths=0.7,
    zorder=3
)

ax.scatter(
    x_dem, dem_nwbv,
    s=38, alpha=0.75,
    color="#F1948A", edgecolors="#C0392B", linewidths=0.7,
    marker="s", zorder=3
)

# Means as horizontal markers
ax.scatter(positions[0], ctrl_nwbv.mean(), marker="_", s=700, color="black", linewidths=2.2, zorder=4)
ax.scatter(positions[1], dem_nwbv.mean(),  marker="_", s=700, color="#8B0000", linewidths=2.2, zorder=4)

# Annotation
y_max = max(ctrl_nwbv.max(), dem_nwbv.max())
y_min = min(ctrl_nwbv.min(), dem_nwbv.min())
y_range = y_max - y_min

line_y = y_max + 0.08 * y_range
text_y = y_max + 0.11 * y_range

ax.plot([1, 1, 2, 2], [line_y - 0.003, line_y, line_y, line_y - 0.003],
        color="black", linewidth=1.4)
ax.text(
    1.5, text_y,
    f"Welch t-test: t={t_stat:.2f}, p={p_val:.4f}",
    ha="center", va="bottom", fontsize=11
)

# Labels / formatting
ax.set_xticks(positions)
ax.set_xticklabels([
    f"Control\n(CDR=0, MMSE≥25)\n n={len(ctrl_nwbv)}",
    f"Demented\n(CDR>0.5, MMSE<25)\n n={len(dem_nwbv)}"
], fontsize=11)

ax.set_ylabel("nWBV (Normalized Whole Brain Volume)", fontsize=13)
ax.set_title("nWBV by Group", fontsize=14, fontweight="bold")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlim(0.5, 2.5)
ax.set_ylim(y_min - 0.03 * y_range, y_max + 0.18 * y_range)

plt.tight_layout()

save_path = os.path.join(desktop, "boxplot_jitter_nWBV_filtered.png")
plt.savefig(save_path, dpi=200, bbox_inches="tight")
print(f"\nPlot saved to: {save_path}")
plt.show()