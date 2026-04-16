import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, ttest_ind
import os

# ── File paths ────────────────────────────────────────────────────────────────
desktop = os.path.join(os.path.expanduser("~"), "desktop")
data_dir = os.path.join(desktop, "dementia",'MUST DOWNLOAD')
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

# ── Extract nWBV for each group ───────────────────────────────────────────────
ctrl_nwbv = control_df["nWBV"].dropna()
dem_nwbv  = dementia_df["nWBV"].dropna()

print(f"Control  n={len(ctrl_nwbv)}")
print(f"Demented n={len(dem_nwbv)}")

print("\nDemented CDR counts:")
print(dementia_df["CDR"].value_counts().sort_index())

# ── Fit Gaussians ─────────────────────────────────────────────────────────────
ctrl_mu, ctrl_std = norm.fit(ctrl_nwbv)
dem_mu,  dem_std  = norm.fit(dem_nwbv)

print(f"\nControl  — mean: {ctrl_mu:.4f}  std: {ctrl_std:.4f}")
print(f"Demented — mean: {dem_mu:.4f}  std: {dem_std:.4f}")

# ── T-test ────────────────────────────────────────────────────────────────────
t_stat, p_val = ttest_ind(ctrl_nwbv, dem_nwbv, equal_var=False)
print(f"\nWelch t-test: t={t_stat:.3f}, p={p_val:.4f}")
if p_val < 0.05:
    print("=> Statistically significant difference (p < 0.05)")
else:
    print("=> No statistically significant difference (p >= 0.05)")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(ctrl_nwbv, bins=5, color="black", alpha=0.4, density=True,
        label="Control (CDR=0, MMSE≥25)", edgecolor="black")
ax.hist(dem_nwbv, bins=5, color="red", alpha=0.4, density=True,
        label="Demented (CDR>0.5, MMSE<25)", edgecolor="darkred")

x = np.linspace(min(ctrl_nwbv.min(), dem_nwbv.min()) - 0.02,
                max(ctrl_nwbv.max(), dem_nwbv.max()) + 0.02, 300)

ax.plot(x, norm.pdf(x, ctrl_mu, ctrl_std), color="black", linewidth=2.5,
        label=f"Control fit  μ={ctrl_mu:.3f}  σ={ctrl_std:.3f}")
ax.plot(x, norm.pdf(x, dem_mu, dem_std), color="red", linewidth=2.5,
        label=f"Demented fit  μ={dem_mu:.3f}  σ={dem_std:.3f}")

# Dotted vertical lines from peak of each Gaussian down to x-axis
ctrl_peak = norm.pdf(ctrl_mu, ctrl_mu, ctrl_std)
dem_peak  = norm.pdf(dem_mu, dem_mu, dem_std)

ax.vlines(ctrl_mu, 0, ctrl_peak, color="black", linewidth=1.8, linestyle="dotted")
ax.vlines(dem_mu, 0, dem_peak, color="red", linewidth=1.8, linestyle="dotted")

# Legend + p-value text
legend = ax.legend(fontsize=11, loc="upper right")
fig.canvas.draw()
legend_bb = legend.get_window_extent().transformed(ax.transAxes.inverted())

p_text = f"Welch t-test: t={t_stat:.2f}, p={p_val:.4f}"
ax.text(legend_bb.x1, legend_bb.y0 - 0.01, p_text, transform=ax.transAxes,
        fontsize=11, ha="right", va="top")

ax.set_xlabel("nWBV (Normalized Whole Brain Volume)", fontsize=13)
ax.set_ylabel("Density", fontsize=13)
ax.set_title("Gaussian fit — nWBV: Demented vs Control", fontsize=14)

plt.tight_layout()

save_path = os.path.join(desktop, "gaussian_nWBV_filtered.png")
plt.savefig(save_path, dpi=150)
print(f"\nPlot saved to: {save_path}")
plt.show()