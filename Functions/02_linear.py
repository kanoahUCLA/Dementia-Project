import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, ttest_ind
import os

# ── File paths ────────────────────────────────────────────────────────────────
desktop = os.path.join(os.path.expanduser("~"), "Desktop")

control_df  = pd.read_csv(os.path.join(desktop, "control_patients.csv"),  index_col=0)
dementia_df = pd.read_csv(os.path.join(desktop, "dementia_patients.csv"), index_col=0)
ptinfo      = pd.read_csv(os.path.join(desktop, "ptinfo.csv"))

# ── Get one nWBV value per subject (first visit only) ─────────────────────────
ptinfo_first = (ptinfo.drop_duplicates(subset="Subject ID")
                      .set_index("Subject ID")[["nWBV"]])

# ── Extract nWBV for each group ───────────────────────────────────────────────
ctrl_nwbv = ptinfo_first.loc[ptinfo_first.index.isin(control_df.index), "nWBV"].dropna()
dem_nwbv  = ptinfo_first.loc[ptinfo_first.index.isin(dementia_df.index), "nWBV"].dropna()

print(f"Control  n={len(ctrl_nwbv)}")
print(f"Demented n={len(dem_nwbv)}")

# ── Fit Gaussians ─────────────────────────────────────────────────────────────
ctrl_mu, ctrl_std = norm.fit(ctrl_nwbv)
dem_mu,  dem_std  = norm.fit(dem_nwbv)

print(f"\nControl  — mean: {ctrl_mu:.4f}  std: {ctrl_std:.4f}")
print(f"Demented — mean: {dem_mu:.4f}  std: {dem_std:.4f}")

# ── T-test ────────────────────────────────────────────────────────────────────
t_stat, p_val = ttest_ind(ctrl_nwbv, dem_nwbv)
print(f"\nIndependent t-test: t={t_stat:.3f}, p={p_val:.4f}")
if p_val < 0.05:
    print("=> Statistically significant difference (p < 0.05)")
else:
    print("=> No statistically significant difference (p >= 0.05)")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(ctrl_nwbv, bins=15, color="black", alpha=0.4, density=True,
        label="Control (Nondemented)", edgecolor="black")
ax.hist(dem_nwbv,  bins=15, color="red",   alpha=0.4, density=True,
        label="Demented", edgecolor="darkred")

x = np.linspace(min(ctrl_nwbv.min(), dem_nwbv.min()) - 0.02,
                max(ctrl_nwbv.max(), dem_nwbv.max()) + 0.02, 300)

ax.plot(x, norm.pdf(x, ctrl_mu, ctrl_std), color="black", linewidth=2.5,
        label=f"Control fit  μ={ctrl_mu:.3f}  σ={ctrl_std:.3f}")
ax.plot(x, norm.pdf(x, dem_mu,  dem_std),  color="red",   linewidth=2.5,
        label=f"Demented fit  μ={dem_mu:.3f}  σ={dem_std:.3f}")

# Dotted vertical lines from peak of each Gaussian down to x-axis
ctrl_peak = norm.pdf(ctrl_mu, ctrl_mu, ctrl_std)
dem_peak  = norm.pdf(dem_mu,  dem_mu,  dem_std)

ax.vlines(ctrl_mu, 0, ctrl_peak, color="black", linewidth=1.8, linestyle="dotted")
ax.vlines(dem_mu,  0, dem_peak,  color="red",   linewidth=1.8, linestyle="dotted")

# Draw legend and get its bounding box to position stats just below
legend = ax.legend(fontsize=11, loc="upper right")
fig.canvas.draw()
legend_bb = legend.get_window_extent().transformed(ax.transAxes.inverted())

# Add p-value annotation flush below the legend, no box
p_text = f"t-test: t={t_stat:.2f}, p={p_val:.4f}"
ax.text(legend_bb.x1, legend_bb.y0 - 0.01, p_text, transform=ax.transAxes,
        fontsize=11, ha="right", va="top")

ax.set_xlabel("nWBV (Normalised Whole Brain Volume)", fontsize=13)
ax.set_ylabel("Density", fontsize=13)
ax.set_title("Gaussian fit — nWBV: Demented vs Control", fontsize=14)
plt.tight_layout()

save_path = os.path.join(desktop, "gaussian_nWBV.png")
plt.savefig(save_path, dpi=150)
print(f"\nPlot saved to: {save_path}")
plt.show()