import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

# ── File paths ────────────────────────────────────────────────────────────────
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
data_dir = os.path.join(desktop, "dementia")

ptinfo = pd.read_csv(os.path.join(data_dir, "ptinfo.csv"))

# ── Baseline visit only ───────────────────────────────────────────────────────
baseline = ptinfo[ptinfo["Visit"] == 1].copy()
baseline = baseline.dropna(subset=["CDR", "nWBV"])

# ── Group strictly by Group column, not MMSE ─────────────────────────────────
baseline = baseline[baseline["Group"].isin(["Nondemented", "Demented"])]

# ── Enforce clean boundaries: Nondemented must have CDR=0 ────────────────────
baseline = baseline[~((baseline["Group"] == "Nondemented") & (baseline["CDR"] != 0))]

baseline["Severity"] = baseline["Group"].map(
    {"Nondemented": "Control", "Demented": "Severe"}
)

ctrl = baseline[baseline["Severity"] == "Control"]
sev  = baseline[baseline["Severity"] == "Severe"]

print(f"Control (Nondemented, CDR=0) n={len(ctrl)}")
print(f"Severe  (Demented)           n={len(sev)}")
print(f"CDR values in Severe: {sorted(sev['CDR'].unique())}")

# ── Pearson correlation ───────────────────────────────────────────────────────
r, p = stats.pearsonr(baseline["CDR"], baseline["nWBV"])
print(f"\nPearson r={r:.3f}, p={p:.4f}")

# ── OLS regression + 95% CI ───────────────────────────────────────────────────
x_all = baseline["CDR"].values
y_all = baseline["nWBV"].values

slope, intercept, _, _, _ = stats.linregress(x_all, y_all)

x_range  = np.linspace(-0.05, x_all.max() + 0.1, 300)
y_line   = slope * x_range + intercept

n        = len(x_all)
x_mean   = np.mean(x_all)
dof      = n - 2
mse      = np.sum((y_all - (slope * x_all + intercept))**2) / dof
se_line  = np.sqrt(mse * (1/n + (x_range - x_mean)**2 / np.sum((x_all - x_mean)**2)))
t_crit   = stats.t.ppf(0.975, dof)
ci_upper = y_line + t_crit * se_line
ci_lower = y_line - t_crit * se_line

# ── Group means ± SEM per CDR stage ──────────────────────────────────────────
cdr_means = baseline.groupby("CDR")["nWBV"].agg(["mean", "sem", "count"]).reset_index()
print("\nMean nWBV per CDR stage:")
print(cdr_means.to_string(index=False))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.grid(False)

ctrl_dark, ctrl_light = "#222222", "#888888"
sev_dark,  sev_light  = "#C0392B", "#F1948A"

np.random.seed(42)
jitter = 0.035

ax.scatter(ctrl["CDR"] + np.random.uniform(-jitter, jitter, len(ctrl)),
           ctrl["nWBV"],
           color=ctrl_light, edgecolors=ctrl_dark, linewidths=0.7,
           s=45, marker="o", zorder=3, alpha=0.75,
           label="Control: CDR = 0")

ax.scatter(sev["CDR"] + np.random.uniform(-jitter, jitter, len(sev)),
           sev["nWBV"],
           color=sev_light, edgecolors=sev_dark, linewidths=0.7,
           s=45, marker="s", zorder=3, alpha=0.75,
           label="Demented CDR ≥ 0.5" )

# OLS line + CI
ax.plot(x_range, y_line, color="#333333", linewidth=2.5, zorder=4)
ax.fill_between(x_range, ci_lower, ci_upper,
                color="#333333", alpha=0.15, zorder=2, label="95% CI")

# Mean ± SEM diamonds per CDR stage
for _, row in cdr_means.iterrows():
    ax.errorbar(row["CDR"], row["mean"], yerr=row["sem"],
                fmt="D", color="white", markeredgecolor="#333333",
                markeredgewidth=1.8, markersize=11,
                ecolor="#333333", elinewidth=1.5, capsize=4, zorder=5)

# CDR stage x-axis labels
cdr_labels = {0: "0\n(None)", 0.5: "0.5\n(Very mild)", 1: "1\n(Mild)", 2: "2\n(Moderate)"}
ax.set_xticks(sorted(baseline["CDR"].unique()))
ax.set_xticklabels([cdr_labels.get(c, str(c)) for c in sorted(baseline["CDR"].unique())],
                   fontsize=11)

# Pearson annotation
# p_str = f"p={p:.4f}" if p >= 0.0001 else "p<0.0001"
# ax.text(0.03, 0.97,
#         f"Pearson r = {r:.3f}\n{p_str}\nn = {n}",
#         transform=ax.transAxes, fontsize=12,
#         va="top", ha="left",
#         bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
#                   edgecolor="#cccccc", alpha=0))

ax.legend(fontsize=10, loc="upper right", edgecolor="#cccccc")
ax.set_xlabel("CDR Stage (Clinical Dementia Rating)", fontsize=13)
ax.set_ylabel("nWBV (Normalised Whole Brain Volume)", fontsize=13)
ax.set_title("C   CDR Stage vs nWBV: Dementia Severity and Brain Volume",
             fontsize=13, fontweight="bold")

plt.tight_layout()

save_path = os.path.join(desktop, "cdr_nwbv_correlation.png")
plt.savefig(save_path, dpi=150)
print(f"\nPlot saved to: {save_path}")
plt.show()