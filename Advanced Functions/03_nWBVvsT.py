import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

desktop  = os.path.join(os.path.expanduser("~"), "Desktop")
data_dir = os.path.join(desktop, "dementia")
ptinfo   = pd.read_csv(os.path.join(data_dir, "ptinfo.csv"))

ptinfo["Years_since_baseline"] = ptinfo["MR Delay"] / 365.25

# ── Baseline nWBV & Group per subject ─────────────────────────────────────────
baseline_visit = ptinfo[ptinfo["Visit"] == 1].set_index("Subject ID")
ptinfo = ptinfo.join(baseline_visit["nWBV"].rename("nWBV_baseline"), on="Subject ID")
ptinfo = ptinfo.join(baseline_visit["Group"].rename("Group_baseline"), on="Subject ID")
ptinfo["delta_nWBV"] = ptinfo["nWBV"] - ptinfo["nWBV_baseline"]

# ── Group-based sorting, CDR=0 enforced for Nondemented ──────────────────────
ptinfo = ptinfo[ptinfo["Group_baseline"].isin(["Nondemented", "Demented"])]
ptinfo = ptinfo[~((ptinfo["Group_baseline"] == "Nondemented") & (ptinfo["CDR"] != 0))]
ptinfo = ptinfo.dropna(subset=["Years_since_baseline", "nWBV", "delta_nWBV"])

ptinfo["Severity"] = ptinfo["Group_baseline"].map(
    {"Nondemented": "Control", "Demented": "Demented"}
)

print("Subject counts per group:")
print(ptinfo.groupby("Severity")["Subject ID"].nunique())

severity_colors  = {"Control": ("#222222", "#888888"), "Demented": ("#C0392B", "#F1948A")}
severity_markers = {"Control": "o", "Demented": "s"}
order = ["Control", "Demented"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor("white")

x_max   = ptinfo["Years_since_baseline"].max()
x_range = np.linspace(0, x_max + 0.2, 300)

for ax, y_col, ylabel, title in [
    (ax1, "nWBV",       "Normalized Whole Brain Volume (nWBV)",  "A   Longitudinal Trajectories by Group"),
    (ax2, "delta_nWBV", "Change in nWBV from baseline (ΔnWBV)", "B   Normalized Change from Baseline"),
]:
    ax.set_facecolor("white")
    ax.grid(False)

    for severity in order:
        color, lcolor = severity_colors[severity]
        marker = severity_markers[severity]
        grp = ptinfo[ptinfo["Severity"] == severity]

        for subject_id, subj_df in grp.groupby("Subject ID"):
            subj_df = subj_df.sort_values("Years_since_baseline")
            if len(subj_df) > 1:
                ax.plot(subj_df["Years_since_baseline"], subj_df[y_col],
                        color=lcolor, linewidth=0.9, alpha=0.6, zorder=1)
            ax.scatter(subj_df["Years_since_baseline"], subj_df[y_col],
                       color=lcolor, edgecolors=color, linewidths=0.6,
                       s=28, marker=marker, zorder=2, alpha=0.85)

    for severity in order:
        color, _ = severity_colors[severity]
        grp = ptinfo[ptinfo["Severity"] == severity]
        x = grp["Years_since_baseline"].values
        y = grp[y_col].values

        slope, intercept, r, p, se = stats.linregress(x, y)
        y_pred_all = slope * x + intercept
        residuals  = y - y_pred_all
        n_obs  = len(x)
        n_subj = grp["Subject ID"].nunique()
        dof    = n_obs - 2
        mse    = np.sum(residuals**2) / dof
        x_mean = np.mean(x)
        x_line = slope * x_range + intercept

        se_line  = np.sqrt(mse * (1/n_obs + (x_range - x_mean)**2 / np.sum((x - x_mean)**2)))
        t_crit   = stats.t.ppf(0.975, dof)
        ci_upper = x_line + t_crit * se_line
        ci_lower = x_line - t_crit * se_line

        ax.plot(x_range, x_line, color=color, linewidth=3.0, zorder=4)
        ax.fill_between(x_range, ci_lower, ci_upper, color=color, alpha=0.18, zorder=3)

        print(f"[{title[:1]}] {severity}: slope={slope:.4f}/yr, r={r:.3f}, p={p:.4f} | n_subjects={n_subj}, n_obs={n_obs}")

    if y_col == "delta_nWBV":
        ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.4, zorder=0)

    ax.set_xlabel("Years since baseline", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(-0.2, x_max + 0.5)

from matplotlib.lines import Line2D
leg_handles = [
    Line2D([0],[0], color=severity_colors["Control"][0],  linewidth=2.5, label="Control (Nondemented, CDR=0)"),
    Line2D([0],[0], color=severity_colors["Demented"][0], linewidth=2.5, label="Demented"),
]
ax2.legend(handles=leg_handles, fontsize=10, loc="lower left",
           framealpha=0.9, edgecolor="#cccccc")

plt.tight_layout()
save_path = os.path.join(desktop, "longitudinal_trajectories_MMSE.png")
plt.savefig(save_path, dpi=150)
print(f"\nPlot saved to: {save_path}")
plt.show()