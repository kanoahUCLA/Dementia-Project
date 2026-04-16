import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os
from matplotlib.lines import Line2D

# ── Paths ─────────────────────────────────────────────────────────────────────
desktop  = os.path.join(os.path.expanduser("~"), "Desktop")
data_dir = os.path.join(desktop, "dementia", "MUST DOWNLOAD")

OASIS1_PATH = os.path.join(data_dir, "oasis1.csv")
OASIS2_PATH = os.path.join(data_dir, "ptinfo.csv")

# ── Common filter settings ────────────────────────────────────────────────────
CDR_THRESHOLD  = 0.5
MMSE_THRESHOLD = 25


# ── OASIS-1 loader ────────────────────────────────────────────────────────────
def load_oasis1_filtered(path, cdr_min=0.5, mmse_max=25, include_controls=True):
    """
    OASIS-1: cross-sectional.
    Keeps rows for:
      - Demented: CDR > cdr_min and MMSE < mmse_max
      - Control:  CDR == 0 and MMSE >= mmse_max
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    required = ["CDR", "MMSE", "nWBV"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"OASIS-1 missing required columns: {missing}")

    df = df.dropna(subset=["CDR", "MMSE", "nWBV"]).copy()

    if "Subject ID" not in df.columns:
        if "ID" in df.columns:
            df = df.rename(columns={"ID": "Subject ID"})
        else:
            df["Subject ID"] = [f"OAS1_{i}" for i in range(len(df))]

    df["Dataset"] = "OASIS-1"
    df["Visit"] = np.nan
    df["Years_since_baseline"] = 0.0
    df["Years_since_anchor"] = 0.0
    df["delta_nWBV_anchor"] = 0.0

    frames = []

    cases = df[(df["CDR"] > cdr_min) & (df["MMSE"] < mmse_max)].copy()
    cases["Severity"] = "Demented"
    frames.append(cases)

    if include_controls:
        ctrls = df[(df["CDR"] == 0) & (df["MMSE"] >= mmse_max)].copy()
        ctrls["Severity"] = "Control"
        frames.append(ctrls)

    out = pd.concat(frames, ignore_index=True)
    return out


# ── OASIS-2 loader ────────────────────────────────────────────────────────────
def load_oasis2_filtered(path, cdr_min=0.5, mmse_max=25, include_controls=True):
    """
    OASIS-2: longitudinal.
    Visit-level filtering for plotting:
      - Demented visits: CDR > cdr_min and MMSE < mmse_max
      - Control visits:  CDR == 0 and MMSE >= mmse_max

    IMPORTANT:
    After filtering, each subject is re-anchored so that:
      - first included visit has Years_since_anchor = 0
      - first included visit has delta_nWBV_anchor = 0
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    required = ["Subject ID", "CDR", "MMSE", "nWBV"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"OASIS-2 missing required columns: {missing}")

    df = df.dropna(subset=["Subject ID", "CDR", "MMSE", "nWBV"]).copy()

    if "MR Delay" in df.columns:
        df["Years_since_baseline"] = df["MR Delay"] / 365.25
    else:
        df["Years_since_baseline"] = 0.0

    df["Dataset"] = "OASIS-2"

    frames = []

    # Demented rows
    cases = df[(df["CDR"] > cdr_min) & (df["MMSE"] < mmse_max)].copy()
    cases["Severity"] = "Demented"
    frames.append(cases)

    # Control rows
    if include_controls:
        ctrls = df[(df["CDR"] == 0) & (df["MMSE"] >= mmse_max)].copy()
        ctrls["Severity"] = "Control"
        frames.append(ctrls)

    out = pd.concat(frames, ignore_index=True)

    # Re-anchor within the FILTERED data
    out = out.sort_values(["Severity", "Subject ID", "Years_since_baseline"]).copy()

    first_time = out.groupby("Subject ID")["Years_since_baseline"].transform("min")
    out["Years_since_anchor"] = out["Years_since_baseline"] - first_time

    first_nwbv = out.groupby("Subject ID")["nWBV"].transform("first")
    out["delta_nWBV_anchor"] = out["nWBV"] - first_nwbv

    return out


# ── Combined plotting function ────────────────────────────────────────────────
def plot_nwbv_trajectories(df, save_name="filtered_longitudinal_trajectories.png"):
    severity_colors  = {"Control": ("#222222", "#888888"), "Demented": ("#C0392B", "#F1948A")}
    severity_markers = {"Control": "o", "Demented": "s"}
    order = ["Control", "Demented"]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.patch.set_facecolor("white")

    x_max = max(df["Years_since_anchor"].max(), 0.1)
    x_range = np.linspace(0, x_max + 0.2, 300)

    ax.set_facecolor("white")
    ax.grid(False)

    # ── spaghetti + points ─────────────────────────────────────────────────────
    for severity in order:
        grp = df[df["Severity"] == severity].copy()
        if grp.empty:
            continue

        color, lcolor = severity_colors[severity]
        marker = severity_markers[severity]

        for subject_id, subj_df in grp.groupby("Subject ID"):
            subj_df = subj_df.sort_values("Years_since_anchor")

            if len(subj_df) > 1:
                ax.plot(
                    subj_df["Years_since_anchor"],
                    subj_df["delta_nWBV_anchor"],
                    color=lcolor,
                    linewidth=0.9,
                    alpha=0.6,
                    zorder=1
                )

            ax.scatter(
                subj_df["Years_since_anchor"],
                subj_df["delta_nWBV_anchor"],
                color=lcolor,
                edgecolors=color,
                linewidths=0.6,
                s=28,
                marker=marker,
                zorder=2,
                alpha=0.85
            )

    # ── regression + CI ────────────────────────────────────────────────────────
    for severity in order:
        grp = df[df["Severity"] == severity].copy()
        if grp.empty or grp["Years_since_anchor"].nunique() < 2:
            print(f"{severity}: skipped regression (no x variation)")
            continue

        color, _ = severity_colors[severity]
        x = grp["Years_since_anchor"].values
        y = grp["delta_nWBV_anchor"].values

        slope, intercept, r, p, se = stats.linregress(x, y)
        y_pred_all = slope * x + intercept
        residuals = y - y_pred_all

        n_obs = len(x)
        dof = n_obs - 2
        if dof <= 0:
            continue

        mse = np.sum(residuals**2) / dof
        x_mean = np.mean(x)
        denom = np.sum((x - x_mean)**2)
        if denom == 0:
            continue

        y_line = slope * x_range + intercept
        se_line = np.sqrt(mse * (1/n_obs + (x_range - x_mean)**2 / denom))
        t_crit = stats.t.ppf(0.975, dof)
        ci_upper = y_line + t_crit * se_line
        ci_lower = y_line - t_crit * se_line

        ax.plot(x_range, y_line, color=color, linewidth=3.0, zorder=4)
        ax.fill_between(x_range, ci_lower, ci_upper, color=color, alpha=0.18, zorder=3)

        print(
            f"{severity}: slope={slope:.4f}/yr, r={r:.3f}, p={p:.4f}, "
            f"n_subjects={grp['Subject ID'].nunique()}, n_obs={n_obs}"
        )

    # ── formatting ─────────────────────────────────────────────────────────────
    ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.4, zorder=0)

    ax.set_xlabel("Years since first included visit", fontsize=13)
    ax.set_ylabel("Change in nWBV (ΔnWBV)", fontsize=13)
    ax.set_title("Filtered Longitudinal Change in nWBV", fontsize=13, fontweight="bold")
    ax.set_xlim(-0.2, x_max + 0.5)

    leg_handles = [
        Line2D([0], [0], color="#222222", linewidth=2.5, label="Control (CDR=0, MMSE≥25)"),
        Line2D([0], [0], color="#C0392B", linewidth=2.5, label="Demented (CDR>0.5, MMSE<25)")
    ]
    ax.legend(handles=leg_handles, fontsize=10, loc="lower left",
              framealpha=0.9, edgecolor="#cccccc")

    plt.tight_layout()
    save_path = os.path.join(desktop, save_name)
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to: {save_path}")
    plt.show()


# ── Pull separately, then combine ─────────────────────────────────────────────
oasis1_filtered = load_oasis1_filtered(
    OASIS1_PATH,
    cdr_min=CDR_THRESHOLD,
    mmse_max=MMSE_THRESHOLD,
    include_controls=True
)

oasis2_filtered = load_oasis2_filtered(
    OASIS2_PATH,
    cdr_min=CDR_THRESHOLD,
    mmse_max=MMSE_THRESHOLD,
    include_controls=True
)

common_cols = sorted(set(oasis1_filtered.columns).union(set(oasis2_filtered.columns)))
oasis1_filtered = oasis1_filtered.reindex(columns=common_cols)
oasis2_filtered = oasis2_filtered.reindex(columns=common_cols)

all_filtered = pd.concat([oasis1_filtered, oasis2_filtered], ignore_index=True)

print("\nCounts by dataset and severity:")
print(all_filtered.groupby(["Dataset", "Severity"]).size())

print("\nCounts by dataset and CDR:")
print(all_filtered.groupby(["Dataset", "CDR"]).size())

preview_cols = [c for c in ["Dataset", "Subject ID", "Visit", "CDR", "MMSE", "nWBV", "Group", "Severity"] if c in all_filtered.columns]
print("\nPreview:")
print(all_filtered[preview_cols].head(20))

# ── Plot ──────────────────────────────────────────────────────────────────────
plot_nwbv_trajectories(
    all_filtered,
    save_name="nWBV_filtered_CDRgt0.5_MMSElt25.png"
)