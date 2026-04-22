import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from Paths import CONTROL_PATH, DEMENTIA_PATH, OUTPUT_PATH

# -----------------------------
# USER SETTINGS
# -----------------------------
COLUMN = "Left-Amygdala"
GROUP1_LABEL = "Control"
GROUP2_LABEL = "Dementia"
TITLE = "Left Amygdala Volume"
YLABEL = "Volume (mm³)"
OUTPUT_NAME = "left_amygdala_boxplot.png"
# -----------------------------

# Load data
control_df = pd.read_csv(CONTROL_PATH)
dementia_df = pd.read_csv(DEMENTIA_PATH)

# Check that the column exists
for name, df in [("Control", control_df), ("Dementia", dementia_df)]:
    if COLUMN not in df.columns:
        raise ValueError(f"Missing column '{COLUMN}' in {name} dataset")

# Extract data
control_values = control_df[COLUMN].dropna()
dementia_values = dementia_df[COLUMN].dropna()

# Stats
t_stat, p_val = stats.ttest_ind(control_values, dementia_values)

# Labels with sample size
GROUP1_LABEL = f"Control (n={len(control_values)})"
GROUP2_LABEL = f"Dementia (n={len(dementia_values)})"

# Plot
fig, ax = plt.subplots(figsize=(6, 5))
ax.boxplot([control_values, dementia_values], labels=[GROUP1_LABEL, GROUP2_LABEL])

# Overlay data points
for x, vals in zip([1, 2], [control_values, dementia_values]):
    ax.scatter([x] * len(vals), vals, alpha=0.4, color='black', s=15, zorder=3)

ax.set_title(f"{TITLE}\np = {p_val:.4f}")
ax.set_ylabel(YLABEL)
plt.tight_layout()

# Save
output_path = OUTPUT_PATH / OUTPUT_NAME
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Figure saved to: {output_path}")