import pandas as pd
from pathlib import Path

# ---- FILE PATHS ----
PT_PATH = Path("~/Desktop/ptinfo.csv").expanduser()
ASEG_PATH = Path("~/Desktop/volumeall.tsv").expanduser()
OUT_PATH = Path("~/Desktop/oasis2_baseline_aseg_labeled.csv").expanduser()

# ---- LOAD ----
pt = pd.read_csv(PT_PATH)
aseg = pd.read_csv(ASEG_PATH, sep="\t")

# ---- CLEAN SUBJECT IDs ----
aseg = aseg.rename(columns={"Measure:volume": "SubjectID"})
aseg["SubjectID"] = aseg["SubjectID"].astype(str).str.strip()

pt["SubjectID"] = pt["Subject ID"].astype(str).str.strip()
pt["Group"] = pt["Group"].astype(str).str.strip()

# ---- BASELINE ONLY ----
if "Visit" in pt.columns:
    pt = pt[pt["Visit"] == 1].copy()

# ---- MAP DEMENTIA STATUS ----
pt["DementiaStatus"] = pt["Group"].map({
    "Nondemented": "Control",
    "Demented": "Demented",
    "Converted": "Control"
})

pt = pt[pt["DementiaStatus"].isin(["Control", "Demented"])]

# ---- ONE ROW PER SUBJECT ----
pt = pt.drop_duplicates("SubjectID")

# ---- MERGE ----
df = aseg.merge(
    pt[["SubjectID", "DementiaStatus", "Group"]],
    on="SubjectID",
    how="left"
)

# ---- DIAGNOSTICS ----
print("Total aseg rows:", len(aseg))
print("Matched labels:", df["DementiaStatus"].notna().sum())
print("Unmatched:", df["DementiaStatus"].isna().sum())

# ---- OPTIONAL: bilateral hippocampus sum ----
if "Left-Hippocampus" in df.columns and "Right-Hippocampus" in df.columns:
    df["HippocampusSum"] = df["Left-Hippocampus"] + df["Right-Hippocampus"]

# ---- SAVE ----
df.to_csv(OUT_PATH, index=False)
print("Saved:", OUT_PATH)

