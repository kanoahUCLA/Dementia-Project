from pathlib import Path

current = Path(__file__).resolve()

for parent in current.parents:
    if parent.name == "Dementia-Project-Main":
        BASE_DIR = parent
        break
else:
    raise RuntimeError("Could not find 'Dementia-Project-Main' directory in path")

CONTROL_PATH = BASE_DIR / "Ventricle_Team" / "Raw_Data" / "ventricles_control.csv"
DEMENTIA_PATH = BASE_DIR / "Ventricle_Team" / "Raw_Data" / "ventricles_dementia.csv"
OUTPUT_PATH = BASE_DIR / "Ventricle_Team" / "Output"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)