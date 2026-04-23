from pathlib import Path

current = Path(__file__).resolve()

for parent in current.parents:
    if parent.name == "dementia":
        BASE_DIR = parent
        break
else:
    raise RuntimeError("Could not find 'dementia' directory in path")

CONTROL_PATH = BASE_DIR / "Amygdala_Team" / "Raw_Data" / "amygdala_control.csv"
DEMENTIA_PATH = BASE_DIR / "Amygdala_Team" / "Raw_Data" / "amygdala_dementia.csv"
PTINFO_PATH = BASE_DIR / "ptinfo.csv"

OUTPUT_PATH = BASE_DIR / "Amygdala_Team" / "Output"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
