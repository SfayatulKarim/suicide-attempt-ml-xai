"""Run the canonical preprocessing and primary model evaluation pipeline.

Expected input: original_dataset.csv in the repository root.
Outputs are written to cleaned_reproducible_dataset.csv,
model_ready_reproducible_dataset.csv, and results/.
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

for script in ["src/preprocessing.py", "src/modeling.py"]:
    subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True)

print("\nCanonical preprocessing + primary modeling completed.")
print("For SHAP, calibration, bootstrap, and sensitivity analyses, see the preserved research notebook and the analysis documentation.")
