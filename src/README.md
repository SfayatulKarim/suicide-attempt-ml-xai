# Source Code

- `preprocessing.py`: raw-to-clean/model-ready transformation and cleaning decisions.
- `modeling.py`: primary 5-model CV and untouched-test evaluation.
- `advanced_analysis.py`: ROC/PR, calibration, bootstrap CIs, sensitivity analyses, SHAP, and optional SHAP interactions.
- `run_all.py`: runs preprocessing and primary modeling sequentially.

All model training uses explicit random seeds where supported. The advanced pipeline records the current Python/package environment in `results/environment_versions.json`.
