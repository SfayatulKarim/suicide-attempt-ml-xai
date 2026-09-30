# Reproducibility Audit

## Current status

The repository contains a canonical preprocessing pipeline and a canonical primary model-evaluation pipeline. The preprocessing pipeline has been checked against the available source data and produces the documented 1,120-row cleaned representation and 82-column model-ready representation.

The historical manuscript reports a specific XGBoost test-set result. A rerun in the current local environment produced different XGBoost numerical results, while the other primary model results matched the manuscript values. The original notebook did not record the XGBoost package version used for the historical run.

Therefore:

1. The manuscript-reported values are retained as provenance in `results/manuscript_reported_results.csv`.
2. The code does not hard-code those values into model training/evaluation.
3. Current-environment reruns are explicitly labeled as such.
4. Before claiming exact computational reproduction, the original XGBoost version/environment should be identified and pinned.

## Advanced analysis

`src/advanced_analysis.py` generates:

- test-set ROC and precision-recall curves;
- calibration curves and Brier scores;
- bootstrap confidence intervals;
- Alcohol-removal and missingness-signal sensitivity analyses;
- XGBoost SHAP global and grouped attribution;
- optional full SHAP interaction analysis.

The full SHAP interaction tensor can be substantially slower than the other analyses. Use the default 2,000 bootstrap resamples and enable full interactions for the manuscript-grade rerun. For a quick environment check, `--skip-interactions --bootstrap-resamples 500` can be used.

## Data handling

The raw Mendeley dataset and derived CSV files are intentionally excluded from the GitHub release because they contain sensitive suicide-related records. Researchers should obtain the source dataset from the official Mendeley Data record and save it locally as `original_dataset.csv`.
