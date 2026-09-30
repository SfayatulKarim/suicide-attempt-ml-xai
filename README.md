# Artificial Intelligence for Retrospective Classification of Documented Suicide-Attempt History

Reproducible machine-learning and explainable-AI analysis using the public Mendeley Data dataset **Behavioral Data-Driven Prediction of Suicide Risk Using Machine Learning Approaches**, version 2 (DOI: `10.17632/bpns8vzwrj.2`).

## Study scope

This repository supports a retrospective comparative analysis of Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (RBF), and XGBoost for classification of documented suicide-attempt history.

**This is not a prospective suicide-risk prediction system and is not intended for clinical decision-making.**

## Main methodology

- 1,128 original records
- reproducible cleaning and categorical standardization
- exact duplicate assessment; 1,120 observations retained
- `Past Attempt` excluded from primary modeling because of severe target leakage
- heterogeneous free-text `Illness` excluded from the primary feature set
- 80/20 stratified train/test split (`random_state=42`)
- 5-fold stratified cross-validation on the training set
- Logistic Regression, Decision Tree, Random Forest, SVM (RBF), XGBoost
- Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, Average Precision
- calibration, bootstrap confidence intervals, SHAP, SHAP interactions, and sensitivity analyses

## Repository structure

```text
.
├── README.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── data/
│   └── README.md
├── docs/
│   ├── github_release_checklist.md
│   └── reproducibility_audit.md
├── figures/
│   └── README.md
├── notebooks/
│   └── Suicide_Attempt_ML_Research_Final.ipynb
├── results/
│   ├── manuscript_reported_results.csv
│   └── README.md
└── src/
    ├── preprocessing.py
    ├── modeling.py
    ├── advanced_analysis.py
    ├── run_all.py
    └── README.md
```

## Data

The source dataset is **not redistributed in this repository**. Obtain it from the official Mendeley Data record and place the downloaded CSV at the repository root as:

```text
original_dataset.csv
```

Preserve the raw file unchanged. The cleaning scripts generate derived CSV files locally; these should not be committed to the public repository unless their redistribution is separately justified and permitted.

## Quick start

```bash
pip install -r requirements.txt
python src/preprocessing.py
python src/modeling.py
python src/advanced_analysis.py
```

For a lightweight sequential run:

```bash
python src/run_all.py
```

## Reproducibility note

The repository deliberately distinguishes **historical manuscript-reported results** from results regenerated under the currently installed software environment.

The original notebook reports an XGBoost test accuracy of `0.9866` and ROC-AUC of `0.9984`. A rerun under a newer XGBoost environment produced different XGBoost values. The other primary models were consistent with the reported results. Therefore, this repository does **not** claim exact reproduction of the historical XGBoost result until the original software environment is identified and pinned.

This is an intentional transparency measure, not a replacement of the manuscript's reported results.

See [`docs/reproducibility_audit.md`](docs/reproducibility_audit.md) for details.

## Leakage and interpretation safeguards

- `Past Attempt` is excluded from the primary model because it exhibited near-direct correspondence with the target and severe target leakage.
- `Illness` is excluded from the primary feature set because it is heterogeneous free text.
- High retrospective discrimination must not be interpreted as evidence of prospective or clinical validity.
- SHAP values describe model attribution and are not causal effects.
- Sensitivity analyses are included to examine dependence on Alcohol-related variables and encoded missingness categories.

## Citation

If you use this repository, please cite the accompanying manuscript and the underlying Mendeley Data record. See `CITATION.cff`.
