# GitHub Release Checklist

## Repository
- [ ] Repository visibility: Public
- [ ] Repository name: `suicide-attempt-ml-xai`
- [ ] Default branch: `main`
- [ ] README visible on landing page
- [ ] LICENSE present
- [ ] CITATION.cff present
- [ ] `.gitignore` present

## Data protection
- [ ] Do not upload `original_dataset.csv`
- [ ] Do not upload `cleaned_reproducible_dataset.csv`
- [ ] Do not upload `model_ready_reproducible_dataset.csv`
- [ ] Do not upload API keys, credentials, or private files

## Reproducibility
- [ ] Obtain the official Mendeley Data version 2 dataset
- [ ] Save it locally as `original_dataset.csv`
- [ ] Run `python src/preprocessing.py`
- [ ] Run `python src/modeling.py`
- [ ] Run `python src/advanced_analysis.py`
- [ ] Record Python and package versions
- [ ] Check `docs/reproducibility_audit.md`

## Publication
- [ ] Add the final GitHub URL to the manuscript
- [ ] Keep the distinction between historical manuscript results and current-environment results
- [ ] Do not describe the XGBoost result as exactly reproduced until the original XGBoost environment is identified
- [ ] Add an archival DOI (e.g., Zenodo) after the public repository is finalized, if desired
