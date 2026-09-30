"""Canonical advanced analysis pipeline for the suicide-attempt ML study.

Generates test-set ROC/PR curves, calibration, bootstrap 95% CIs,
XGBoost SHAP explanations/interactions, and sensitivity analyses.

Important: outputs are generated from the current installed dependency versions.
Historical manuscript values are not hard-coded here.
"""
from pathlib import Path
import json
import platform
import sys
import warnings
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss, roc_curve, precision_recall_curve
)
from xgboost import XGBClassifier
import shap

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "model_ready_reproducible_dataset.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
TARGET = "Attempted?"
RANDOM_STATE = 42
THRESHOLD = 0.50
BOOTSTRAPS = 2000


def make_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=42)),
        ]),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", C=1.0, probability=True, random_state=42)),
        ]),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric="logloss", n_jobs=-1,
        ),
    }


def safe_xgb_columns(X):
    X = X.copy()
    X.columns = (X.columns.astype(str)
                 .str.replace("[", "_", regex=False)
                 .str.replace("]", "_", regex=False)
                 .str.replace("<", "_", regex=False)
                 .str.replace(">", "_", regex=False))
    return X


def evaluate(y_true, pred, prob):
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "F1": f1_score(y_true, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, prob),
        "AP": average_precision_score(y_true, prob),
        "Brier": brier_score_loss(y_true, prob),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


def calibration_slope_intercept(y, prob):
    eps = 1e-6
    p = np.clip(np.asarray(prob), eps, 1 - eps)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    cal.fit(logit, np.asarray(y))
    return float(cal.intercept_[0]), float(cal.coef_[0, 0])


def bootstrap_ci(y, pred, prob, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    y = np.asarray(y); pred = np.asarray(pred); prob = np.asarray(prob)
    n = len(y)
    rows = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yy, pp, pr = y[idx], pred[idx], prob[idx]
        # AUC is undefined if a bootstrap sample has one class.
        if len(np.unique(yy)) < 2:
            continue
        rows.append({
            "Accuracy": accuracy_score(yy, pp),
            "Precision": precision_score(yy, pp, zero_division=0),
            "Recall": recall_score(yy, pp, zero_division=0),
            "F1": f1_score(yy, pp, zero_division=0),
            "ROC-AUC": roc_auc_score(yy, pr),
        })
    b = pd.DataFrame(rows)
    out = []
    for metric in b.columns:
        out.append({
            "Metric": metric,
            "Point": float(metric_values := {
                "Accuracy": accuracy_score(y, pred),
                "Precision": precision_score(y, pred, zero_division=0),
                "Recall": recall_score(y, pred, zero_division=0),
                "F1": f1_score(y, pred, zero_division=0),
                "ROC-AUC": roc_auc_score(y, prob),
            }[metric]),
            "CI95_Lower": float(np.percentile(b[metric], 2.5)),
            "CI95_Upper": float(np.percentile(b[metric], 97.5)),
            "N_Bootstrap": len(b),
        })
    return pd.DataFrame(out)


def train_predict(X_train, X_test, y_train, y_test):
    fitted = {}
    rows = []
    predictions = {}
    probabilities = {}
    for name, model in make_models().items():
        Xtr, Xte = X_train, X_test
        if name == "XGBoost":
            Xtr, Xte = safe_xgb_columns(Xtr), safe_xgb_columns(Xte)
        model.fit(Xtr, y_train)
        pred = model.predict(Xte)
        prob = model.predict_proba(Xte)[:, 1]
        fitted[name] = model
        predictions[name] = pred
        probabilities[name] = prob
        rows.append({"Model": name, **evaluate(y_test, pred, prob)})
    return fitted, predictions, probabilities, pd.DataFrame(rows)


def plot_roc_pr(y, probs):
    FIGURES.mkdir(exist_ok=True)
    plt.figure(figsize=(7, 5))
    for name, prob in probs.items():
        fpr, tpr, _ = roc_curve(y, prob)
        auc = roc_auc_score(y, prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Untouched Test Set"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIGURES / "roc_curves_test.png", dpi=300); plt.close()

    plt.figure(figsize=(7, 5))
    baseline = np.mean(y)
    for name, prob in probs.items():
        precision, recall, _ = precision_recall_curve(y, prob)
        ap = average_precision_score(y, prob)
        plt.plot(recall, precision, label=f"{name} (AP={ap:.4f})")
    plt.axhline(baseline, linestyle="--", label=f"Prevalence={baseline:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall Curves — Untouched Test Set"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIGURES / "pr_curves_test.png", dpi=300); plt.close()


def calibration_analysis(y, probs):
    rows = []
    plt.figure(figsize=(7, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    for name, prob in probs.items():
        from sklearn.calibration import calibration_curve
        frac, mean = calibration_curve(y, prob, n_bins=5, strategy="quantile")
        intercept, slope = calibration_slope_intercept(y, prob)
        rows.append({"Model": name, "Brier": brier_score_loss(y, prob),
                     "Calibration_Intercept": intercept, "Calibration_Slope": slope})
        plt.plot(mean, frac, marker="o", label=name)
    plt.xlabel("Mean predicted probability"); plt.ylabel("Observed frequency")
    plt.title("Calibration — 5 Quantile Bins, Test Set"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIGURES / "calibration_test.png", dpi=300); plt.close()
    return pd.DataFrame(rows)


def sensitivity_analysis(X, y, X_train, X_test, y_train, y_test):
    analyses = {"Original": X.columns.tolist()}
    analyses["Alcohol Removed"] = [c for c in X.columns if not c.startswith("Alcohol__")]
    analyses["Missingness Removed"] = [c for c in X.columns if "__Unknown/Not reported" not in c]
    rows = []
    for label, cols in analyses.items():
        Xt, Xv = X_train[cols], X_test[cols]
        for name, model in make_models().items():
            if name == "XGBoost":
                Xt2, Xv2 = safe_xgb_columns(Xt), safe_xgb_columns(Xv)
            else:
                Xt2, Xv2 = Xt, Xv
            model.fit(Xt2, y_train)
            pred = model.predict(Xv2); prob = model.predict_proba(Xv2)[:, 1]
            m = evaluate(y_test, pred, prob)
            rows.append({"Analysis": label, "Model": name,
                         "Features": len(cols), "Accuracy": m["Accuracy"],
                         "F1": m["F1"], "ROC-AUC": m["ROC-AUC"]})
    return pd.DataFrame(rows)


def shap_analysis(xgb_model, X_test, compute_interactions=True):
    Xx = safe_xgb_columns(X_test)
    explainer = shap.TreeExplainer(xgb_model)
    sv = explainer.shap_values(Xx)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    names = list(X_test.columns)
    global_df = pd.DataFrame({"Feature": names, "MeanAbsSHAP": np.abs(sv).mean(axis=0)})
    global_df = global_df.sort_values("MeanAbsSHAP", ascending=False)
    global_df.to_csv(RESULTS / "shap_global_feature_importance.csv", index=False)

    top = global_df.head(15).copy()
    plt.figure(figsize=(8, 6))
    plt.barh(top["Feature"][::-1], top["MeanAbsSHAP"][::-1])
    plt.xlabel("Mean |SHAP value|"); plt.title("XGBoost Global SHAP Attribution")
    plt.tight_layout(); plt.savefig(FIGURES / "shap_global_importance.png", dpi=300); plt.close()

    grouped = {}
    for col, value in zip(names, np.abs(sv).mean(axis=0)):
        group = col.split("__", 1)[0]
        grouped[group] = grouped.get(group, 0.0) + float(value)
    grouped_df = pd.DataFrame(sorted(grouped.items(), key=lambda z: z[1], reverse=True),
                              columns=["OriginalFeature", "GroupedMeanAbsSHAP"])
    grouped_df.to_csv(RESULTS / "shap_grouped_original_features.csv", index=False)

    # Dependence plots for top five encoded features.
    for i, feature in enumerate(top["Feature"].head(5), start=1):
        idx = names.index(feature)
        plt.figure(figsize=(6, 4))
        plt.scatter(X_test[feature], sv[:, idx], s=16, alpha=0.7)
        plt.xlabel(feature); plt.ylabel("SHAP value")
        plt.title(f"SHAP Dependence — {feature}")
        plt.tight_layout(); plt.savefig(FIGURES / f"shap_dependence_{i}.png", dpi=300); plt.close()

    if not compute_interactions:
        return global_df, grouped_df, pd.DataFrame(columns=["Feature_A", "Feature_B", "MeanAbsInteraction"])

    interactions = explainer.shap_interaction_values(Xx)
    if isinstance(interactions, list):
        interactions = interactions[1]
    interactions = np.asarray(interactions)
    mean_abs = np.abs(interactions).mean(axis=0)
    pair_rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pair_rows.append((names[i], names[j], mean_abs[i, j]))
    pair_df = pd.DataFrame(pair_rows, columns=["Feature_A", "Feature_B", "MeanAbsInteraction"])
    pair_df = pair_df.sort_values("MeanAbsInteraction", ascending=False)
    pair_df.to_csv(RESULTS / "shap_interactions_encoded.csv", index=False)

    # Aggregate encoded interactions to original feature groups.
    gnames = [n.split("__", 1)[0] for n in names]
    agg = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = gnames[i], gnames[j]
            if a == b:
                continue
            key = tuple(sorted((a, b)))
            agg[key] = agg.get(key, 0.0) + float(mean_abs[i, j])
    agg_df = pd.DataFrame([{"Feature_A": k[0], "Feature_B": k[1], "MeanAbsInteraction": v}
                           for k, v in agg.items()]).sort_values("MeanAbsInteraction", ascending=False)
    agg_df.to_csv(RESULTS / "shap_interactions_grouped.csv", index=False)
    return global_df, grouped_df, agg_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-interactions", action="store_true", help="Skip full SHAP interaction tensor; useful for fast reproducibility runs.")
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAPS, help="Number of bootstrap resamples (default: 2000).")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
    data = pd.read_csv(DATA_FILE)
    X = data.drop(columns=[TARGET]); y = data[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    fitted, preds, probs, test_df = train_predict(X_train, X_test, y_train, y_test)
    test_df.to_csv(RESULTS / "advanced_test_set_results_current_environment.csv", index=False)

    bootstrap_rows = []
    for name in probs:
        b = bootstrap_ci(y_test, preds[name], probs[name], args.bootstrap_resamples, RANDOM_STATE)
        b.insert(0, "Model", name)
        bootstrap_rows.append(b)
    pd.concat(bootstrap_rows, ignore_index=True).to_csv(RESULTS / "bootstrap_95ci_current_environment.csv", index=False)

    cal = calibration_analysis(y_test, probs)
    cal.to_csv(RESULTS / "calibration_current_environment.csv", index=False)

    plot_roc_pr(y_test, probs)
    sens = sensitivity_analysis(X, y, X_train, X_test, y_train, y_test)
    sens.to_csv(RESULTS / "sensitivity_current_environment.csv", index=False)

    xgb = fitted["XGBoost"]
    shap_analysis(xgb, X_test, compute_interactions=not args.skip_interactions)

    versions = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": __import__("sklearn").__version__,
        "xgboost": __import__("xgboost").__version__,
        "shap": shap.__version__,
        "random_state": RANDOM_STATE,
        "test_size": 0.20,
        "threshold": THRESHOLD,
        "bootstrap_resamples": args.bootstrap_resamples,
    }
    (RESULTS / "environment_versions.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")
    print("Advanced analysis completed.")
    print(test_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
