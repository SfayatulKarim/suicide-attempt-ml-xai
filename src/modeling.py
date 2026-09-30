"""
Canonical model-evaluation pipeline.

This script generates CV and untouched-test metrics from the model-ready dataset.
It intentionally avoids hard-coded result values.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    make_scorer, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix
)
from xgboost import XGBClassifier

DATA_FILE = "model_ready_reproducible_dataset.csv"
TARGET = "Attempted?"

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) else 0.0

SCORING = {
    "accuracy": make_scorer(accuracy_score),
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "specificity": make_scorer(specificity_score),
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
}

def make_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=42))
        ]),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="rbf", C=1.0, probability=True, random_state=42
            ))
        ]),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric="logloss", n_jobs=-1
        ),
    }

def safe_xgb_columns(X):
    X = X.copy()
    X.columns = (
        X.columns.astype(str)
        .str.replace("[", "_", regex=False)
        .str.replace("]", "_", regex=False)
        .str.replace("<", "_", regex=False)
        .str.replace(">", "_", regex=False)
    )
    return X

def evaluate_test(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "Specificity": tn / (tn + fp),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, prob),
        "AP": average_precision_score(y_test, prob),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }, pred, prob

def main():
    data = pd.read_csv(DATA_FILE)
    X = data.drop(columns=[TARGET])
    y = data[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_rows = []
    test_rows = []

    for name, model in make_models().items():
        Xtr, Xte = X_train, X_test
        if name == "XGBoost":
            Xtr, Xte = safe_xgb_columns(Xtr), safe_xgb_columns(Xte)

        cv_out = cross_validate(
            model, Xtr, y_train, cv=cv, scoring=SCORING,
            return_train_score=False, n_jobs=-1
        )

        row = {"Model": name}
        for metric in SCORING:
            values = cv_out[f"test_{metric}"]
            row[f"CV {metric} mean"] = values.mean()
            row[f"CV {metric} sd"] = values.std()
        cv_rows.append(row)

        metrics, _, _ = evaluate_test(
            model, Xtr, y_train, Xte, y_test
        )
        test_rows.append({"Model": name, **metrics})

    cv_df = pd.DataFrame(cv_rows)
    test_df = pd.DataFrame(test_rows)

    Path("results").mkdir(exist_ok=True)
    cv_df.to_csv("results/cross_validation_results.csv", index=False)
    test_df.to_csv("results/test_set_results.csv", index=False)

    print("\\nCross-validation results:")
    print(cv_df.round(4).to_string(index=False))
    print("\\nUntouched test-set results:")
    print(test_df.round(4).to_string(index=False))

if __name__ == "__main__":
    main()
