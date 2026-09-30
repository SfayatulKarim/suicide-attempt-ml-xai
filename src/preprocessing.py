"""
Reproducible preprocessing for the suicide-attempt retrospective-classification study.

Input:
    original_dataset.csv

Outputs:
    cleaned_reproducible_dataset.csv
    model_ready_reproducible_dataset.csv

The transformations mirror the documented cleaning decisions used in the manuscript.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd

RAW_FILE = "original_dataset.csv"
CLEANED_FILE = "cleaned_reproducible_dataset.csv"
MODEL_READY_FILE = "model_ready_reproducible_dataset.csv"
TARGET = "Attempted?"

NULL_TOKENS = {"", "null", "nan", "none", "na", "n/a"}

BINARY_COLUMNS = ["Anger", "Sleep Problem", "Isolation", "Humiliated"]

def normalize_text(value):
    if pd.isna(value):
        return pd.NA
    value = str(value).strip()
    if value.lower() in NULL_TOKENS:
        return pd.NA
    return value

def age_to_group(value):
    if pd.isna(value):
        return "Unknown/Not reported"
    s = str(value).strip()
    if not s or s.lower() in NULL_TOKENS:
        return "Unknown/Not reported"

    # Preserve explicit source ranges.
    if re.fullmatch(r"\d+\s*-\s*\d+", s):
        lo, hi = [int(x) for x in re.findall(r"\d+", s)]
        # Preserve the source-provided age-band labels.
        return f"{lo}-{hi}"

    try:
        age = int(float(s))
    except ValueError:
        return "Unknown/Not reported"

    # Canonical study age groups. Numeric ages are mapped to the same
    # decade bands used by the source-provided ranges.
    if age < 15:
        return "<15"
    if age <= 24:
        return "15-24"
    if age <= 34:
        return "25-34"
    if age <= 44:
        return "35-44"
    if age <= 54:
        return "45-54"
    if age <= 64:
        return "55-64"
    if age <= 74:
        return "65-74"
    if age <= 84:
        return "75-84"
    if age <= 94:
        return "85-94"
    return "95+"

def standardize(df):
    df = df.copy()

    # Whitespace/null-token normalization.
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].map(normalize_text)

    # Target.
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[df[TARGET].isin([0, 1])].copy()
    df[TARGET] = df[TARGET].astype(int)

    # Gender.
    if "Gender" in df:
        df["Gender"] = df["Gender"].replace({
            "M": "Male", "Male": "Male",
            "F": "Female", "Female": "Female"
        })

    # Religion.
    if "Religion" in df:
        df["Religion"] = df["Religion"].map(
            lambda x: x.title() if pd.notna(x) else x
        )

    # Civil status.
    if "Civil Status" in df:
        df["Civil Status"] = df["Civil Status"].replace({
            "Single": "Unmarried",
            "Divourced": "Divorced",
            "Divorce": "Divorced",
            "Widow": "Widowed"
        })

    # Education: merge documented duplicate spellings.
    if "Education Level" in df:
        education_map = {
            "Secondary Level(class 6 -10)": "Secondary Level (Class 6-10)",
            "Secondary Level (class 6 -10)": "Secondary Level (Class 6-10)",
            "Higher Secondary Level(class 11-12)": "Higher Secondary Level (Class 11-12)",
            "Higher Secondary Level (class 11-12)": "Higher Secondary Level (Class 11-12)",
            "Higher Secondary Level (Class 11-12 )": "Higher Secondary Level (Class 11-12)"
        }
        df["Education Level"] = df["Education Level"].replace(education_map)

    # Psychological session.
    if "Psych Session" in df:
        df["Psych Session"] = df["Psych Session"].replace({
            0: "No", 1: "Yes", "0": "No", "1": "Yes",
            "No": "No", "Yes": "Yes"
        })

    # Past Attempt: standardized for audit, never included in model features.
    if "Past Attempt" in df:
        df["Past Attempt"] = df["Past Attempt"].replace({
            0: "No", 1: "Yes", "0": "No", "1": "Yes"
        })

    # Binary behavioral fields.
    for col in BINARY_COLUMNS:
        if col in df:
            df[col] = df[col].replace({
                0: "No", 1: "Yes", "0": "No", "1": "Yes",
                "Not Sure": "Not sure"
            })

    # Disorder typo.
    if "Disorder" in df:
        df["Disorder"] = df["Disorder"].replace({
            "Bipolar Disorer": "Bipolar Disorder"
        })

    # Alcohol category labels.
    if "Alcohol" in df:
        df["Alcohol"] = df["Alcohol"].replace({
            "medium": "Medium",
            "moderate": "Moderate",
            "frequent": "Frequent",
            "no": "No"
        })

    # Sad/Weary.
    if "Sad/ Weary" in df:
        df["Sad/ Weary"] = df["Sad/ Weary"].replace({
            0.0: "0.0", 1.0: "1.0",
            0: "0.0", 1: "1.0"
        })

    # Age groups.
    df["Age_Group"] = df["Age"].map(age_to_group)

    # Occupation normalization.
    if "Occupation" in df:
        occ = df["Occupation"].astype("string").str.strip()
        occ = occ.replace({
            "Student": "Student",
            "Students": "Student",
            "students": "Student",
            "student": "Student",
            "House Wife": "Housewife",
            "House wife": "Housewife",
            "housewife": "Housewife"
        })
        df["Occupation_Standard"] = occ

        # Missing occupation is represented explicitly.
        df["Occupation_Standard"] = df["Occupation_Standard"].fillna(
            "Unknown/Not reported"
        )

        counts = df["Occupation_Standard"].value_counts()
        rare = counts[counts < 5].index
        df.loc[
            df["Occupation_Standard"].isin(rare),
            "Occupation_Standard"
        ] = "Other"

    # Categorical missing values: preserve missingness explicitly.
    categorical = [
        "Gender", "Religion", "Occupation", "Civil Status",
        "Education Level", "Psych Session", "Past Attempt",
        "Disorder", "Alcohol", "Anger", "Sleep Problem",
        "Isolation", "Humiliated", "Sad/ Weary"
    ]
    for col in categorical:
        if col in df:
            df[col] = df[col].fillna("Unknown/Not reported")

    # Remove exact duplicate records after standardization.
    df = df.drop_duplicates().reset_index(drop=True)

    return df

def make_model_ready(cleaned):
    feature_columns = [
        "Age_Group", "Gender", "Religion", "Occupation_Standard",
        "Civil Status", "Education Level", "Psych Session", "Disorder",
        "Alcohol", "Anger", "Sleep Problem", "Isolation", "Humiliated",
        "Sad/ Weary"
    ]

    model_df = cleaned[feature_columns + [TARGET]].copy()

    encoded = pd.get_dummies(
        model_df.drop(columns=[TARGET]),
        prefix_sep="__",
        dtype=int
    )

    encoded[TARGET] = model_df[TARGET].astype(int).values
    return encoded

def main():
    raw = pd.read_csv(RAW_FILE)
    cleaned = standardize(raw)

    # Preserve the cleaned audit file with an explicit row index, matching
    # the historical analysis artifact.
    cleaned.to_csv(CLEANED_FILE, index=True)

    model_ready = make_model_ready(cleaned)
    model_ready.to_csv(MODEL_READY_FILE, index=False)

    print("Raw shape:", raw.shape)
    print("Cleaned shape:", cleaned.shape)
    print("Model-ready shape:", model_ready.shape)
    print("Target distribution:")
    print(model_ready[TARGET].value_counts().sort_index())
    print("Past Attempt included in model:", "Past Attempt" in model_ready.columns)
    print("Illness included in model:", "Illness" in model_ready.columns)

if __name__ == "__main__":
    main()
