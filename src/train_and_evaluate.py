import argparse
import json
import os
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET_CANDIDATES = ["label", "attack", "attack_type", "class", "target", "Label"]
ID_CANDIDATES = ["id", "flow_id", "src_ip", "dst_ip", "src_port", "dst_port", "timestamp", "ts"]


def find_target_column(df: pd.DataFrame):
    for candidate in TARGET_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "Could not find a target column. Supported names: " + ", ".join(TARGET_CANDIDATES)
    )


def find_id_columns(df: pd.DataFrame):
    id_cols = []
    for col in df.columns:
        low = col.lower()
        if low in ID_CANDIDATES or "id" in low:
            id_cols.append(col)
    return id_cols


def clean_dataframe(df: pd.DataFrame):
    df = df.copy()
    df = df.dropna(axis=1, how="all")

    target_col = find_target_column(df)
    id_cols = find_id_columns(df)
    drop_cols = set(id_cols) | {target_col}
    features_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    for col in features_df.columns:
        if features_df[col].dtype == object:
            features_df[col] = features_df[col].astype(str)

    features_df = features_df.replace(["nan", "NaN", "None", "none", "null", "NULL"], pd.NA)
    return features_df, target_col


def build_model_pipeline(X):
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    transformers = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")),
        ]
    )
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(y_test, y_pred, digits=4, zero_division=0),
    }
    return metrics


def run_training(data_path: str, output_dir: str):
    df = pd.read_csv(data_path)
    X, target_col = clean_dataframe(df)
    y = df[target_col]

    if X.shape[1] == 0:
        raise ValueError("No feature columns remain after cleaning. Check the dataset schema.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = build_model_pipeline(X_train)
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "benchmark_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    comparison = {
        "dataset": "CICIoT2023",
        "paper_reference": "CICIoT2023: A Real-time Dataset and Benchmark for Large-Scale Attacks in IoT Environment",
        "reported_metrics": {
            "accuracy": "> 99%",
            "f1_score": "~0.99",
            "notes": "Random Forest and LightGBM are reported as top-performing models on this dataset."
        },
        "current_implementation": {
            "accuracy": metrics["accuracy"],
            "weighted_f1": metrics["f1_weighted"],
            "macro_f1": metrics["f1_macro"],
            "precision_weighted": metrics["precision_weighted"],
            "recall_weighted": metrics["recall_weighted"],
        },
    }

    with open(output_path / "benchmark_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    with open(output_path / "benchmark_summary.md", "w", encoding="utf-8") as f:
        f.write("# CICIoT2023 Benchmark Summary\n\n")
        f.write("## Reported benchmark from the paper\n")
        f.write("- Accuracy: > 99%\n")
        f.write("- F1-score: approximately 0.99 for leading models such as Random Forest\n")
        f.write("- The paper confirms strong detection performance across multiple IoT attack categories.\n\n")
        f.write("## Current implementation\n")
        f.write(f"- Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"- Weighted Precision: {metrics['precision_weighted']:.4f}\n")
        f.write(f"- Weighted Recall: {metrics['recall_weighted']:.4f}\n")
        f.write(f"- Weighted F1: {metrics['f1_weighted']:.4f}\n")
        f.write(f"- Macro F1: {metrics['f1_macro']:.4f}\n\n")
        f.write("## Notes\n")
        f.write("The benchmark values are approximate because the exact train/test split and preprocessing strategy can vary by implementation. The comparison is intended to match the published IQ of the CICIoT2023 paper while keeping the pipeline reproducible and easy to extend.\n")

    print(json.dumps({"dataset": "CICIoT2023", "metrics": metrics}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a Random Forest IoT intrusion detection model on CICIoT2023.")
    parser.add_argument("--data_path", required=True, help="Path to the CICIoT2023 CSV file.")
    parser.add_argument("--output_dir", default="results", help="Directory for metrics output.")
    args = parser.parse_args()

    run_training(args.data_path, args.output_dir)
