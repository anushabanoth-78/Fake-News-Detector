#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

# -----------------------------
# Constants
# -----------------------------
LABELS: Final[Tuple[str, str]] = ("REAL", "FAKE")


# -----------------------------
# Helper functions
# -----------------------------
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_any(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def build_combined_text(df: pd.DataFrame) -> pd.Series:
    """
    Safely combine title and text columns.
    Works even if 'title' column does not exist.
    """
    title = df["title"].fillna("") if "title" in df.columns else ""
    text = df["text"].fillna("") if "text" in df.columns else ""
    return title + " " + text


# -----------------------------
# Plot functions
# -----------------------------
def plot_confusion_matrix(cm: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(LABELS)
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_curve(x, y, out: Path, title, xlabel, ylabel) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Fake News Detection Training Script")
    parser.add_argument("--real", required=True, help="Path to REAL news CSV")
    parser.add_argument("--fake", required=True, help="Path to FAKE news CSV")
    parser.add_argument("--outdir", default="outputs", help="Output directory")
    args = parser.parse_args()

    outdir = ensure_dir(Path(args.outdir))
    charts = ensure_dir(outdir / "charts")

    # Load datasets
    real_df = read_csv_any(Path(args.real))
    fake_df = read_csv_any(Path(args.fake))

    # Build combined text safely
    real_df["combined_text"] = build_combined_text(real_df)
    fake_df["combined_text"] = build_combined_text(fake_df)

    # Features and labels
    X = pd.concat(
        [real_df["combined_text"], fake_df["combined_text"]],
        ignore_index=True,
    )
    y = np.array([0] * len(real_df) + [1] * len(fake_df))  # 0=REAL, 1=FAKE

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 3),
                    max_df=0.8,
                    min_df=3,
                    max_features=20000,
                ),
            ),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_macro")
    print(f"CV F1-score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    # Save metrics
    metrics = {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "cv_f1_mean": cv_scores.mean(),
        "cv_f1_std": cv_scores.std(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=LABELS, output_dict=True
        ),
    }

    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Plots
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, charts / "confusion_matrix.png")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plot_curve(fpr, tpr, charts / "roc_curve.png", "ROC Curve", "FPR", "TPR")

    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    plot_curve(rec, prec, charts / "pr_curve.png", "Precision-Recall Curve", "Recall", "Precision")

    # Save model
    joblib.dump(pipeline, outdir / "pipeline.joblib")
    joblib.dump(pipeline.named_steps["tfidf"], outdir / "vectorizer.joblib")
    joblib.dump(pipeline.named_steps["clf"], outdir / "model.joblib")

    print("Training completed successfully.")
    print("Outputs saved in:", outdir.resolve())


if __name__ == "__main__":
    main()
