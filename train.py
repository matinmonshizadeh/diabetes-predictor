"""Train the from-scratch logistic regression and compare it with scikit-learn.

Usage:
    python train.py

Writes models/diabetes_lr.json (weights, bias and scaler statistics) and
models/metrics.json, and prints a comparison table.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from model import FEATURE_NAMES, DiabetesPredictor, LogisticRegression, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "diabetes.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "diabetes_lr.json"
METRICS_PATH = MODEL_DIR / "metrics.json"

SEED = 42
TEST_SIZE = 0.2


def evaluate(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_NAMES].to_numpy(dtype=float)
    y = df["Outcome"].to_numpy(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    # Standardise using training statistics only.
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Our model.
    ours = LogisticRegression(learning_rate=0.1, n_iterations=3000, seed=SEED)
    ours.fit(X_train_s, y_train)
    ours_metrics = evaluate(y_test, ours.predict_proba(X_test_s))

    # scikit-learn sanity check on the same standardised split.
    sk = SkLogisticRegression(max_iter=1000, random_state=SEED)
    sk.fit(X_train_s, y_train)
    sk_metrics = evaluate(y_test, sk.predict_proba(X_test_s)[:, 1])

    # Report.
    print(f"Train size: {len(y_train)}   Test size: {len(y_test)}")
    print(f"Final training loss (ours): {ours.loss_history[-1]:.4f}\n")
    print(f"{'Metric':<10}{'Ours':>10}{'sklearn':>10}")
    for key in ("accuracy", "precision", "recall", "roc_auc"):
        print(f"{key:<10}{ours_metrics[key]:>10.3f}{sk_metrics[key]:>10.3f}")
    print("\nWeights (standardised features):")
    for name, w_ours, w_sk in zip(FEATURE_NAMES, ours.weights, sk.coef_[0]):
        print(f"  {name:<26}{w_ours:>8.3f}{w_sk:>8.3f}")
    print(f"  {'bias':<26}{ours.bias:>8.3f}{sk.intercept_[0]:>8.3f}")

    # Save.
    MODEL_DIR.mkdir(exist_ok=True)
    DiabetesPredictor(scaler, ours).save(MODEL_PATH)
    metrics = {
        "seed": SEED,
        "test_size": TEST_SIZE,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "ours": ours_metrics,
        "sklearn": sk_metrics,
    }
    with open(METRICS_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(metrics, indent=2) + "\n")
    print(f"\nSaved {MODEL_PATH.relative_to(ROOT)} and {METRICS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
