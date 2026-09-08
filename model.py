"""Logistic regression from scratch with NumPy.

Only NumPy is used here. The model is trained with full-batch gradient
descent on the binary cross-entropy loss, and it standardises its inputs
(zero mean, unit variance) using statistics computed on the training set.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


class StandardScaler:
    """Per-feature standardisation fitted on the training split."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        scale[scale == 0] = 1.0  # avoid division by zero on constant columns
        self.scale_ = scale
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler has not been fitted")
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class LogisticRegression:
    """Binary logistic regression trained by gradient descent.

    Parameters
    ----------
    learning_rate:
        Step size for gradient descent.
    n_iterations:
        Number of full-batch gradient steps.
    l2:
        L2 penalty strength (0 disables regularisation).
    seed:
        Seed for the small random initialisation of the weights.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        n_iterations: int = 3000,
        l2: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.l2 = l2
        self.seed = seed
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []

    # ------------------------------------------------------------------ core
    def _linear(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias

    @staticmethod
    def _cross_entropy(y: np.ndarray, p: np.ndarray) -> float:
        eps = 1e-12
        p = np.clip(p, eps, 1 - eps)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """Fit on already-standardised features X (n_samples, n_features)."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n_samples, n_features = X.shape

        rng = np.random.default_rng(self.seed)
        self.weights = rng.normal(0.0, 0.01, size=n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iterations):
            p = sigmoid(self._linear(X))
            error = p - y  # shape (n_samples,)

            # Gradient of the mean cross-entropy: one entry per feature.
            dw = X.T @ error / n_samples + self.l2 * self.weights
            db = error.mean()

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            self.loss_history.append(self._cross_entropy(y, p))

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability of the positive class for each row of X."""
        if self.weights is None:
            raise RuntimeError("Model has not been fitted")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return sigmoid(self._linear(X))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Hard 0/1 labels."""
        return (self.predict_proba(X) >= threshold).astype(int)

    # --------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {
            "weights": self.weights.tolist(),
            "bias": float(self.bias),
            "learning_rate": self.learning_rate,
            "n_iterations": self.n_iterations,
            "l2": self.l2,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LogisticRegression":
        model = cls(
            learning_rate=d["learning_rate"],
            n_iterations=d["n_iterations"],
            l2=d.get("l2", 0.0),
            seed=d.get("seed", 42),
        )
        model.weights = np.asarray(d["weights"], dtype=float)
        model.bias = float(d["bias"])
        return model


class DiabetesPredictor:
    """Scaler + model bundle, the object the GUI loads."""

    def __init__(self, scaler: StandardScaler, model: LogisticRegression) -> None:
        self.scaler = scaler
        self.model = model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(np.atleast_2d(X)))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path: str | Path) -> None:
        payload = {
            "feature_names": FEATURE_NAMES,
            "scaler": {
                "mean": self.scaler.mean_.tolist(),
                "scale": self.scaler.scale_.tolist(),
            },
            "model": self.model.to_dict(),
        }
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "DiabetesPredictor":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(payload["scaler"]["mean"], dtype=float)
        scaler.scale_ = np.asarray(payload["scaler"]["scale"], dtype=float)
        model = LogisticRegression.from_dict(payload["model"])
        return cls(scaler, model)
