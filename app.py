"""PyQt5 front end for the trained diabetes model.

Usage:
    python train.py   # once, to create models/diabetes_lr.json
    python app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from model import DiabetesPredictor

MODEL_PATH = Path(__file__).resolve().parent / "models" / "diabetes_lr.json"

# (label, model feature name, min, max, example placeholder)
FIELDS = [
    ("Pregnancies", "Pregnancies", 0, 20, "e.g. 2"),
    ("Glucose (mg/dL)", "Glucose", 1, 300, "e.g. 120"),
    ("Blood pressure (mm Hg)", "BloodPressure", 1, 200, "e.g. 70"),
    ("Skin thickness (mm)", "SkinThickness", 0, 100, "e.g. 20"),
    ("Insulin (mu U/mL)", "Insulin", 0, 900, "e.g. 80"),
    ("BMI (kg/m^2)", "BMI", 10, 70, "e.g. 32.0"),
    ("Diabetes pedigree function", "DiabetesPedigreeFunction", 0, 3, "e.g. 0.47"),
    ("Age (years)", "Age", 1, 120, "e.g. 33"),
]


class DiabetesPredictionWindow(QMainWindow):
    def __init__(self, predictor: DiabetesPredictor) -> None:
        super().__init__()
        self.predictor = predictor
        self.inputs: dict[str, QLineEdit] = {}

        self.setWindowTitle("Diabetes Predictor")
        self.setMinimumWidth(380)

        form = QFormLayout()
        for label, key, _lo, _hi, placeholder in FIELDS:
            edit = QLineEdit()
            edit.setPlaceholderText(placeholder)
            edit.returnPressed.connect(self.predict)
            self.inputs[key] = edit
            form.addRow(f"{label}:", edit)

        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self.predict)

        self.result_label = QLabel("Enter the values above and press Predict.")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setWordWrap(True)
        self.result_label.setObjectName("result")

        disclaimer = QLabel("Educational demo, not medical advice.")
        disclaimer.setAlignment(Qt.AlignCenter)
        disclaimer.setObjectName("disclaimer")

        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(self.predict_button)
        layout.addWidget(self.result_label)
        layout.addWidget(disclaimer)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.setStyleSheet(
            """
            QLabel { font-size: 14px; }
            QLineEdit { font-size: 14px; padding: 6px; }
            QPushButton {
                font-size: 16px; padding: 10px;
                background-color: #4CAF50; color: white; border: none;
            }
            QPushButton:hover { background-color: #45a049; }
            QLabel#result { font-size: 16px; font-weight: bold; padding: 12px; }
            QLabel#disclaimer { font-size: 12px; color: #a00; }
            """
        )

    # ------------------------------------------------------------------
    def read_features(self) -> np.ndarray | None:
        """Parse and range-check every field. Returns None after showing an error."""
        values = []
        for label, key, lo, hi, _ in FIELDS:
            text = self.inputs[key].text().strip().replace(",", ".")
            if not text:
                self.show_error(f"Please enter a value for {label}.")
                return None
            try:
                value = float(text)
            except ValueError:
                self.show_error(f"{label} must be a number (you entered '{text}').")
                return None
            if not np.isfinite(value) or not (lo <= value <= hi):
                self.show_error(f"{label} must be between {lo} and {hi}.")
                return None
            values.append(value)
        return np.array(values, dtype=float)

    def predict(self) -> None:
        features = self.read_features()
        if features is None:
            return
        probability = float(self.predictor.predict_proba(features)[0])
        label = "likely diabetic" if probability >= 0.5 else "likely not diabetic"
        self.result_label.setText(
            f"Estimated probability of diabetes: {probability:.1%}\n({label} at a 50% threshold)"
        )

    def show_error(self, message: str) -> None:
        QMessageBox.warning(self, "Invalid input", message)


def main() -> int:
    app = QApplication(sys.argv)
    if not MODEL_PATH.exists():
        QMessageBox.critical(
            None,
            "Model not found",
            f"Could not find {MODEL_PATH}.\nRun 'python train.py' first to train and save the model.",
        )
        return 1
    predictor = DiabetesPredictor.load(MODEL_PATH)
    window = DiabetesPredictionWindow(predictor)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
