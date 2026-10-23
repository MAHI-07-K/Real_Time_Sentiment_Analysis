"""Report test metrics, confusion matrices, and ensemble ablation results."""
import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.config import MODELS_DIR
from src.ensemble import SentimentEnsemble
from src.features import build_features


def main():
    _, _, test, _, _, X_test = build_features()
    y_true = test["label"]
    report = {}
    for name in ("naive_bayes", "svm", "lightgbm"):
        predictions = joblib.load(MODELS_DIR / f"{name}.pkl").predict(X_test)
        report[name] = {"accuracy": accuracy_score(y_true, predictions), "macro_f1": f1_score(y_true, predictions, average="macro")}
    ensemble = SentimentEnsemble()
    predictions, probabilities = ensemble.predict(test["text"].tolist())
    report["ensemble"] = {"accuracy": accuracy_score(y_true, predictions), "macro_f1": f1_score(y_true, predictions, average="macro"),
                          "classification_report": classification_report(y_true, predictions, output_dict=True),
                          "confusion_matrix": confusion_matrix(y_true, predictions, labels=ensemble.classes_).tolist(),
                          "labels": ensemble.classes_.tolist()}
    with open(MODELS_DIR / "evaluation.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, default=float)
    print(pd.DataFrame(report).T[["accuracy", "macro_f1"]])


if __name__ == "__main__":
    main()

