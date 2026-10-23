"""Train all base classifiers and calculate validation-derived soft-voting weights."""
import json
import joblib
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.config import MODELS_DIR
from src.features import build_features


def main():
    train, val, _, X_train, X_val, _ = build_features()
    models = {
        "naive_bayes": MultinomialNB(),
        "svm": CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=5_000), cv=3),
        "lightgbm": LGBMClassifier(n_estimators=200, num_leaves=31, random_state=42, n_jobs=-1, verbosity=-1),
    }
    scores = {}
    for name, model in models.items():
        model.fit(X_train, train["label"])
        joblib.dump(model, MODELS_DIR / f"{name}.pkl")
        scores[name] = f1_score(val["label"], model.predict(X_val), average="macro")
    total = sum(scores.values())
    with open(MODELS_DIR / "ensemble_weights.json", "w", encoding="utf-8") as file:
        json.dump({name: score / total for name, score in scores.items()}, file, indent=2)
    print({"validation_macro_f1": scores, "weights": {name: score / total for name, score in scores.items()}})


if __name__ == "__main__":
    main()

