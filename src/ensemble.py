import json
import joblib
import numpy as np

from src.config import MODELS_DIR


class SentimentEnsemble:
    def __init__(self):
        self.vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
        self.models = {name: joblib.load(MODELS_DIR / f"{name}.pkl") for name in ("naive_bayes", "svm", "lightgbm")}
        with open(MODELS_DIR / "ensemble_weights.json", encoding="utf-8") as file:
            self.weights = json.load(file)
        self.classes_ = self.models["naive_bayes"].classes_
        self._tokenizer = self.vectorizer.build_tokenizer()

    def compute_vocab_coverage(self, raw_or_cleaned_text: str) -> float:
        """Compute the ratio of tokens present in the TF-IDF vocabulary."""
        tokens = self._tokenizer(raw_or_cleaned_text)
        if not tokens:
            return 0.0
        in_vocab = sum(1 for t in tokens if t in self.vectorizer.vocabulary_)
        return in_vocab / len(tokens)

    def predict(self, texts: list[str]):
        from src.preprocessing import clean_text
        matrix = self.vectorizer.transform([clean_text(text) for text in texts])
        probabilities = sum(self.weights[name] * model.predict_proba(matrix) for name, model in self.models.items())
        indexes = probabilities.argmax(axis=1)
        return self.classes_[indexes], probabilities

    def predict_detailed(self, texts: list[str]):
        """Vectorized prediction returning labels, probabilities, and vocab coverage for each text."""
        from src.preprocessing import clean_text
        cleaned_texts = [clean_text(text) for text in texts]
        matrix = self.vectorizer.transform(cleaned_texts)
        probabilities = sum(self.weights[name] * model.predict_proba(matrix) for name, model in self.models.items())
        indexes = probabilities.argmax(axis=1)
        labels = self.classes_[indexes]
        coverages = [self.compute_vocab_coverage(cleaned) for cleaned in cleaned_texts]
        return labels, probabilities, coverages


