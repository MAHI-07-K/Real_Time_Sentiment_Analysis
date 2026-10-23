"""Fit TF-IDF only on training data and persist it for inference."""
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import MODELS_DIR, PROCESSED_DIR


def load_splits():
    return tuple(pd.read_csv(PROCESSED_DIR / f"{name}.csv") for name in ("train", "val", "test"))


def build_features():
    train, val, test = load_splits()
    # Empty values written to CSV are read back as NaN by pandas; vectorizers
    # require strings even when a document contains no usable tokens.
    for split in (train, val, test):
        split["clean_text"] = split["clean_text"].fillna("").astype(str)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20_000, min_df=2, sublinear_tf=True)
    X_train = vectorizer.fit_transform(train["clean_text"])
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")
    return train, val, test, X_train, vectorizer.transform(val["clean_text"]), vectorizer.transform(test["clean_text"])

