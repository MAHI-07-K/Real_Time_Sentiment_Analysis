"""Dataset discovery, text cleaning, and leakage-safe stratified splitting."""
import re
from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PROCESSED_DIR, RANDOM_STATE, RAW_DIR

NEGATIONS = {"no", "not", "never", "nor", "none", "cannot"}
TEXT_CANDIDATES = ("text", "tweet", "comment", "content", "post", "body", "message")
LABEL_CANDIDATES = ("label", "sentiment", "category", "target", "class")


def clean_text(value: object) -> str:
    text = str(value or "").lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _column(columns: list[str], candidates: tuple[str, ...], kind: str) -> str:
    lowered = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    for normalized, original in lowered.items():
        if any(candidate in normalized for candidate in candidates):
            return original
    raise ValueError(f"Could not find a {kind} column. Available columns: {columns}")


def find_datasets(raw_dir: Path = RAW_DIR) -> list[Path]:
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV found in {raw_dir}. Run `python download_data.py` or copy the Kaggle CSV there."
        )
    return files


def prepare(csv_paths: list[Path] | None = None) -> dict:
    csv_paths = csv_paths or find_datasets()
    frames, schemas = [], []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        text_col = _column(list(df.columns), TEXT_CANDIDATES, "text")
        label_col = _column(list(df.columns), LABEL_CANDIDATES, "label")
        frames.append(pd.DataFrame({"text": df[text_col], "label": df[label_col]}))
        schemas.append({"file": csv_path.name, "text_column": text_col, "label_column": label_col})
    prepared = pd.concat(frames, ignore_index=True)
    prepared["clean_text"] = prepared["text"].map(clean_text)
    prepared = prepared.dropna(subset=["label"]).query("clean_text != ''").drop_duplicates("clean_text")
    prepared["label"] = prepared["label"].astype(str).str.strip()
    if prepared["label"].nunique() < 2:
        raise ValueError("At least two sentiment classes are required.")
    train, remaining = train_test_split(prepared, test_size=0.30, stratify=prepared["label"], random_state=RANDOM_STATE)
    val, test = train_test_split(remaining, test_size=0.50, stratify=remaining["label"], random_state=RANDOM_STATE)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("val", val), ("test", test)):
        split.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
    return {"sources": [str(path) for path in csv_paths], "schemas": schemas,
            "rows": len(prepared), "classes": prepared["label"].value_counts().to_dict()}


if __name__ == "__main__":
    try:
        print(prepare())
    except Exception as exc:
        print(f"Preparation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)

