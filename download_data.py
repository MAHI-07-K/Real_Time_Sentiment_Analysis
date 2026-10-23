"""Download the specified Kaggle dataset into data/raw (requires Kaggle authentication)."""
import shutil
from pathlib import Path
import kagglehub

from src.config import RAW_DIR

DATASET = "cosmos98/twitter-and-reddit-sentimental-analysis-dataset"

if __name__ == "__main__":
    downloaded = Path(kagglehub.dataset_download(DATASET))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for csv_file in downloaded.rglob("*.csv"):
        shutil.copy2(csv_file, RAW_DIR / csv_file.name)
        print(f"Copied {csv_file.name}")
