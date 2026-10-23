
# Sentiment Analysis Ensemble

Naive Bayes, calibrated linear SVM, and LightGBM combined through validation-F1-weighted soft voting. The project exposes the trained model via FastAPI and a small Streamlit dashboard.

## Setup

```powershell
cd C:\Users\malli\Downloads\Python\.venv\5sem\ml_project
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset

This project uses [the supplied Kaggle dataset](https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset).

Authenticate Kaggle/kagglehub, then run `python download_data.py`. Alternatively, download its CSV manually and place it in `data/raw/`. The preprocessing step detects common text and sentiment-label column names automatically.

## Run the pipeline

```powershell
python -m src.preprocessing
python -m src.train
python -m src.evaluate
uvicorn api.main:app --reload --port 8000
streamlit run dashboard/app.py
```

Use `GET /health` to check model readiness, and `POST /predict` with `{ "text": "..." }` for a prediction. Copy `.env.example` to `.env` before using Reddit ingestion.

## Notes

- TF-IDF is fitted only to the training split (70/15/15 stratified split), preventing validation/test leakage.
- Model weights are proportional to validation macro-F1, which is more appropriate than accuracy for imbalanced labels.
