# Sentiment Analysis Ensemble - Project Details

## Overview

This project classifies short, user-generated text into sentiment categories:
- `-1`: Negative
- `0`: Neutral
- `1`: Positive

It is designed for Twitter, Reddit, YouTube, and live streaming comments, exposing predictions through both a REST API and a real-time Streamlit web dashboard.

The deployed baseline is a weighted soft-voting ensemble of three text classifiers:

- Multinomial Naive Bayes
- Calibrated linear Support Vector Machine (SVM)
- LightGBM classifier

Each model contributes class probabilities. Its ensemble weight is proportional to its validation macro-F1 score, so models that perform better across all sentiment classes have more influence on the final prediction.

## System flow

```text
Raw Live Comments (YouTube Live Chat / Comments, Reddit, Twitch)
    -> Buffer & Batcher (every 2s or 50 messages)
    -> FastAPI POST /predict_batch (single vectorized TF-IDF transform & predict_proba)
    -> SQLite Store (data/comments.db, deduplicated by source_comment_id, zero author info)
    -> Streamlit Dashboard (st.fragment run_every=2s polling GET /stats & GET /feed)
```

## Data processing

The preprocessing module automatically detects common text and label column names, allowing compatible CSV files to be used without renaming their columns first.

Text is normalized by:

- converting to lowercase;
- removing URLs and @mentions;
- converting hashtags to their word content;
- removing non-alphanumeric punctuation while retaining apostrophes;
- collapsing repeated whitespace;
- removing empty rows, missing labels, and duplicate cleaned texts.

The retained data is divided with stratified sampling into 70% training, 15% validation, and 15% test data. TF-IDF is fitted only on the training partition, preventing vocabulary leakage from validation or test examples.

## Features and models

Text is represented with a TF-IDF vectorizer using unigrams and bigrams. The vocabulary is limited to 20,000 features, ignores terms appearing in fewer than two documents, and uses sublinear term-frequency scaling.

The classifiers are trained on the same training features. Validation macro-F1 determines their soft-voting weights. The completed baseline artifacts are stored in `models/`:

- `tfidf_vectorizer.pkl`
- `naive_bayes.pkl`
- `svm.pkl`
- `lightgbm.pkl`
- `ensemble_weights.json`
- `evaluation.json`

## Real-Time Architecture & Privacy

1. **Storage (`src/store.py`)**: SQLite table `comments` with WAL mode enabled.
   - Columns: `id` (autoincrement), `source`, `source_comment_id` (UNIQUE), `text`, `label`, `confidence`, `p_neg`, `p_neu`, `p_pos`, `vocab_coverage`, `published_at`, `ingested_at`.
   - **Privacy Policy**: Author usernames, user handles, and account IDs are **strictly omitted** across all ingestion pathways to safeguard privacy.
2. **Batch Pipeline (`ingestion/buffer.py`)**: Flushes queued items to `/predict_batch` every 2 seconds or 50 messages, and persists predictions to SQLite.
3. **Ingestion Workers**:
   - `ingestion/youtube_live.py`: Resolves `activeLiveChatId` via `videos.list`. If live, polls `liveChatMessages.list` honoring `pollingIntervalMillis` and `nextPageToken`. If standard video or ended stream, falls back to polling `commentThreads.list`. Exponential backoff is applied for quota/network errors.
   - `ingestion/reddit_stream.py`: Streams Reddit comments via PRAW into the shared batch buffer.
   - `ingestion/twitch_chat.py`: Connects to live Twitch IRC channels using `twitchio`.
   - `ingestion/run_ingestion.py`: Unified CLI runner for all sources.

> [!WARNING]
> **Domain Limitation for Live Chat**: The baseline TF-IDF model was trained on conventional social media posts (Twitter and Reddit sentences). Live chat (YouTube Live and Twitch) contains high frequencies of internet slang, emoticons, character repetition, emotes, and non-standard tokens not present in the 20,000 TF-IDF training vocabulary. When token vocabulary coverage is below 50% (`vocab_coverage < 0.5`), the prediction relies on fewer known words or falls back toward model priors, flagged with a warning marker in the live feed.

## Applications and Interfaces

### FastAPI Service

Start the API with PowerShell:

```powershell
uvicorn api.main:app --reload --port 8000
```

Endpoints:

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | GET | Returns whether baseline models are loaded. |
| `/predict` | POST | Classifies a single input text. |
| `/predict_batch` | POST | Classifies up to 200 texts in a single vectorized pass; returns predictions, confidence, class probabilities, vocabulary coverage, and total inference time in ms. |
| `/feed?since_id=0&limit=50` | GET | Returns newest comments after `since_id`. |
| `/stats?window_minutes=60` | GET | Returns class counts, shares, average confidence, comments/min, low-coverage count, and per-minute bucket statistics with net sentiment. |

Example Batch Request in PowerShell:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict_batch `
  -ContentType 'application/json' `
  -Body '{"texts":["This stream is incredible!","Terrible lagging","Just tuning in"]}'
```

### Streamlit Real-Time Dashboard

Run the interface with PowerShell:

```powershell
streamlit run dashboard/app.py
```

Dashboard Features:
- **KPI Row**: Total comments, comments/min, positive/neutral/negative shares, average confidence, low-coverage count.
- **Left Column**: Live chat feed of recent comments with color-coded sentiment badges, confidence %, and a `⚠️ Vocab: <50%` warning badge.
- **Right Column**: Interactive class distribution bar chart, rolling net-sentiment per minute line chart, volume counts over time, and throughput/latency indicator.
- **Sidebar**: Subprocess worker controls (start/stop), platform source selector (YouTube/Reddit/Twitch), target input, time window selector, minimum-confidence filter, and pause-feed toggle.
- **Single-Comment Tester**: Expandable manual testing box.

### Ingestion Workers

Run ingestion workers directly or control them via the dashboard sidebar:

```powershell
# YouTube Live Stream or Video Comments
python ingestion/run_ingestion.py --source youtube --target "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Reddit Subreddit
python ingestion/run_ingestion.py --source reddit --target "technology"

# Twitch Chat (requires twitchio)
python ingestion/run_ingestion.py --source twitch --target "shroud"
```

## Running Automated Tests

Run unit tests covering text cleaning, SQLite store deduplication, stats calculations, and YouTube mocked response parsing:

```powershell
python -m pytest
```

## Project Layout

```text
api/             FastAPI application (health, predict, predict_batch, feed, stats)
dashboard/       Streamlit user interface (real-time st.fragment dashboard)
data/            SQLite database (comments.db) and raw/processed training splits
ingestion/       Live ingestion workers (YouTube, Reddit, Twitch, shared buffer)
models/          Saved baseline model artifacts (do not retrain or overwrite)
src/             Preprocessing, features, training, evaluation, ensemble, and store
tests/           Pytest test suite (preprocessing, store, youtube parsing)
```

## Main Technologies

Python, FastAPI, Uvicorn, Streamlit (`st.fragment`), SQLite (WAL mode), scikit-learn, LightGBM, joblib, pandas, PRAW, Google API Client (`googleapiclient`), twitchio, and pytest.
