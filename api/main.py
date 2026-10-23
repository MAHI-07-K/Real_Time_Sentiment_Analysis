import time
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.ensemble import SentimentEnsemble
from src import store

app = FastAPI(title="Sentiment Ensemble API", version="1.0.0")
ensemble: SentimentEnsemble | None = None


class TextIn(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)


class BatchIn(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=200)


@app.on_event("startup")
def load_models():
    global ensemble
    try:
        ensemble = SentimentEnsemble()
    except FileNotFoundError:
        ensemble = None
    # Ensure database schema is ready
    store.init_db()


@app.get("/health")
def health():
    return {"ready": ensemble is not None}


@app.post("/predict")
def predict(payload: TextIn):
    if ensemble is None:
        raise HTTPException(503, "Models are not ready. Run preprocessing and training first.")
    labels, probabilities = ensemble.predict([payload.text])
    return {"prediction": str(labels[0]), "confidence": float(probabilities[0].max()),
            "probabilities": {str(label): float(probability) for label, probability in zip(ensemble.classes_, probabilities[0])}}

last_batch_inference_ms: float = 0.0


@app.post("/predict_batch")
def predict_batch(payload: BatchIn):
    """Classify up to 200 texts in a single vectorized pass.
    
    Returns predictions with label, confidence, class probabilities,
    vocab_coverage, and total inference time in ms.
    """
    global last_batch_inference_ms
    if ensemble is None:
        raise HTTPException(503, "Models are not ready. Run preprocessing and training first.")

    start_time = time.perf_counter()
    labels, probabilities, coverages = ensemble.predict_detailed(payload.texts)
    inference_ms = (time.perf_counter() - start_time) * 1000.0
    last_batch_inference_ms = round(inference_ms, 2)

    classes_str = [str(cls) for cls in ensemble.classes_]
    predictions = []
    for idx, (label, probs, cov) in enumerate(zip(labels, probabilities, coverages)):
        prob_dict = {classes_str[i]: float(probs[i]) for i in range(len(classes_str))}
        predictions.append({
            "text": payload.texts[idx],
            "prediction": str(label),
            "confidence": float(probs.max()),
            "probabilities": prob_dict,
            "vocab_coverage": float(cov),
        })

    return {
        "predictions": predictions,
        "count": len(predictions),
        "inference_time_ms": round(inference_ms, 2),
    }


@app.get("/feed")
def feed(
    since_id: Annotated[int, Query(description="Fetch comments with id > since_id")] = 0,
    limit: Annotated[int, Query(ge=1, le=500, description="Max comments to fetch")] = 50,
    order: Annotated[str, Query(description="Order of results: 'desc' (newest first) or 'asc'")] = "desc",
    source: Annotated[str | None, Query(description="Platform source filter")] = None,
):
    """Return comments after since_id, with optional descending ordering (newest first)."""
    if order.lower() == "desc" and since_id == 0:
        comments = store.fetch_recent(limit=limit, source=source)
    else:
        comments = store.fetch_since(since_id=since_id, limit=limit, source=source)
        if order.lower() == "desc":
            comments = sorted(comments, key=lambda c: c["id"], reverse=True)
    return {"comments": comments, "count": len(comments)}


@app.delete("/feed")
def delete_feed(
    source: Annotated[str | None, Query(description="Platform source to drop ('youtube', 'reddit', 'twitch'), or omit for all")] = None,
):
    """Drop comments from store, optionally filtering by source platform."""
    deleted_count = store.delete_comments(source=source)
    return {"status": "ok", "deleted": deleted_count, "source": source or "all"}


@app.get("/stats")
def stats(
    window_minutes: Annotated[int | None, Query(description="Lookback window in minutes; omit or 0 for all")] = 60,
    source: Annotated[str | None, Query(description="Platform source filter")] = None,
):
    """Return aggregated sentiment statistics and per-minute trends."""
    data = store.get_stats(window_minutes=window_minutes, source=source)
    data["last_batch_inference_ms"] = last_batch_inference_ms
    return data




