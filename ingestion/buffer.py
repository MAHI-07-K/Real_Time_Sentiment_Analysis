"""Shared batching buffer for streaming comment ingestion.

Buffers comments and flushes them to /predict_batch every flush_interval_seconds
or when reaching max_batch_size (whichever comes first), then inserts predictions
directly into the SQLite store.
"""
import logging
import os
import time
from typing import Any
import requests

from src import store

logger = logging.getLogger("ingestion.buffer")


class CommentBuffer:
    def __init__(
        self,
        api_url: str | None = None,
        max_batch_size: int = 50,
        flush_interval_seconds: float = 2.0,
        db_path=None,
    ):
        self.api_url = (api_url or os.getenv("SENTIMENT_API_URL", "http://127.0.0.1:8000")).rstrip("/")
        self.max_batch_size = max_batch_size
        self.flush_interval_seconds = flush_interval_seconds
        self.db_path = db_path
        self._buffer: list[dict[str, Any]] = []
        self._last_flush_time = time.monotonic()
        if self.db_path:
            store.init_db(self.db_path)
        else:
            store.init_db()

    def add(self, item: dict[str, Any]) -> None:
        """Add a comment dict to buffer and flush if full or interval elapsed.
        
        Expected item keys:
          - source: str
          - source_comment_id: str
          - text: str
          - published_at: str | None
        """
        # Strictly ensure author/user identifiers are never carried
        clean_item = {
            "source": item.get("source", "unknown"),
            "source_comment_id": str(item["source_comment_id"]),
            "text": str(item.get("text", "")).strip(),
            "published_at": item.get("published_at"),
        }
        if not clean_item["text"]:
            return

        self._buffer.append(clean_item)
        
        if len(self._buffer) >= self.max_batch_size:
            self.flush()
        elif time.monotonic() - self._last_flush_time >= self.flush_interval_seconds:
            self.flush()

    def check_timer(self) -> None:
        """Flush buffer if flush_interval_seconds has elapsed since last flush."""
        if self._buffer and (time.monotonic() - self._last_flush_time >= self.flush_interval_seconds):
            self.flush()

    def flush(self) -> int:
        """Flush buffered items to /predict_batch, then persist to SQLite store."""
        if not self._buffer:
            self._last_flush_time = time.monotonic()
            return 0

        batch = self._buffer
        self._buffer = []
        self._last_flush_time = time.monotonic()

        texts = [b["text"] for b in batch]
        try:
            resp = requests.post(
                f"{self.api_url}/predict_batch",
                json={"texts": texts},
                timeout=15.0,
            )
            resp.raise_for_status()
            data = resp.json()
            predictions = data.get("predictions", [])
        except Exception as exc:
            logger.error("Failed to call %s/predict_batch: %s. Dropping batch.", self.api_url, exc)
            return 0

        records_to_insert = []
        for meta, pred in zip(batch, predictions):
            probs = pred.get("probabilities", {})
            records_to_insert.append({
                "source": meta["source"],
                "source_comment_id": meta["source_comment_id"],
                "text": meta["text"],
                "label": float(pred.get("prediction", 0.0)),
                "confidence": float(pred.get("confidence", 0.0)),
                "p_neg": float(probs.get("-1.0", probs.get("-1", 0.0))),
                "p_neu": float(probs.get("0.0", probs.get("0", 0.0))),
                "p_pos": float(probs.get("1.0", probs.get("1", 0.0))),
                "vocab_coverage": float(pred.get("vocab_coverage", 0.0)),
                "published_at": meta.get("published_at"),
            })

        if self.db_path:
            inserted = store.insert_many(records_to_insert, db_path=self.db_path)
        else:
            inserted = store.insert_many(records_to_insert)

        logger.info(
            "Flushed %d comments (inserted %d new, duplicates ignored) in %.1fms inference time.",
            len(batch),
            inserted,
            data.get("inference_time_ms", 0.0),
        )
        return inserted

