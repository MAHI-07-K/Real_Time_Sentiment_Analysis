"""Unit tests for the SQLite comment store (deduplication, retrieval, stats, privacy)."""
import pytest
from src.store import init_db, insert_many, fetch_since, fetch_recent, get_stats


@pytest.fixture
def test_db(tmp_path):
    """Provides an isolated SQLite database path for testing."""
    db_file = tmp_path / "test_comments.db"
    init_db(db_file)
    return db_file


def test_insert_many_and_deduplication(test_db):
    """Verify that duplicate source_comment_id records are ignored."""
    records = [
        {
            "source": "youtube_live",
            "source_comment_id": "yt_msg_001",
            "text": "Great stream!",
            "label": 1.0,
            "confidence": 0.95,
            "p_neg": 0.02,
            "p_neu": 0.03,
            "p_pos": 0.95,
            "vocab_coverage": 0.8,
            "published_at": "2026-10-10T10:00:00Z",
        },
        {
            "source": "youtube_live",
            "source_comment_id": "yt_msg_002",
            "text": "Average performance",
            "label": 0.0,
            "confidence": 0.70,
            "p_neg": 0.15,
            "p_neu": 0.70,
            "p_pos": 0.15,
            "vocab_coverage": 0.6,
            "published_at": "2026-10-10T10:01:00Z",
        },
    ]

    # First insert: 2 records inserted
    inserted = insert_many(records, db_path=test_db)
    assert inserted == 2

    # Duplicate insertion attempt
    duplicates = [
        {
            "source": "youtube_live",
            "source_comment_id": "yt_msg_001",  # duplicate
            "text": "Duplicate content",
            "label": 1.0,
            "confidence": 0.95,
            "p_neg": 0.02,
            "p_neu": 0.03,
            "p_pos": 0.95,
            "vocab_coverage": 0.8,
        },
        {
            "source": "reddit",
            "source_comment_id": "reddit_t1_123",  # new
            "text": "This is terrible",
            "label": -1.0,
            "confidence": 0.88,
            "p_neg": 0.88,
            "p_neu": 0.07,
            "p_pos": 0.05,
            "vocab_coverage": 0.4,
        },
    ]

    inserted_second = insert_many(duplicates, db_path=test_db)
    assert inserted_second == 1  # only reddit_t1_123 is inserted

    feed = fetch_since(0, limit=100, db_path=test_db)
    assert len(feed) == 3


def test_fetch_since_pagination(test_db):
    """Verify incremental polling via fetch_since(since_id)."""
    records = [
        {
            "source": "test",
            "source_comment_id": f"msg_{i}",
            "text": f"Comment {i}",
            "label": 1.0,
            "confidence": 0.9,
            "p_neg": 0.05,
            "p_neu": 0.05,
            "p_pos": 0.9,
            "vocab_coverage": 0.75,
        }
        for i in range(1, 6)
    ]
    insert_many(records, db_path=test_db)

    # First poll: all 5
    batch1 = fetch_since(0, limit=3, db_path=test_db)
    assert len(batch1) == 3
    last_id = batch1[-1]["id"]

    # Next poll: remaining 2
    batch2 = fetch_since(last_id, limit=3, db_path=test_db)
    assert len(batch2) == 2
    assert batch2[0]["id"] > last_id


def test_get_stats_aggregation(test_db):
    """Verify calculation of counts, shares, CPM, low-coverage count, and net sentiment."""
    records = [
        # Positive
        {
            "source": "yt",
            "source_comment_id": "p1",
            "text": "Love it",
            "label": 1.0,
            "confidence": 0.90,
            "p_neg": 0.05,
            "p_neu": 0.05,
            "p_pos": 0.90,
            "vocab_coverage": 1.0,
            "ingested_at": "2026-10-10T10:00:10Z",
        },
        # Neutral
        {
            "source": "yt",
            "source_comment_id": "neu1",
            "text": "Okay okay",
            "label": 0.0,
            "confidence": 0.60,
            "p_neg": 0.20,
            "p_neu": 0.60,
            "p_pos": 0.20,
            "vocab_coverage": 0.3,  # low coverage (<0.5)
            "ingested_at": "2026-10-10T10:00:20Z",
        },
        # Negative
        {
            "source": "yt",
            "source_comment_id": "n1",
            "text": "Awful",
            "label": -1.0,
            "confidence": 0.85,
            "p_neg": 0.85,
            "p_neu": 0.10,
            "p_pos": 0.05,
            "vocab_coverage": 0.2,  # low coverage (<0.5)
            "ingested_at": "2026-10-10T10:01:05Z",
        },
    ]
    insert_many(records, db_path=test_db)

    stats = get_stats(window_minutes=None, db_path=test_db)
    assert stats["total_comments"] == 3
    assert stats["class_counts"] == {"-1": 1, "0": 1, "1": 1}
    assert stats["low_coverage_count"] == 2
    assert abs(stats["avg_confidence"] - (0.90 + 0.60 + 0.85) / 3) < 0.001

    # Check shares
    assert abs(stats["class_shares"]["1"] - (1 / 3)) < 0.001

    # Check buckets
    assert len(stats["buckets"]) == 2  # 10:00 and 10:01
    bucket_1000 = [b for b in stats["buckets"] if b["minute"] == "2026-10-10T10:00"][0]
    assert bucket_1000["total"] == 2
    # net_sentiment for 10:00 = mean of (0.90 - 0.05) and (0.20 - 0.20) = mean(0.85, 0.0) = 0.425
    assert abs(bucket_1000["net_sentiment"] - 0.425) < 0.001


def test_delete_comments(test_db):
    """Verify deleting comments by source platform or clearing all."""
    from src.store import delete_comments
    records = [
        {"source": "youtube_live", "source_comment_id": "yt_1", "text": "yt 1", "label": 1.0, "confidence": 0.9, "p_neg": 0.0, "p_neu": 0.1, "p_pos": 0.9, "vocab_coverage": 1.0},
        {"source": "youtube_live", "source_comment_id": "yt_2", "text": "yt 2", "label": 0.0, "confidence": 0.8, "p_neg": 0.1, "p_neu": 0.8, "p_pos": 0.1, "vocab_coverage": 1.0},
        {"source": "reddit", "source_comment_id": "rd_1", "text": "rd 1", "label": -1.0, "confidence": 0.7, "p_neg": 0.7, "p_neu": 0.2, "p_pos": 0.1, "vocab_coverage": 1.0},
    ]
    insert_many(records, db_path=test_db)
    assert len(fetch_since(0, 10, db_path=test_db)) == 3

    # Delete youtube comments
    deleted_yt = delete_comments("youtube", db_path=test_db)
    assert deleted_yt == 2
    remaining = fetch_since(0, 10, db_path=test_db)
    assert len(remaining) == 1
    assert remaining[0]["source"] == "reddit"

    # Delete all remaining
    deleted_all = delete_comments(None, db_path=test_db)
    assert deleted_all == 1
    assert len(fetch_since(0, 10, db_path=test_db)) == 0


def test_fetch_text_display_and_source_filter(test_db):
    """Verify that comments return text_display and support platform filtering."""
    records = [
        {"source": "youtube", "source_comment_id": "yt_1", "text": "yt comment with emoji 😀", "label": 1.0, "confidence": 0.9, "p_neg": 0.0, "p_neu": 0.1, "p_pos": 0.9, "vocab_coverage": 1.0},
        {"source": "reddit", "source_comment_id": "rd_1", "text": "reddit comment", "label": -1.0, "confidence": 0.7, "p_neg": 0.7, "p_neu": 0.2, "p_pos": 0.1, "vocab_coverage": 1.0},
    ]
    insert_many(records, db_path=test_db)

    # Check text_display field presence
    recent = fetch_recent(limit=10, db_path=test_db)
    assert len(recent) == 2
    assert "text_display" in recent[0]
    assert recent[0]["text_display"] == recent[0]["text"]

    # Filter by source: youtube
    yt_comments = fetch_recent(limit=10, source="youtube", db_path=test_db)
    assert len(yt_comments) == 1
    assert yt_comments[0]["source"] == "youtube"

    # Filter by source: reddit
    rd_comments = fetch_since(0, limit=10, source="reddit", db_path=test_db)
    assert len(rd_comments) == 1
    assert rd_comments[0]["source"] == "reddit"


def test_get_stats_5m_metrics(test_db):
    """Verify comments_per_minute_5m and messages_per_second calculation."""
    records = [
        {"source": "youtube", "source_comment_id": f"msg_{i}", "text": f"hi {i}", "label": 1.0, "confidence": 0.9, "p_neg": 0.0, "p_neu": 0.1, "p_pos": 0.9, "vocab_coverage": 1.0}
        for i in range(10)
    ]
    insert_many(records, db_path=test_db)

    stats = get_stats(window_minutes=5, db_path=test_db)
    assert "comments_per_minute_5m" in stats
    assert "messages_per_second" in stats
    assert stats["comments_per_minute_5m"] > 0
    assert stats["messages_per_second"] > 0



