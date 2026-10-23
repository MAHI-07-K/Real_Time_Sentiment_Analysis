"""Unit tests for YouTube response parsing and video resolution without external API calls."""
import pytest
from ingestion.youtube_live import (
    extract_video_id,
    parse_live_chat_message,
    parse_comment_thread_item,
    resolve_live_chat_id,
)


def test_extract_video_id_variants():
    """Verify URL parsing handles regular, share, live, and raw video IDs."""
    # Raw 11-char ID
    assert extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    # Full standard watch URL
    assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    # Short youtu.be URL
    assert extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    # Live URL
    assert extract_video_id("https://www.youtube.com/live/dQw4w9WgXcQ?feature=share") == "dQw4w9WgXcQ"
    # Embed URL
    assert extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_extract_video_id_invalid():
    """Verify that invalid strings raise ValueError."""
    with pytest.raises(ValueError):
        extract_video_id("not-a-valid-youtube-url-or-id")


def test_parse_live_chat_message():
    """Verify parsing liveChatMessages item and privacy: author name is excluded."""
    mock_item = {
        "id": "LCC_message_12345",
        "snippet": {
            "type": "textMessageEvent",
            "publishedAt": "2026-10-10T11:00:00.000Z",
            "textMessageDetails": {
                "messageText": "This stream is amazing!",
            },
            "authorChannelId": "UC_SOME_AUTHOR_CHANNEL_ID",  # MUST NOT BE RETAINED
        },
        "authorDetails": {
            "displayName": "John Doe",  # MUST NOT BE RETAINED
        },
    }

    parsed = parse_live_chat_message(mock_item)
    assert parsed is not None
    assert parsed["source"] == "youtube_live"
    assert parsed["source_comment_id"] == "LCC_message_12345"
    assert parsed["text"] == "This stream is amazing!"
    assert parsed["published_at"] == "2026-10-10T11:00:00.000Z"

    # Privacy check: Verify no author name or author channel ID is present
    assert "displayName" not in parsed
    assert "authorChannelId" not in parsed
    assert "John Doe" not in str(parsed)


def test_parse_comment_thread_item():
    """Verify parsing commentThreads item without user identifiers."""
    mock_thread_item = {
        "id": "Ugx_comment_thread_987",
        "snippet": {
            "videoId": "dQw4w9WgXcQ",
            "topLevelComment": {
                "id": "Ugx_comment_thread_987",
                "snippet": {
                    "textDisplay": "Informative video, learned a lot.",
                    "authorDisplayName": "Jane Smith",  # MUST NOT BE RETAINED
                    "authorChannelId": {"value": "UC_JANE"},  # MUST NOT BE RETAINED
                    "publishedAt": "2026-10-10T09:30:00Z",
                },
            },
        },
    }

    parsed = parse_comment_thread_item(mock_thread_item)
    assert parsed is not None
    assert parsed["source"] == "youtube_comment"
    assert parsed["source_comment_id"] == "Ugx_comment_thread_987"
    assert parsed["text"] == "Informative video, learned a lot."
    assert parsed["published_at"] == "2026-10-10T09:30:00Z"

    # Privacy check
    assert "authorDisplayName" not in parsed
    assert "Jane Smith" not in str(parsed)


def test_resolve_live_chat_id_live_video():
    """Verify live chat ID resolution when video is currently live."""
    class MockVideosRequest:
        def execute(self):
            return {
                "items": [
                    {
                        "id": "dQw4w9WgXcQ",
                        "liveStreamingDetails": {
                            "activeLiveChatId": "chat_live_abc123",
                        },
                    }
                ]
            }

    class MockVideosResource:
        def list(self, **kwargs):
            return MockVideosRequest()

    class MockYouTubeClient:
        def videos(self):
            return MockVideosResource()

    chat_id, is_live = resolve_live_chat_id(MockYouTubeClient(), "dQw4w9WgXcQ")
    assert is_live is True
    assert chat_id == "chat_live_abc123"


def test_resolve_live_chat_id_non_live_video():
    """Verify resolution when video is a regular uploaded video (not live)."""
    class MockVideosRequest:
        def execute(self):
            return {
                "items": [
                    {
                        "id": "dQw4w9WgXcQ",
                        "snippet": {"title": "Standard Video"},
                    }
                ]
            }

    class MockVideosResource:
        def list(self, **kwargs):
            return MockVideosRequest()

    class MockYouTubeClient:
        def videos(self):
            return MockVideosResource()

    chat_id, is_live = resolve_live_chat_id(MockYouTubeClient(), "dQw4w9WgXcQ")
    assert is_live is False
    assert chat_id is None

