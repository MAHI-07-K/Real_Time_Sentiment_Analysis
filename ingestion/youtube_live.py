"""Live YouTube comment ingestion worker supporting live chats and video comments.

- Resolves activeLiveChatId via videos.list.
- If live, polls liveChatMessages.list honoring pollingIntervalMillis and nextPageToken.
- If not live or no active chat, falls back to polling commentThreads.list (newest first)
  every N seconds, deduping by comment id.
- Buffers comments through CommentBuffer (flushing every 2s or 50 messages).
- Handles quota-exceeded and network errors with exponential backoff.
- Privacy: Never captures or stores author names or user identifiers.
"""
import logging
import os
import re
import sys
import time
from typing import Any
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from ingestion.buffer import CommentBuffer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("youtube_live")


def extract_video_id(target: str) -> str:
    """Extract YouTube 11-character video ID from a URL or raw ID."""
    target = target.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", target):
        return target

    parsed = urlparse(target)
    if parsed.hostname in ("www.youtube.com", "youtube.com", "m.youtube.com"):
        if parsed.path == "/watch":
            qs = parse_qs(parsed.query)
            if "v" in qs and qs["v"]:
                return qs["v"][0]
        elif parsed.path.startswith(("/live/", "/embed/", "/v/", "/shorts/")):
            parts = parsed.path.strip("/").split("/")
            if len(parts) >= 2 and re.fullmatch(r"[A-Za-z0-9_-]{11}", parts[1]):
                return parts[1]
    elif parsed.hostname in ("youtu.be", "www.youtu.be"):
        path = parsed.path.strip("/")
        if re.fullmatch(r"[A-Za-z0-9_-]{11}", path):
            return path

    raise ValueError(f"Could not parse valid YouTube video ID from: {target}")


def parse_live_chat_message(item: dict[str, Any]) -> dict[str, Any] | None:
    """Parse a single liveChatMessage item without storing any author/user info."""
    msg_id = item.get("id")
    if not msg_id:
        return None

    snippet = item.get("snippet", {})
    text_details = snippet.get("textMessageDetails", {})
    message_text = text_details.get("messageText", "")
    if not message_text:
        display_msg = snippet.get("displayMessage", "")
        if display_msg:
            message_text = display_msg

    if not message_text.strip():
        return None

    published_at = snippet.get("publishedAt")
    return {
        "source": "youtube_live",
        "source_comment_id": str(msg_id),
        "text": message_text.strip(),
        "published_at": published_at,
    }


def parse_comment_thread_item(item: dict[str, Any]) -> dict[str, Any] | None:
    """Parse a topLevelComment from commentThreads.list without storing author info."""
    top_level = item.get("snippet", {}).get("topLevelComment", {})
    c_id = top_level.get("id")
    if not c_id:
        return None

    c_snippet = top_level.get("snippet", {})
    text = c_snippet.get("textDisplay", "")
    if not text.strip():
        return None

    published_at = c_snippet.get("publishedAt")
    return {
        "source": "youtube_comment",
        "source_comment_id": str(c_id),
        "text": text.strip(),
        "published_at": published_at,
    }


def resolve_live_chat_id(youtube, video_id: str) -> tuple[str | None, bool]:
    """Check videos.list for activeLiveChatId and live streaming status."""
    resp = youtube.videos().list(
        part="snippet,liveStreamingDetails",
        id=video_id,
    ).execute()

    items = resp.get("items", [])
    if not items:
        raise ValueError(f"No YouTube video found with ID {video_id}")

    live_details = items[0].get("liveStreamingDetails", {})
    active_chat_id = live_details.get("activeLiveChatId")
    is_live = bool(active_chat_id)
    return active_chat_id, is_live


def stream_live_chat(youtube, active_chat_id: str, buffer: CommentBuffer, stop_event=None) -> None:
    """Poll liveChatMessages.list honoring pollingIntervalMillis and nextPageToken."""
    logger.info("Starting live chat polling for activeLiveChatId: %s", active_chat_id)
    page_token = None
    backoff = 2.0

    while True:
        if stop_event and stop_event.is_set():
            break

        try:
            req_kwargs: dict[str, Any] = {
                "liveChatId": active_chat_id,
                "part": "snippet",
                "maxResults": 200,
            }
            if page_token:
                req_kwargs["pageToken"] = page_token

            resp = youtube.liveChatMessages().list(**req_kwargs).execute()
            backoff = 2.0  # Reset backoff on success

            items = resp.get("items", [])
            for item in items:
                parsed = parse_live_chat_message(item)
                if parsed:
                    buffer.add(parsed)

            # Flush any remaining items immediately so the live dashboard updates without waiting
            buffer.flush()

            poll_interval_ms = resp.get("pollingIntervalMillis", 2000)
            page_token = resp.get("nextPageToken")
            sleep_sec = max(poll_interval_ms / 1000.0, 1.0)
            time.sleep(sleep_sec)

        except HttpError as err:
            status_code = err.resp.status
            if status_code in (403, 429):
                logger.error("YouTube API quota/rate limit error (HTTP %d): %s. Backing off for %.1fs.", status_code, err, backoff)
            else:
                logger.error("YouTube API HTTP error %d: %s. Backing off for %.1fs.", status_code, err, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 120.0)
        except Exception as exc:
            logger.error("Unexpected network error while polling live chat: %s. Backing off for %.1fs.", exc, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 120.0)


def stream_video_comments(youtube, video_id: str, buffer: CommentBuffer, poll_interval: float = 10.0, stop_event=None) -> None:
    """Fall back to polling commentThreads.list for normal or ended videos."""
    logger.info("Polling commentThreads.list for video %s every %.1fs (newest first).", video_id, poll_interval)
    backoff = 2.0
    seen_ids: set[str] = set()

    while True:
        if stop_event and stop_event.is_set():
            break

        try:
            resp = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                order="time",
                maxResults=50,
                textFormat="plainText",
            ).execute()
            backoff = 2.0

            items = resp.get("items", [])
            for item in items:
                parsed = parse_comment_thread_item(item)
                if parsed and parsed["source_comment_id"] not in seen_ids:
                    seen_ids.add(parsed["source_comment_id"])
                    buffer.add(parsed)

            # Flush immediately to ensure comments appear in the dashboard right away
            buffer.flush()

            # Cap in-memory set size
            if len(seen_ids) > 10_000:
                seen_ids = set(list(seen_ids)[-5000:])

            time.sleep(poll_interval)

        except HttpError as err:
            status_code = err.resp.status
            if status_code in (403, 429):
                logger.error("YouTube API quota/rate limit error (HTTP %d): %s. Backing off for %.1fs.", status_code, err, backoff)
            else:
                logger.error("YouTube API HTTP error %d: %s. Backing off for %.1fs.", status_code, err, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 120.0)
        except Exception as exc:
            logger.error("Unexpected error while fetching video comments: %s. Backing off for %.1fs.", exc, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 120.0)


def run(target: str) -> None:
    """Main YouTube ingestion entrypoint."""
    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        logger.error("YOUTUBE_API_KEY is not set in environment or .env. Exiting.")
        sys.exit(1)

    video_id = extract_video_id(target)
    logger.info("Resolved video ID: %s", video_id)

    youtube = build("youtube", "v3", developerKey=api_key)
    buffer = CommentBuffer()

    try:
        active_chat_id, is_live = resolve_live_chat_id(youtube, video_id)
        if is_live and active_chat_id:
            logger.info("Video is LIVE! Found activeLiveChatId: %s", active_chat_id)
            stream_live_chat(youtube, active_chat_id, buffer)
        else:
            logger.info("Video is not a live stream or chat is disabled. Falling back to commentThreads.list.")
            stream_video_comments(youtube, video_id, buffer)
    except KeyboardInterrupt:
        logger.info("Stopping YouTube ingestion gracefully...")
        buffer.flush()
    except Exception as exc:
        logger.critical("Fatal error in YouTube ingestion: %s", exc, exc_info=True)
        buffer.flush()
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingestion/youtube_live.py VIDEO_ID_OR_URL", file=sys.stderr)
        sys.exit(1)
    run(sys.argv[1])

