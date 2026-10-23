"""Stream Reddit comments and send them through the batch prediction & persistence pipeline."""
import logging
import os
import sys
import time

import praw
from prawcore.exceptions import ResponseException, RequestException, OAuthException
from dotenv import load_dotenv

from ingestion.buffer import CommentBuffer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("reddit_stream")


def stream_subreddit(subreddit_name: str, buffer: CommentBuffer, stop_event=None) -> None:
    """Stream comments from a subreddit with exponential backoff on error."""
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "sentiment-ensemble-pipeline")

    if not client_id or not client_secret:
        logger.error("REDDIT_CLIENT_ID or REDDIT_CLIENT_SECRET not set in .env. Exiting.")
        sys.exit(1)

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )

    backoff = 2.0
    logger.info("Connecting to Reddit stream for r/%s...", subreddit_name)

    while True:
        if stop_event and stop_event.is_set():
            break

        try:
            sub = reddit.subreddit(subreddit_name)
            for comment in sub.stream.comments(skip_existing=True, pause_after=5):
                if stop_event and stop_event.is_set():
                    break
                if comment is None:
                    # pause_after yielded None: period check
                    buffer.check_timer()
                    continue

                backoff = 2.0  # Reset backoff on receiving comments

                # Exclude any user / author metadata to respect privacy
                item = {
                    "source": "reddit",
                    "source_comment_id": f"t1_{comment.id}",
                    "text": comment.body,
                    "published_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(comment.created_utc)),
                }
                buffer.add(item)

        except (ResponseException, OAuthException) as auth_err:
            logger.error("Reddit API auth/response error: %s. Backing off for %.1fs.", auth_err, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 120.0)
        except (RequestException, Exception) as req_err:
            logger.error("Reddit connection error: %s. Backing off for %.1fs.", req_err, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 120.0)


def run(target: str | None = None) -> None:
    load_dotenv()
    subreddit_name = target or os.getenv("REDDIT_SUBREDDIT", "technology")
    buffer = CommentBuffer()
    try:
        stream_subreddit(subreddit_name, buffer)
    except KeyboardInterrupt:
        logger.info("Stopping Reddit ingestion gracefully...")
        buffer.flush()
    except Exception as exc:
        logger.critical("Fatal error in Reddit ingestion: %s", exc, exc_info=True)
        buffer.flush()
        sys.exit(1)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    run(target)
