"""Ingestion worker for Twitch chat channels using twitchio (optional dependency).

Buffers live chat messages through CommentBuffer.
Privacy: Does not record username or author identities.
"""
import logging
import os
import sys
import time

from dotenv import load_dotenv

from ingestion.buffer import CommentBuffer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("twitch_chat")


def run(channel: str) -> None:
    load_dotenv()
    token = os.getenv("TWITCH_ACCESS_TOKEN") or os.getenv("TWITCH_OAUTH_TOKEN")
    channel_name = channel.lstrip("#").strip().lower()

    try:
        from twitchio.ext import commands
    except ImportError:
        logger.error(
            "twitchio is not installed. To stream Twitch chat, run: pip install twitchio"
        )
        sys.exit(1)

    buffer = CommentBuffer()

    class Bot(commands.Bot):
        def __init__(self):
            super().__init__(
                token=token or "oauth:placeholder",
                prefix="!",
                initial_channels=[channel_name],
            )

        async def event_ready(self):
            logger.info("Connected to Twitch channel: %s", channel_name)

        async def event_message(self, message):
            if message.echo:
                return
            # Strict privacy: do not record author names or sender IDs
            msg_id = getattr(message, "id", None) or f"twitch_{time.time_ns()}"
            item = {
                "source": "twitch",
                "source_comment_id": str(msg_id),
                "text": message.content,
                "published_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            buffer.add(item)

    bot = Bot()
    try:
        bot.run()
    except KeyboardInterrupt:
        logger.info("Stopping Twitch ingestion...")
        buffer.flush()
    except Exception as exc:
        logger.error("Twitch client error: %s", exc)
        buffer.flush()
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingestion/twitch_chat.py CHANNEL", file=sys.stderr)
        sys.exit(1)
    run(sys.argv[1])

