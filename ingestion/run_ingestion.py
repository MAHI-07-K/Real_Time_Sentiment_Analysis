"""CLI launcher for comment ingestion workers.

Usage:
    python ingestion/run_ingestion.py --source youtube --target dQw4w9WgXcQ
    python ingestion/run_ingestion.py --source reddit --target technology
    python ingestion/run_ingestion.py --source twitch --target shroud
"""
import argparse
import logging
import sys

from ingestion import youtube_live, reddit_stream, twitch_chat

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_ingestion")


def main():
    parser = argparse.ArgumentParser(description="Start an ingestion worker for live social comments.")
    parser.add_argument(
        "--source",
        choices=["youtube", "reddit", "twitch"],
        required=True,
        help="Data source platform",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Target video ID/URL (YouTube), subreddit (Reddit), or channel name (Twitch)",
    )
    args = parser.parse_args()

    logger.info("Starting ingestion worker for source '%s' targeting '%s'...", args.source, args.target)

    if args.source == "youtube":
        youtube_live.run(args.target)
    elif args.source == "reddit":
        reddit_stream.run(args.target)
    elif args.source == "twitch":
        twitch_chat.run(args.target)
    else:
        logger.error("Unknown source: %s", args.source)
        sys.exit(1)


if __name__ == "__main__":
    main()

