"""Fetch public YouTube comments and classify them with the local API."""
import os
import sys
import requests
from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()


def main(video_id: str):
    key = os.getenv("YOUTUBE_API_KEY")
    if not key:
        raise RuntimeError("Set YOUTUBE_API_KEY in .env before running this script.")
    youtube = build("youtube", "v3", developerKey=key)
    response = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=100, textFormat="plainText").execute()
    for item in response.get("items", []):
        text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        prediction = requests.post("http://127.0.0.1:8000/predict", json={"text": text}, timeout=15).json()
        print(text[:80], "->", prediction)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python ingestion/youtube_pull.py VIDEO_ID")
    main(sys.argv[1])
