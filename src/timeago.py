"""Relative time formatting helper for social media comments."""
from datetime import datetime, timezone
from typing import Any


def format_seconds_ago(seconds: float | int) -> str:
    """Format elapsed seconds into a relative time string.
    
    Examples:
        0s -> '0 sec ago'
        59s -> '59 sec ago'
        60s -> '1 min ago'
        3599s -> '59 min ago'
        3600s -> '1 hr ago'
        86400s -> '1 day ago'
    """
    sec = max(0, int(seconds))
    if sec < 60:
        return f"{sec} sec ago"
    if sec < 3600:
        minutes = sec // 60
        return f"{minutes} min ago"
    if sec < 86400:
        hours = sec // 3600
        return f"{hours} hr ago"
    days = sec // 86400
    return f"{days} day ago" if days == 1 else f"{days} days ago"


def time_ago(
    val: Any,
    now: datetime | None = None,
) -> str:
    """Convert an ISO timestamp string, datetime, or raw seconds count into a relative time string.
    
    If `val` is numeric (int/float), it is treated as elapsed seconds directly.
    If `val` is an ISO timestamp string or datetime, elapsed seconds are computed against `now`
    (defaults to current UTC time).
    """
    if val is None:
        return ""

    # Direct seconds passed
    if isinstance(val, (int, float)):
        return format_seconds_ago(val)

    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    # Datetime instance
    if isinstance(val, datetime):
        dt = val if val.tzinfo is not None else val.replace(tzinfo=timezone.utc)
        elapsed = (now - dt).total_seconds()
        return format_seconds_ago(elapsed)

    # String ISO timestamp
    if isinstance(val, str):
        text = val.strip()
        if not text:
            return ""
        try:
            # Handle trailing 'Z' if present
            clean_text = text.replace("Z", "+00:00")
            dt = datetime.fromisoformat(clean_text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            elapsed = (now - dt).total_seconds()
            return format_seconds_ago(elapsed)
        except (ValueError, TypeError):
            return ""

    return ""

