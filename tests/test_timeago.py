from datetime import datetime, timezone, timedelta
import pytest
from src.timeago import time_ago, format_seconds_ago


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0, "0 sec ago"),
        (59, "59 sec ago"),
        (60, "1 min ago"),
        (3599, "59 min ago"),
        (3600, "1 hr ago"),
        (86400, "1 day ago"),
    ],
)
def test_time_ago_boundaries_direct_seconds(seconds, expected):
    """Test boundary conditions for time_ago when given seconds directly."""
    assert format_seconds_ago(seconds) == expected
    assert time_ago(seconds) == expected


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0, "0 sec ago"),
        (59, "59 sec ago"),
        (60, "1 min ago"),
        (3599, "59 min ago"),
        (3600, "1 hr ago"),
        (86400, "1 day ago"),
    ],
)
def test_time_ago_boundaries_datetime_and_iso(seconds, expected):
    """Test boundary conditions for time_ago using datetime and ISO formatted strings."""
    fixed_now = datetime(2026, 10, 10, 12, 0, 0, tzinfo=timezone.utc)
    target_dt = fixed_now - timedelta(seconds=seconds)
    
    # Passing datetime object
    assert time_ago(target_dt, now=fixed_now) == expected
    
    # Passing ISO 8601 string
    iso_str = target_dt.isoformat()
    assert time_ago(iso_str, now=fixed_now) == expected


def test_time_ago_edge_cases():
    """Test empty, invalid, and negative inputs."""
    assert time_ago(None) == ""
    assert time_ago("") == ""
    assert time_ago("invalid-timestamp") == ""
    # Negative seconds (clock drift into slight future) should clamp to 0 sec ago
    assert time_ago(-5) == "0 sec ago"

