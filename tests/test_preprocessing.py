from src.preprocessing import clean_text


def test_clean_text_removes_urls_and_mentions():
    assert clean_text("WOW! @user Not bad: https://example.com #great") == "wow not bad great"


def test_clean_text_handles_empty_values():
    assert clean_text(None) == ""
    assert clean_text("😀!!!") == ""

