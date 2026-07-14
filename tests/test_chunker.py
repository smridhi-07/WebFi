"""
Real pytest tests for the chunker — actual assertions, not printed
output. Run with: pytest
"""

from app.ingestion.chunker import chunk_text


def test_empty_text_returns_no_chunks():
    chunks = chunk_text("", source_url="http://example.com", source_title="Test")
    assert chunks == []


def test_short_text_returns_one_chunk():
    text = "This is a short piece of text."
    chunks = chunk_text(text, source_url="http://example.com", source_title="Test")
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_long_text_splits_into_multiple_chunks():
    text = "a" * 1200  # well over one chunk_size (default 500)
    chunks = chunk_text(text, source_url="http://example.com", source_title="Test", chunk_size=500, overlap=50)
    assert len(chunks) > 1


def test_chunks_carry_correct_metadata():
    text = "Some content here."
    chunks = chunk_text(text, source_url="http://example.com/page", source_title="My Page")
    assert chunks[0].source_url == "http://example.com/page"
    assert chunks[0].source_title == "My Page"


def test_chunk_indices_are_sequential():
    text = "a" * 1500
    chunks = chunk_text(text, source_url="http://example.com", source_title="Test", chunk_size=500, overlap=50)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_overlap_means_consecutive_chunks_share_content():
    text = "a" * 1000
    chunks = chunk_text(text, source_url="http://example.com", source_title="Test", chunk_size=500, overlap=50)
    # last 50 chars of chunk 0 should appear at the start of chunk 1
    assert chunks[0].text[-50:] == chunks[1].text[:50]