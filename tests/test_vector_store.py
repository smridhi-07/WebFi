"""
Tests for the vector store — using small hand-crafted vectors so we
know exactly what similarity scores to expect, rather than relying on
a real embedding model (which would make tests slow and non-deterministic
across model versions).
"""

import os
import pytest

from app.retrieval.vector_store import VectorStore
from app.ingestion.chunker import Chunk

TEST_STORE_PATH = "tests/_temp_test_store.json"


@pytest.fixture
def store():
    """
    A fresh VectorStore for each test, using a temp file that gets
    cleaned up afterward — so tests never interfere with each other
    or with your real vector_store.json.
    """
    if os.path.exists(TEST_STORE_PATH):
        os.remove(TEST_STORE_PATH)

    vs = VectorStore(path=TEST_STORE_PATH)
    yield vs

    if os.path.exists(TEST_STORE_PATH):
        os.remove(TEST_STORE_PATH)


def test_empty_store_search_returns_nothing(store):
    results = store.search([1.0, 0.0, 0.0], top_k=3)
    assert results == []


def test_add_increases_store_length(store):
    chunks = [Chunk(text="hello", source_url="url1", source_title="Title", chunk_index=0)]
    embeddings = [[1.0, 0.0, 0.0]]
    store.add(chunks, embeddings)
    assert len(store) == 1


def test_search_finds_most_similar_vector(store):
    chunks = [
        Chunk(text="about cats", source_url="url1", source_title="T", chunk_index=0),
        Chunk(text="about stock markets", source_url="url2", source_title="T", chunk_index=0),
    ]
    embeddings = [
        [1.0, 0.0, 0.0],   # points in the same direction as our query
        [0.0, 1.0, 0.0],   # points in a completely different direction
    ]
    store.add(chunks, embeddings)

    results = store.search([1.0, 0.0, 0.0], top_k=1)

    assert len(results) == 1
    top_chunk, score = results[0]
    assert top_chunk.text == "about cats"
    assert score == pytest.approx(1.0)  # identical direction = similarity of 1.0


def test_search_respects_top_k(store):
    chunks = [Chunk(text=f"chunk {i}", source_url="url", source_title="T", chunk_index=i) for i in range(5)]
    embeddings = [[1.0, 0.0, 0.0] for _ in range(5)]
    store.add(chunks, embeddings)

    results = store.search([1.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2


def test_data_persists_after_reload(store):
    chunks = [Chunk(text="persisted", source_url="url1", source_title="T", chunk_index=0)]
    embeddings = [[1.0, 0.0, 0.0]]
    store.add(chunks, embeddings)

    # simulate a fresh program run by creating a new VectorStore
    # pointed at the same file
    reloaded = VectorStore(path=TEST_STORE_PATH)
    assert len(reloaded) == 1
    assert reloaded.items[0].text == "persisted"