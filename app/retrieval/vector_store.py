"""
A minimal file-backed vector store.

Stores chunks + their embedding vectors, and supports similarity
search. This exists in place of ChromaDB (which failed to install
due to a Windows build-tools issue) — same core idea, simplified:
everything lives in one JSON file, and search is a brute-force
cosine-similarity scan across all stored vectors.

This does NOT scale to millions of chunks (a real vector database
uses specialized indexes for that), but for a single crawled site's
worth of pages, a brute-force scan is fast enough and much simpler
to understand and debug.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

import numpy as np

from app.ingestion.chunker import Chunk

STORE_PATH = "vector_store.json"


@dataclass
class StoredChunk:
    text: str
    source_url: str
    source_title: str | None
    chunk_index: int
    embedding: list[float]


class VectorStore:
    def __init__(self, path: str = STORE_PATH):
        self.path = path
        self.items: list[StoredChunk] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.items = [StoredChunk(**item) for item in raw]

    def _save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump([asdict(item) for item in self.items], f)

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        for chunk, embedding in zip(chunks, embeddings):
            self.items.append(
                StoredChunk(
                    text=chunk.text,
                    source_url=chunk.source_url,
                    source_title=chunk.source_title,
                    chunk_index=chunk.chunk_index,
                    embedding=embedding,
                )
            )
        self._save()

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[tuple[StoredChunk, float]]:
        """
        Returns the top_k stored chunks most similar to the query
        embedding, as (chunk, similarity_score) pairs, sorted by
        similarity descending.
        """
        if not self.items:
            return []

        query_vec = np.array(query_embedding)
        query_norm = np.linalg.norm(query_vec)

        scored: list[tuple[StoredChunk, float]] = []
        for item in self.items:
            item_vec = np.array(item.embedding)
            similarity = np.dot(query_vec, item_vec) / (
                query_norm * np.linalg.norm(item_vec)
            )
            scored.append((item, float(similarity)))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self.items)