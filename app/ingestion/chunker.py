"""
Splits page text into overlapping chunks suitable for embedding.

Why overlap matters: if we chunk with zero overlap, a sentence or
idea that happens to fall right at a chunk boundary gets split in
half, and neither resulting chunk fully captures its meaning.
Overlapping chunks means the boundary content still appears whole in
at least one chunk.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source_url: str
    source_title: str | None
    chunk_index: int


def chunk_text(
    text: str,
    source_url: str,
    source_title: str | None,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """
    Splits text into chunks of roughly `chunk_size` characters, with
    `overlap` characters repeated at the start of each chunk from the
    end of the previous one.
    """
    text = text.strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text_piece = text[start:end].strip()

        if chunk_text_piece:
            chunks.append(
                Chunk(
                    text=chunk_text_piece,
                    source_url=source_url,
                    source_title=source_title,
                    chunk_index=index,
                )
            )
            index += 1

        # move the window forward, but step back by `overlap` so
        # consecutive chunks share some content
        start += chunk_size - overlap

    return chunks