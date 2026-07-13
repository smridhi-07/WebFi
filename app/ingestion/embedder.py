"""
Turns chunk text into embedding vectors using a local sentence-transformers
model — no API key needed, runs entirely on your machine.

An embedding is just a list of numbers (a vector) that represents the
*meaning* of a piece of text. Texts with similar meaning end up with
vectors that are numerically close together, which is what lets us
later do "find chunks similar to this question" via math instead of
keyword matching.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

# loaded once, reused for every embedding call — loading the model
# from disk/downloading it is slow, so we don't want to repeat that
# for every single chunk
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        # small, fast, well-regarded general-purpose embedding model
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Takes a list of raw strings, returns a list of embedding vectors
    (one per input string, same order).
    """
    if not texts:
        return []
    model = get_model()
    vectors = model.encode(texts, show_progress_bar=False)
    return vectors.tolist()


def embed_text(text: str) -> list[float]:
    """Convenience wrapper for embedding a single string."""
    return embed_texts([text])[0]