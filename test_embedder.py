from app.ingestion.embedder import embed_texts

texts = [
    "The cat sat on the mat.",
    "A feline rested on the rug.",
    "The stock market crashed today.",
]

vectors = embed_texts(texts)

print(f"Got {len(vectors)} vectors, each with {len(vectors[0])} dimensions")

# quick sanity check: sentences 0 and 1 mean similar things,
# sentence 2 is unrelated — let's confirm the math agrees
import numpy as np

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("\nSimilarity (cat/mat vs feline/rug):", cosine_similarity(vectors[0], vectors[1]))
print("Similarity (cat/mat vs stock market):", cosine_similarity(vectors[0], vectors[2]))