from app.crawler.fetcher import fetch_page
from app.ingestion.chunker import chunk_text
from app.ingestion.embedder import embed_texts, embed_text
from app.retrieval.vector_store import VectorStore

# fetch and chunk one real page
page = fetch_page("https://docs.python.org/3/tutorial/introduction.html")
chunks = chunk_text(page.text, page.url, page.title)
print(f"Got {len(chunks)} chunks")

# embed all chunks at once (batch, as discussed)
embeddings = embed_texts([c.text for c in chunks])

# store them
store = VectorStore(path="test_store.json")
store.add(chunks, embeddings)
print(f"Store now has {len(store)} items")

# now search: ask a question, embed it, find the most similar chunks
question = "How do I write a comment in Python?"
question_vec = embed_text(question)

results = store.search(question_vec, top_k=3)

print(f"\nTop 3 results for: '{question}'\n")
for chunk, score in results:
    print(f"score={score:.3f} | {chunk.text[:150]}...\n")