from app.crawler.fetcher import fetch_page
from app.ingestion.chunker import chunk_text

page = fetch_page("https://docs.python.org/3/tutorial/introduction.html")
print("Fetched:", page.success, "| text length:", len(page.text))

chunks = chunk_text(page.text, page.url, page.title, chunk_size=500, overlap=50)
print(f"\nSplit into {len(chunks)} chunks\n")

for c in chunks[:3]:
    print(f"--- chunk {c.chunk_index} ({len(c.text)} chars) ---")
    print(c.text[:150], "...\n")