"""
Fetch a single page and extract clean, readable text from it —
stripping nav bars, footers, ads, and scripts.

trafilatura does the heavy lifting here; it's specifically built for
"give me the main article content" extraction and handles messy real
world HTML far better than hand-rolled BeautifulSoup heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass
from bs4 import BeautifulSoup
from urllib.parse import urljoin

import requests
import trafilatura

HEADERS = {"User-Agent": "rag-website-chat-bot/0.1 (educational project)"}
TIMEOUT = 10


@dataclass
class FetchedPage:
    url: str
    title: str | None
    text: str
    success: bool
    error: str | None = None
    links: list[tuple[str, str]] | None = None  # (link_text, url) pairs

def _extract_page_links(html: str, page_url: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        text = a.get_text(strip=True)
        href = a["href"]
        if not text or href.startswith(("mailto:", "javascript:", "#")):
            continue
        full_url = urljoin(page_url, href).split("#")[0]
        links.append((text, full_url))
    return links[:50]  # cap to avoid huge nav-heavy pages bloating storage

def fetch_page(url: str) -> FetchedPage:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    except requests.RequestException as e:
        return FetchedPage(url=url, title=None, text="", success=False, error=str(e))

    if resp.status_code != 200:
        return FetchedPage(
            url=url, title=None, text="", success=False,
            error=f"HTTP {resp.status_code}",
        )
    resp.encoding = resp.apparent_encoding
    html = resp.text
    page_links = _extract_page_links(html, url)

    extracted = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        favor_precision=True,
    )

    if not extracted:
        return FetchedPage(
            url=url, title=None, text="", success=False,
            error="no extractable content",
        )
    
    # detect soft-404s / error pages: technically 200 OK, but content
    # is just an error message rather than real page content
    error_phrases = [
        "page is gone", "page not found", "404", "doesn't exist",
        "no longer available", "page you requested could not be found",
    ]
    lowered = extracted.lower()
    if len(extracted) < 200 or any(phrase in lowered for phrase in error_phrases):
        return FetchedPage(
            url=url, title=None, text="", success=False,
            error="likely a soft-404 / error page, skipped",
        )

    metadata = trafilatura.extract_metadata(html)
    title = metadata.title if metadata else None

    return FetchedPage(url=url, title=title, text=extracted, success=True, links=page_links)