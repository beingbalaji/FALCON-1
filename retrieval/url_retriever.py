import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from config import REQUEST_TIMEOUT, USER_AGENT


def fetch_url_passages(url: str, max_passages: int = 80):
    url = (url or "").strip()
    if not url:
        return []

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http/https URLs are supported.")

    headers = {"User-Agent": USER_AGENT}
    r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside"]):
        tag.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else parsed.netloc

    blocks = []
    for tag in soup.find_all(["p", "li"]):
        text = re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip()
        if len(text) >= 40:
            blocks.append(text)
        if len(blocks) >= max_passages:
            break

    return [
        {
            "source": parsed.netloc,
            "title": title,
            "url": url,
            "passage": p,
            "position": i,
        }
        for i, p in enumerate(blocks)
    ]
