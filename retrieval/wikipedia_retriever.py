import re
from typing import List, Dict

import requests

from config import (
    MAX_WIKI_RESULTS,
    MAX_ARTICLE_SENTENCES,
    REQUEST_TIMEOUT,
    USER_AGENT,
)

WIKI_API = "https://en.wikipedia.org/w/api.php"


def _session():
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _extract_subject_query(claim: str) -> str:
    """
    Heuristic subject extraction for simple factual claims.

    Example:
      "The Eiffel Tower is located in Paris."
      -> "Eiffel Tower"

    This is intentionally conservative and only used to improve retrieval.
    """
    claim = re.sub(r"\s+", " ", (claim or "").strip())

    # Remove a leading determiner.
    cleaned = re.sub(r"^(the|a|an)\s+", "", claim, flags=re.I)

    # Prefer the span before common copular/relational verbs.
    m = re.match(
        r"(.+?)\s+(?:is|are|was|were|has|have|had|became|becomes|"
        r"located|lies|stands|sits|contains|includes|belongs)\b",
        cleaned,
        flags=re.I,
    )
    if m:
        subject = m.group(1).strip(" ,.;:-")
        if 2 <= len(subject) <= 120:
            return subject

    # Fallback: first 6 content-like words.
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'’-]*", cleaned)
    return " ".join(words[:6])


def search_wikipedia(query: str, limit: int = MAX_WIKI_RESULTS) -> List[Dict]:
    query = (query or "").strip()
    if not query:
        return []

    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "format": "json",
        "utf8": 1,
    }

    r = _session().get(WIKI_API, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    payload = r.json()

    results = []
    for item in payload.get("query", {}).get("search", []):
        title = item.get("title", "")
        if not title:
            continue

        results.append({
            "title": title,
            "pageid": item.get("pageid"),
            "snippet": re.sub(r"<.*?>", "", item.get("snippet", "")),
            "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
        })

    return results


def fetch_article_extract(title: str) -> Dict:
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "titles": title,
        "format": "json",
        "utf8": 1,
    }

    r = _session().get(WIKI_API, params=params, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    payload = r.json()

    pages = payload.get("query", {}).get("pages", {})
    if not pages:
        return {}

    page = next(iter(pages.values()))
    page_title = page.get("title", title)
    extract = page.get("extract", "") or ""

    return {
        "title": page_title,
        "extract": extract,
        "url": f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}",
    }


def split_into_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return []

    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)

    cleaned = []
    for s in parts:
        s = s.strip()
        if len(s) < 25:
            continue
        if len(s) > 900:
            s = s[:900]
        cleaned.append(s)

    return cleaned[:MAX_ARTICLE_SENTENCES]


def _page_priority(title: str, subject_query: str, claim: str) -> float:
    """
    Prefer the canonical article and penalize parenthetical replica pages
    unless the parenthetical text also appears in the claim.
    """
    title_norm = _normalize(title)
    subject_norm = _normalize(subject_query)
    claim_norm = _normalize(claim)

    score = 0.0

    if title_norm == subject_norm:
        score += 1.0

    if subject_norm and subject_norm in title_norm:
        score += 0.35

    if "(" in title and ")" in title:
        parenthetical = re.search(r"\((.*?)\)", title)
        parenthetical_text = _normalize(parenthetical.group(1)) if parenthetical else ""

        if parenthetical_text and parenthetical_text not in claim_norm:
            score -= 0.75

    return score


def retrieve_wikipedia_passages(claim: str) -> List[Dict]:
    subject_query = _extract_subject_query(claim)

    # Search the entity/subject first, not only the whole sentence.
    combined_results = []

    seen_titles = set()

    for query in [subject_query, claim]:
        try:
            results = search_wikipedia(query, limit=MAX_WIKI_RESULTS)
        except Exception:
            results = []

        for result in results:
            title = result["title"]
            if title in seen_titles:
                continue
            seen_titles.add(title)

            result["page_priority"] = _page_priority(
                title,
                subject_query,
                claim,
            )
            combined_results.append(result)

    # Canonical title/entity matches first.
    combined_results.sort(
        key=lambda x: x.get("page_priority", 0.0),
        reverse=True,
    )

    # Keep a small focused article set to reduce replica-page noise.
    focused_results = combined_results[:4]

    passages = []

    for result in focused_results:
        try:
            article = fetch_article_extract(result["title"])
        except Exception:
            continue

        page_priority = result.get("page_priority", 0.0)

        for idx, sentence in enumerate(split_into_sentences(article.get("extract", ""))):
            passages.append({
                "source": "Wikipedia",
                "title": article.get("title", result["title"]),
                "url": article.get("url", result["url"]),
                "passage": sentence,
                "position": idx,
                "page_priority": page_priority,
                "subject_query": subject_query,
            })

    return passages
