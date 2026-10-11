import re
from typing import List, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "in", "on", "at", "to", "of", "for", "and", "or",
    "with", "by", "from", "that", "this", "it", "as"
}


def _tokens(text: str):
    return {
        t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
        if t not in STOPWORDS and len(t) > 1
    }


def _normalized(text: str):
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def rank_passages(claim: str, passages: List[Dict], top_k: int = 5) -> List[Dict]:
    claim = (claim or "").strip()
    if not claim or not passages:
        return []

    texts = [claim] + [
        f'{p.get("title", "")} {p.get("passage", "")}'
        for p in passages
    ]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(texts)
    tfidf_scores = cosine_similarity(matrix[0:1], matrix[1:])[0]

    claim_tokens = _tokens(claim)
    claim_norm = _normalized(claim)

    ranked = []

    for p, tfidf in zip(passages, tfidf_scores):
        passage = p.get("passage", "")
        title = p.get("title", "")
        page_priority = float(p.get("page_priority", 0.0))

        passage_tokens = _tokens(passage)
        title_tokens = _tokens(title)

        if claim_tokens:
            passage_overlap = len(claim_tokens & passage_tokens) / len(claim_tokens)
            title_overlap = len(claim_tokens & title_tokens) / len(claim_tokens)
        else:
            passage_overlap = 0.0
            title_overlap = 0.0

        position = int(p.get("position", 999))
        lead_bonus = 1.0 / (1.0 + max(position, 0))
        exact_bonus = 1.0 if claim_norm and claim_norm in _normalized(passage) else 0.0

        score = (
            0.46 * float(tfidf)
            + 0.22 * passage_overlap
            + 0.10 * title_overlap
            + 0.07 * lead_bonus
            + 0.03 * exact_bonus
            + 0.12 * page_priority
        )

        if score < 0.06:
            continue

        item = dict(p)
        item["retrieval_score"] = round(float(score), 6)
        item["tfidf_score"] = round(float(tfidf), 6)
        item["token_overlap"] = round(float(passage_overlap), 6)
        ranked.append(item)

    ranked.sort(key=lambda x: x["retrieval_score"], reverse=True)

    # Avoid one noisy article filling every slot.
    selected = []
    per_title = {}

    for item in ranked:
        title = item.get("title", "")
        count = per_title.get(title, 0)

        # Allow up to 3 passages from the best article, max 2 from others.
        limit = 3 if item.get("page_priority", 0.0) >= 0.8 else 2

        if count >= limit:
            continue

        selected.append(item)
        per_title[title] = count + 1

        if len(selected) >= top_k:
            break

    return selected
