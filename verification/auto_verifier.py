from retrieval.wikipedia_retriever import retrieve_wikipedia_passages
from retrieval.url_retriever import fetch_url_passages
from retrieval.ranker import rank_passages

from verification.nli_verifier import verify_claim_with_evidence
from verification.evidence_aggregator import aggregate_evidence

from config import TOP_EVIDENCE_PASSAGES


def verify_claim_automatically(claim: str, source_url: str = ""):
    claim = (claim or "").strip()

    if not claim:
        raise ValueError("Claim cannot be empty.")

    passages = []

    try:
        passages.extend(retrieve_wikipedia_passages(claim))
        wiki_error = None
    except Exception as exc:
        wiki_error = str(exc)

    url_error = None
    if source_url:
        try:
            passages.extend(fetch_url_passages(source_url))
        except Exception as exc:
            url_error = str(exc)

    ranked = rank_passages(
        claim,
        passages,
        top_k=TOP_EVIDENCE_PASSAGES,
    )

    checked = []

    for item in ranked:
        try:
            nli = verify_claim_with_evidence(claim, item["passage"])
        except Exception as exc:
            item["nli_error"] = str(exc)
            continue

        checked.append({
            **item,
            "label": nli["label"],
            "confidence": nli["confidence"],
            "nli_scores": nli["scores"],
        })

    aggregate = aggregate_evidence(checked)

    return {
        "claim": claim,
        "final": aggregate,
        "evidence": checked,
        "retrieval_count": len(passages),
        "ranked_count": len(ranked),
        "wiki_error": wiki_error,
        "url_error": url_error,
    }
