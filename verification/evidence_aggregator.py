def aggregate_evidence(results):
    """
    Aggregate soft NLI probabilities rather than only passage-level hard labels.

    This prevents a few weak or irrelevant passages from overriding a strong,
    directly relevant supporting passage.
    """

    if not results:
        return {
            "final_label": "NOT ENOUGH INFO",
            "confidence": 1.0,
            "support_score": 0.0,
            "refute_score": 0.0,
            "nei_score": 1.0,
            "decision_note": "No sufficiently relevant evidence was retrieved.",
        }

    support_total = 0.0
    refute_total = 0.0
    nei_total = 0.0
    total_weight = 0.0

    for rank, item in enumerate(results):
        retrieval = max(float(item.get("retrieval_score", 0.0)), 0.0)

        # Strongly prioritize highly relevant passages.
        relevance_weight = retrieval ** 2

        # Slight rank discount so top evidence matters more.
        rank_weight = 1.0 / (1.0 + 0.20 * rank)

        weight = relevance_weight * rank_weight
        if weight <= 0:
            continue

        scores = item.get("nli_scores", {})
        support = float(scores.get("SUPPORTED", 0.0))
        refute = float(scores.get("REFUTED", 0.0))
        nei = float(scores.get("NOT ENOUGH INFO", 0.0))

        support_total += weight * support
        refute_total += weight * refute
        nei_total += weight * nei
        total_weight += weight

    if total_weight <= 0:
        return {
            "final_label": "NOT ENOUGH INFO",
            "confidence": 1.0,
            "support_score": 0.0,
            "refute_score": 0.0,
            "nei_score": 1.0,
            "decision_note": "Retrieved evidence did not pass relevance weighting.",
        }

    support = support_total / total_weight
    refute = refute_total / total_weight
    nei = nei_total / total_weight

    # Guardrail: if support and refute are close, avoid a forced verdict.
    evidence_margin = abs(support - refute)
    best_evidence = max(support, refute)

    if best_evidence < 0.55 or evidence_margin < 0.15:
        final_label = "NOT ENOUGH INFO"
        confidence = max(nei, 1.0 - best_evidence)
        note = "Evidence was mixed or not decisive enough for a supported/refuted verdict."
    elif support > refute:
        final_label = "SUPPORTED"
        confidence = support
        note = "Relevant retrieved evidence predominantly supports the claim."
    else:
        final_label = "REFUTED"
        confidence = refute
        note = "Relevant retrieved evidence predominantly contradicts the claim."

    return {
        "final_label": final_label,
        "confidence": round(float(confidence), 4),
        "support_score": round(float(support), 4),
        "refute_score": round(float(refute), 4),
        "nei_score": round(float(nei), 4),
        "decision_note": note,
    }
