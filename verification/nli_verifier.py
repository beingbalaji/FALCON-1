from functools import lru_cache

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import NLI_MODEL_NAME


def _softmax(values):
    tensor = torch.tensor(values, dtype=torch.float32)
    return torch.softmax(tensor, dim=-1).tolist()


@lru_cache(maxsize=1)
def load_nli_model():
    """
    Load and cache the NLI tokenizer and model.
    The model is loaded only once while Flask is running.
    """
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)

    model.eval()

    return tokenizer, model


def _label_map(model):
    """
    Determine which output index corresponds to:
    - entailment
    - contradiction
    - neutral
    """

    id2label = {
        int(k): str(v).lower()
        for k, v in model.config.id2label.items()
    }

    out = {}

    for idx, name in id2label.items():

        if "entail" in name:
            out["entailment"] = idx

        elif "contrad" in name:
            out["contradiction"] = idx

        elif "neutral" in name:
            out["neutral"] = idx

    # Fallback for checkpoints whose config does not name the labels
    # (standard MNLI order: 0 = entailment, 1 = neutral, 2 = contradiction).
    if len(out) != 3:
        out = {
            "entailment": 0,
            "neutral": 1,
            "contradiction": 2,
        }

    return out


def verify_claim_with_evidence(claim: str, evidence: str):
    """
    Compare a factual claim against supplied evidence.

    Parameters
    ----------
    claim : str
        Statement that needs verification.

    evidence : str
        Text passage used as evidence.

    Returns
    -------
    dict
        Verification label, confidence and probabilities.
    """

    claim = (claim or "").strip()
    evidence = (evidence or "").strip()

    if not claim:
        raise ValueError("Claim cannot be empty.")

    if not evidence:
        raise ValueError("Evidence cannot be empty.")

    tokenizer, model = load_nli_model()

    mapping = _label_map(model)

    # For NLI:
    # Premise = Evidence
    # Hypothesis = Claim
    encoded = tokenizer(
        evidence,
        claim,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )

    with torch.no_grad():

        outputs = model(**encoded)

        logits = outputs.logits[0].tolist()

    probabilities = _softmax(logits)

    entailment = float(
        probabilities[mapping["entailment"]]
    )

    contradiction = float(
        probabilities[mapping["contradiction"]]
    )

    neutral = float(
        probabilities[mapping["neutral"]]
    )

    scores = {

        "SUPPORTED": entailment,

        "REFUTED": contradiction,

        "NOT ENOUGH INFO": neutral,
    }

    label = max(
        scores,
        key=scores.get
    )

    confidence = scores[label]

    return {

        "label": label,

        "confidence": round(
            confidence,
            4
        ),

        "scores": {

            "SUPPORTED": round(
                entailment,
                4
            ),

            "REFUTED": round(
                contradiction,
                4
            ),

            "NOT ENOUGH INFO": round(
                neutral,
                4
            ),
        },

        "model": NLI_MODEL_NAME,
    }


if __name__ == "__main__":

    print("\nFALCON NLI Verification Test\n")

    test_claim = (
        "The Eiffel Tower is located in Paris."
    )

    test_evidence = (
        "The Eiffel Tower is a wrought-iron landmark "
        "located on the Champ de Mars in Paris, France."
    )

    result = verify_claim_with_evidence(
        test_claim,
        test_evidence
    )

    print("CLAIM:")
    print(test_claim)

    print("\nEVIDENCE:")
    print(test_evidence)

    print("\nRESULT:")
    print(result["label"])

    print("\nCONFIDENCE:")
    print(
        f'{result["confidence"] * 100:.2f}%'
    )

    print("\nCLASS PROBABILITIES:")

    for label, probability in result["scores"].items():

        print(
            f"{label}: "
            f"{probability * 100:.2f}%"
        )