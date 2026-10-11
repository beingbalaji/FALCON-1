from pathlib import Path
import sys
import time

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

# ---------------------------------------------------------
# Add project root to Python path
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from verification.auto_verifier import verify_claim_automatically


# ---------------------------------------------------------
# File paths
# ---------------------------------------------------------

DATASET_PATH = PROJECT_ROOT / "evaluation" / "falcon_eval.csv"
RESULT_PATH = PROJECT_ROOT / "evaluation" / "falcon_eval_results.csv"
CONFUSION_PATH = PROJECT_ROOT / "evaluation" / "falcon_confusion_matrix.csv"


# ---------------------------------------------------------
# FALCON output classes
# ---------------------------------------------------------

LABELS = [
    "SUPPORTED",
    "REFUTED",
    "NOT ENOUGH INFO",
]


# ---------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------

def main():

    print("\n" + "=" * 60)
    print("FALCON END-TO-END EVALUATION")
    print("=" * 60)

    # -----------------------------------------------------
    # Load CSV
    # -----------------------------------------------------

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"\nDataset not found:\n{DATASET_PATH}\n"
            "\nCreate evaluation/falcon_eval.csv first."
        )

    df = pd.read_csv(DATASET_PATH)

    required_columns = {"claim", "label"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "falcon_eval.csv must contain these columns:\n"
            "claim,label"
        )

    print(f"\nDataset loaded successfully.")
    print(f"Total claims: {len(df)}")

    # -----------------------------------------------------
    # Storage
    # -----------------------------------------------------

    expected_labels = []
    predicted_labels = []

    results = []

    total = len(df)

    # -----------------------------------------------------
    # Evaluate each claim
    # -----------------------------------------------------

    for index, row in df.iterrows():

        claim = str(row["claim"]).strip()

        expected = (
            str(row["label"])
            .strip()
            .upper()
        )

        print("\n" + "-" * 60)

        print(
            f"Claim {index + 1}/{total}"
        )

        print(
            f"\nCLAIM:\n{claim}"
        )

        print(
            f"\nEXPECTED:\n{expected}"
        )

        start_time = time.time()

        try:

            # Run complete FALCON pipeline
            output = verify_claim_automatically(
                claim
            )

            final_output = output["final"]

            predicted = (
                final_output["final_label"]
                .strip()
                .upper()
            )

            confidence = float(
                final_output.get(
                    "confidence",
                    0.0
                )
            )

            support_score = float(
                final_output.get(
                    "support_score",
                    0.0
                )
            )

            refute_score = float(
                final_output.get(
                    "refute_score",
                    0.0
                )
            )

            nei_score = float(
                final_output.get(
                    "nei_score",
                    0.0
                )
            )

            evidence = output.get(
                "evidence",
                []
            )

            evidence_count = len(evidence)

            retrieval_count = int(
                output.get(
                    "retrieval_count",
                    0
                )
            )

            top_evidence = ""

            top_source = ""

            top_retrieval_score = 0.0

            if evidence:

                first = evidence[0]

                top_evidence = str(
                    first.get(
                        "passage",
                        ""
                    )
                )

                top_source = str(
                    first.get(
                        "title",
                        ""
                    )
                )

                top_retrieval_score = float(
                    first.get(
                        "retrieval_score",
                        0.0
                    )
                )

            error = ""

        except Exception as exc:

            predicted = "ERROR"

            confidence = 0.0

            support_score = 0.0
            refute_score = 0.0
            nei_score = 0.0

            evidence_count = 0
            retrieval_count = 0

            top_evidence = ""
            top_source = ""
            top_retrieval_score = 0.0

            error = str(exc)

            print(
                f"\nERROR:\n{error}"
            )

        elapsed = (
            time.time()
            - start_time
        )

        correct = (
            expected == predicted
        )

        print(
            f"\nPREDICTED:\n{predicted}"
        )

        print(
            f"\nCONFIDENCE:\n"
            f"{confidence * 100:.2f}%"
        )

        print(
            f"\nRESULT:\n"
            f"{'CORRECT' if correct else 'WRONG'}"
        )

        print(
            f"\nTIME:\n"
            f"{elapsed:.2f} seconds"
        )

        # Store labels for metrics
        expected_labels.append(
            expected
        )

        predicted_labels.append(
            predicted
        )

        # Store detailed result
        results.append({

            "claim": claim,

            "expected": expected,

            "predicted": predicted,

            "correct": correct,

            "confidence": confidence,

            "support_score": support_score,

            "refute_score": refute_score,

            "nei_score": nei_score,

            "retrieval_count": retrieval_count,

            "evidence_count": evidence_count,

            "top_source": top_source,

            "top_retrieval_score":
                top_retrieval_score,

            "top_evidence":
                top_evidence,

            "time_seconds":
                round(elapsed, 3),

            "error":
                error,
        })

        # Small delay between external retrieval requests
        time.sleep(0.5)

    # -----------------------------------------------------
    # Save detailed output
    # -----------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        RESULT_PATH,
        index=False
    )

    # -----------------------------------------------------
    # Remove errors before calculating metrics
    # -----------------------------------------------------

    valid_pairs = [

        (expected, predicted)

        for expected, predicted
        in zip(
            expected_labels,
            predicted_labels
        )

        if predicted in LABELS
    ]

    if not valid_pairs:

        print(
            "\nNo valid FALCON predictions "
            "were produced."
        )

        return

    y_true = [
        pair[0]
        for pair in valid_pairs
    ]

    y_pred = [
        pair[1]
        for pair in valid_pairs
    ]

    # -----------------------------------------------------
    # Calculate metrics
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _
    ) = precision_recall_fscore_support(

        y_true,
        y_pred,

        labels=LABELS,

        average="macro",

        zero_division=0,
    )

    (
        weighted_precision,
        weighted_recall,
        weighted_f1,
        _
    ) = precision_recall_fscore_support(

        y_true,
        y_pred,

        labels=LABELS,

        average="weighted",

        zero_division=0,
    )

    # -----------------------------------------------------
    # Print final results
    # -----------------------------------------------------

    print("\n\n" + "=" * 60)

    print(
        "FINAL FALCON RESULTS"
    )

    print("=" * 60)

    print(
        f"\nTotal claims evaluated : "
        f"{len(df)}"
    )

    print(
        f"Valid predictions      : "
        f"{len(y_pred)}"
    )

    errors = (
        len(df)
        - len(y_pred)
    )

    print(
        f"Errors                 : "
        f"{errors}"
    )

    print(
        f"\nAccuracy               : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Macro Precision        : "
        f"{macro_precision * 100:.2f}%"
    )

    print(
        f"Macro Recall           : "
        f"{macro_recall * 100:.2f}%"
    )

    print(
        f"Macro F1               : "
        f"{macro_f1 * 100:.2f}%"
    )

    print(
        f"\nWeighted Precision     : "
        f"{weighted_precision * 100:.2f}%"
    )

    print(
        f"Weighted Recall        : "
        f"{weighted_recall * 100:.2f}%"
    )

    print(
        f"Weighted F1            : "
        f"{weighted_f1 * 100:.2f}%"
    )

    # -----------------------------------------------------
    # Classification report
    # -----------------------------------------------------

    print(
        "\n\nCLASSIFICATION REPORT"
    )

    print("-" * 60)

    print(
        classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            digits=4,
            zero_division=0,
        )
    )

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS
    )

    confusion_df = pd.DataFrame(

        matrix,

        index=[
            "Actual_SUPPORTED",
            "Actual_REFUTED",
            "Actual_NEI",
        ],

        columns=[
            "Pred_SUPPORTED",
            "Pred_REFUTED",
            "Pred_NEI",
        ],
    )

    print(
        "\nCONFUSION MATRIX"
    )

    print("-" * 60)

    print(
        confusion_df
    )

    confusion_df.to_csv(
        CONFUSION_PATH
    )

    # -----------------------------------------------------
    # Count correct predictions
    # -----------------------------------------------------

    correct_count = sum(
        1
        for e, p
        in zip(
            y_true,
            y_pred
        )
        if e == p
    )

    wrong_count = (
        len(y_pred)
        - correct_count
    )

    print(
        f"\nCorrect predictions : "
        f"{correct_count}"
    )

    print(
        f"Wrong predictions   : "
        f"{wrong_count}"
    )

    print(
        "\nDetailed results saved to:"
    )

    print(
        RESULT_PATH
    )

    print(
        "\nConfusion matrix saved to:"
    )

    print(
        CONFUSION_PATH
    )

    print(
        "\nEvaluation complete."
    )


if __name__ == "__main__":
    main()