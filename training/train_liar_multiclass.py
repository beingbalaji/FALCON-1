from pathlib import Path
import json
import sys

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    LIAR_TRAIN,
    LIAR_VALID,
    LIAR_TEST,
    MODEL_PATH,
    MODEL_DIR,
    REPORT_DIR,
)

COLUMNS = [
    "id",
    "label",
    "statement",
    "subject",
    "speaker",
    "speaker_job_title",
    "state_info",
    "party_affiliation",
    "barely_true_count",
    "false_count",
    "half_true_count",
    "mostly_true_count",
    "pants_fire_count",
    "context",
]

LABELS = [
    "pants-fire",
    "false",
    "barely-true",
    "half-true",
    "mostly-true",
    "true",
]

def load_liar(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing LIAR file: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=COLUMNS,
        dtype=str,
        keep_default_na=False,
        quoting=3,
        on_bad_lines="warn",
    )

    df = df[["statement", "label"]].copy()
    df["statement"] = df["statement"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()

    df = df[
        (df["statement"] != "") &
        (df["label"].isin(LABELS))
    ].drop_duplicates(subset=["statement", "label"])

    return df.reset_index(drop=True)

def build_pipeline():
    features = FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.98,
                sublinear_tf=True,
                max_features=45000,
            ),
        ),
        (
            "char",
            TfidfVectorizer(
                analyzer="char_wb",
                lowercase=True,
                ngram_range=(3, 5),
                min_df=2,
                sublinear_tf=True,
                max_features=35000,
            ),
        ),
    ])

    clf = LogisticRegression(
        max_iter=2500,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )

    return Pipeline([
        ("features", features),
        ("clf", clf),
    ])

def metric_dict(y_true, y_pred):
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_precision": round(float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ), 6),
        "macro_recall": round(float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ), 6),
        "macro_f1": round(float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ), 6),
        "weighted_f1": round(float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ), 6),
    }

def save_confusion_matrix(y_true, y_pred, out_path: Path, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(10, 8))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    display.plot(ax=ax, xticks_rotation=45, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

def print_metrics(name, metrics):
    print(f"\n{name}")
    print("-" * len(name))
    for k, v in metrics.items():
        print(f"{k:>16}: {v:.4f}")

def main():
    print("Loading LIAR dataset...")
    train_df = load_liar(LIAR_TRAIN)
    valid_df = load_liar(LIAR_VALID)
    test_df = load_liar(LIAR_TEST)

    print(f"Train rows : {len(train_df)}")
    print(f"Valid rows : {len(valid_df)}")
    print(f"Test rows  : {len(test_df)}")

    print("\nTraining validation model...")
    validation_model = build_pipeline()
    validation_model.fit(train_df["statement"], train_df["label"])

    valid_pred = validation_model.predict(valid_df["statement"])
    valid_metrics = metric_dict(valid_df["label"], valid_pred)

    print_metrics("VALIDATION METRICS", valid_metrics)
    print("\nValidation classification report:")
    print(classification_report(
        valid_df["label"],
        valid_pred,
        labels=LABELS,
        digits=4,
        zero_division=0,
    ))

    print("\nRetraining final model on train + validation...")
    final_train = pd.concat([train_df, valid_df], ignore_index=True)

    final_model = build_pipeline()
    final_model.fit(final_train["statement"], final_train["label"])

    print("Evaluating untouched test set...")
    test_pred = final_model.predict(test_df["statement"])
    test_metrics = metric_dict(test_df["label"], test_pred)

    print_metrics("TEST METRICS", test_metrics)
    print("\nTest classification report:")
    report_text = classification_report(
        test_df["label"],
        test_pred,
        labels=LABELS,
        digits=4,
        zero_division=0,
    )
    print(report_text)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_model, MODEL_PATH)

    metrics_payload = {
        "dataset": "LIAR",
        "task": "6-class truthfulness classification",
        "labels": LABELS,
        "rows": {
            "train": len(train_df),
            "valid": len(valid_df),
            "test": len(test_df),
            "final_training": len(final_train),
        },
        "validation": valid_metrics,
        "test": test_metrics,
    }

    (REPORT_DIR / "liar_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )

    report_df = pd.DataFrame(
        classification_report(
            test_df["label"],
            test_pred,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    report_df.to_csv(REPORT_DIR / "liar_classification_report.csv")

    cm = confusion_matrix(test_df["label"], test_pred, labels=LABELS)
    cm_df = pd.DataFrame(cm, index=LABELS, columns=LABELS)
    cm_df.to_csv(REPORT_DIR / "liar_confusion_matrix.csv")

    save_confusion_matrix(
        test_df["label"],
        test_pred,
        REPORT_DIR / "liar_confusion_matrix.png",
        "FALCON — LIAR Test Confusion Matrix",
    )

    print(f"\nSaved final model to:\n{MODEL_PATH}")
    print(f"\nSaved reports to:\n{REPORT_DIR}")
    print("\nTraining complete.")

if __name__ == "__main__":
    main()
