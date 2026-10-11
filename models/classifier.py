from pathlib import Path
import joblib

ORDERED_LABELS = [
    "pants-fire",
    "false",
    "barely-true",
    "half-true",
    "mostly-true",
    "true",
]

GROUP_MAP = {
    "pants-fire": "LOW CREDIBILITY",
    "false": "LOW CREDIBILITY",
    "barely-true": "LOW / MIXED CREDIBILITY",
    "half-true": "MIXED CREDIBILITY",
    "mostly-true": "HIGH / MIXED CREDIBILITY",
    "true": "HIGH CREDIBILITY",
}

class FalconClassifier:
    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.pipeline = None

    def load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Run: python training/train_liar_multiclass.py"
            )
        self.pipeline = joblib.load(self.model_path)
        return self

    def predict(self, text: str):
        if self.pipeline is None:
            self.load()

        text = (text or "").strip()
        if not text:
            raise ValueError("Input text cannot be empty.")

        label = str(self.pipeline.predict([text])[0])
        probabilities = self.pipeline.predict_proba([text])[0]
        classes = [str(c) for c in self.pipeline.classes_]
        prob_map = {c: float(p) for c, p in zip(classes, probabilities)}

        ordered_probs = {
            name: round(prob_map.get(name, 0.0), 4)
            for name in ORDERED_LABELS
        }

        return {
            "label": label,
            "credibility_group": GROUP_MAP.get(label, "UNSPECIFIED"),
            "confidence": round(prob_map[label], 4),
            "probabilities": ordered_probs,
        }
