from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "saved_models"
MODEL_PATH = MODEL_DIR / "falcon_liar_multiclass.joblib"
DB_PATH = BASE_DIR / "database" / "falcon.db"

LIAR_DIR = BASE_DIR / "training" / "datasets" / "liar"
LIAR_TRAIN = LIAR_DIR / "train.tsv"
LIAR_VALID = LIAR_DIR / "valid.tsv"
LIAR_TEST = LIAR_DIR / "test.tsv"
REPORT_DIR = BASE_DIR / "training" / "reports"

APP_TITLE = "FALCON"
MODEL_VERSION = "liar-tfidf-word-char-logreg-v2"

# Pretrained NLI model used for evidence verification.
NLI_MODEL_NAME = "facebook/bart-large-mnli"
NLI_VERSION = "bart-large-mnli-v1"

# Retrieval controls
MAX_WIKI_RESULTS = 5
MAX_ARTICLE_SENTENCES = 80
TOP_EVIDENCE_PASSAGES = 5
REQUEST_TIMEOUT = 12
USER_AGENT = "FALCON-Research-Project/1.0"
