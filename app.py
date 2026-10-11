from datetime import datetime, timezone
import json
import sqlite3

from flask import Flask, jsonify, render_template, request

from config import APP_TITLE, DB_PATH, MODEL_PATH, MODEL_VERSION
from models.classifier import FalconClassifier
from verification.auto_verifier import verify_claim_automatically

app = Flask(__name__)
classifier = FalconClassifier(MODEL_PATH)


def get_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses_v4 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim TEXT NOT NULL,
                source_url TEXT,
                linguistic_label TEXT,
                linguistic_confidence REAL,
                final_label TEXT,
                final_confidence REAL,
                evidence_json TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_analysis(claim, source_url, linguistic, verification):
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO analyses_v4
            (claim, source_url, linguistic_label, linguistic_confidence,
             final_label, final_confidence, evidence_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                claim,
                source_url,
                linguistic.get("label") if linguistic else None,
                linguistic.get("confidence") if linguistic else None,
                verification["final"]["final_label"],
                verification["final"]["confidence"],
                json.dumps(verification["evidence"]),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", title=APP_TITLE)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_version": MODEL_VERSION})


@app.route("/analyze", methods=["POST"])
def analyze():
    claim = request.form.get("claim", "").strip()
    source_url = request.form.get("source_url", "").strip()

    if not claim:
        return render_template(
            "index.html",
            title=APP_TITLE,
            error="Please enter a claim.",
        ), 400

    linguistic = None
    linguistic_error = None

    try:
        linguistic = classifier.predict(claim)
    except Exception as exc:
        linguistic_error = str(exc)

    try:
        verification = verify_claim_automatically(
            claim,
            source_url=source_url,
        )
    except Exception as exc:
        return render_template(
            "index.html",
            title=APP_TITLE,
            error=f"Automatic verification failed: {exc}",
        ), 500

    save_analysis(claim, source_url, linguistic, verification)

    return render_template(
        "result.html",
        title=APP_TITLE,
        claim=claim,
        source_url=source_url,
        linguistic=linguistic,
        linguistic_error=linguistic_error,
        verification=verification,
        model_version=MODEL_VERSION,
    )


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    payload = request.get_json(silent=True) or {}
    claim = str(payload.get("claim", "")).strip()
    source_url = str(payload.get("source_url", "")).strip()

    if not claim:
        return jsonify({"error": "claim is required"}), 400

    linguistic = None
    try:
        linguistic = classifier.predict(claim)
    except Exception:
        pass

    verification = verify_claim_automatically(
        claim,
        source_url=source_url,
    )

    return jsonify({
        "claim": claim,
        "linguistic": linguistic,
        "verification": verification,
    })


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
