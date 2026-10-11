import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app as falcon_app

def test_health_endpoint():
    falcon_app.app.config["TESTING"] = True
    client = falcon_app.app.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"

def test_empty_api_input():
    falcon_app.app.config["TESTING"] = True
    client = falcon_app.app.test_client()
    response = client.post("/api/analyze", json={"claim": ""})
    assert response.status_code == 400
