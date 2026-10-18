"""API tests. They use the mock backend, so no model download is needed:  pytest -q"""
import os
import time

os.environ["MT_BACKEND"] = "mock"

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.translator import split_sentences  # noqa: E402


def client():
    c = TestClient(app)
    c.__enter__()                      # runs the lifespan (starts model loading)
    for _ in range(50):
        if c.get("/api/health").json()["status"] == "ready":
            break
        time.sleep(0.05)
    return c


def test_health_ready():
    assert client().get("/api/health").json()["status"] == "ready"


def test_translate_swa_to_dav():
    r = client().post("/api/translate", json={"text": "Habari yako?", "source": "swa", "target": "dav"})
    assert r.status_code == 200
    body = r.json()
    assert body["translation"] == "[Kidaw'ida] Habari yako?"
    assert body["sentences"] == 1 and body["source"] == "swa" and body["target"] == "dav"


def test_multi_sentence_and_paragraphs_are_kept():
    r = client().post("/api/translate", json={"text": "Moja. Mbili!\n\nTatu?", "source": "dav", "target": "swa"})
    assert r.json()["translation"] == "[Kiswahili] Moja. [Kiswahili] Mbili!\n\n[Kiswahili] Tatu?"
    assert r.json()["sentences"] == 3


def test_validation_errors():
    c = client()
    assert c.post("/api/translate", json={"text": "  ", "source": "swa", "target": "dav"}).status_code == 422
    assert c.post("/api/translate", json={"text": "x", "source": "swa", "target": "swa"}).status_code == 422
    assert c.post("/api/translate", json={"text": "x", "source": "eng", "target": "dav"}).status_code == 422
    assert c.post("/api/translate", json={"text": "x" * 1001, "source": "swa", "target": "dav"}).status_code == 422
    assert c.post("/api/translate", json={"text": "x", "beams": 9}).status_code == 422


def test_batch():
    r = client().post("/api/translate/batch", json={"texts": ["Moja", "", "Mbili"], "source": "swa", "target": "dav"})
    assert r.status_code == 200
    assert r.json()["translations"] == ["[Kidaw'ida] Moja", "", "[Kidaw'ida] Mbili"]


def test_content_and_frontend():
    c = client()
    assert set(c.get("/api/examples").json()) == {"swa", "dav"}
    assert c.get("/api/model-info").json()["results"]
    assert "Kidaw" in c.get("/").text
    assert c.get("/static/app.js").status_code == 200


def test_split_sentences():
    assert split_sentences("A b. C d?  E\n\nF") == [["A b.", "C d?", "E"], [], ["F"]]
