"""FastAPI service for the Kidaw'ida <-> Kiswahili translator.

Run locally:   uvicorn app.main:app --reload --port 7860
Interactive API docs: http://localhost:7860/docs
"""
from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator

from .translator import LANGUAGES, Translator, from_env, split_sentences

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"
DATA_DIR = Path(__file__).resolve().parent / "data"
MAX_CHARS = 1000      # per text
MAX_BATCH = 32        # texts per batch request

translator: Translator = from_env()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load in a background thread so the page and /api/health respond while the
    # 600M-parameter model (≈2.4 GB in fp32) is still loading.
    threading.Thread(target=translator.load, daemon=True).start()
    yield


app = FastAPI(
    title="Kidaw'ida ↔ Kiswahili Translator API",
    description="Machine translation between Kidaw'ida (Taita, ISO 639-3 `dav`) and Kiswahili, "
                "using NLLB-200-distilled-600M fine-tuned on the Lacuna Fund / thinkKenya parallel corpus.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

Lang = Literal["swa", "dav"]


class TranslateRequest(BaseModel):
    text: str = Field(..., description="Text to translate (several sentences are fine)",
                      examples=["Unaenda wapi kesho asubuhi?"])
    source: Lang = Field("swa", description="Source language: 'swa' (Kiswahili) or 'dav' (Kidaw'ida)")
    target: Lang = Field("dav", description="Target language")
    beams: int = Field(4, ge=1, le=5, description="Beam size: 1 = fastest, 4 = what the model was evaluated with")

    @field_validator("text")
    @classmethod
    def check_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text is empty")
        if len(v) > MAX_CHARS:
            raise ValueError(f"text is longer than {MAX_CHARS} characters")
        return v

    @model_validator(mode="after")
    def check_direction(self):
        if self.source == self.target:
            raise ValueError("source and target must be different languages")
        return self


class TranslateResponse(BaseModel):
    translation: str
    source: Lang
    target: Lang
    sentences: int
    time_ms: int


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=MAX_BATCH)
    source: Lang = "swa"
    target: Lang = "dav"
    beams: int = Field(4, ge=1, le=5)

    @field_validator("texts")
    @classmethod
    def check_texts(cls, v: list[str]) -> list[str]:
        for t in v:
            if len(t) > MAX_CHARS:
                raise ValueError(f"each text must be at most {MAX_CHARS} characters")
        return v

    @model_validator(mode="after")
    def check_direction(self):
        if self.source == self.target:
            raise ValueError("source and target must be different languages")
        return self


class BatchResponse(BaseModel):
    translations: list[str]
    source: Lang
    target: Lang
    time_ms: int


def _require_ready() -> None:
    if translator.status == "loading" or translator.status == "not_loaded":
        raise HTTPException(503, detail="The model is still loading. Try again in a minute.")
    if translator.status == "error":
        raise HTTPException(500, detail=f"The model failed to load: {translator.error}")


# ---------------------------------------------------------------------------- API
@app.get("/api/health", tags=["service"])
def health():
    """Model status: `loading`, `ready` or `error`."""
    return {"status": translator.status, "model": translator.model_id, "device": translator.device,
            "load_seconds": translator.load_seconds, "error": translator.error}


@app.get("/api/languages", tags=["service"])
def languages():
    return list(LANGUAGES.values())


@app.post("/api/translate", response_model=TranslateResponse, tags=["translation"])
def translate(req: TranslateRequest):
    """Translate one text. Multi-sentence input is split into sentences, translated as a batch and re-joined."""
    _require_ready()
    t0 = time.perf_counter()
    try:
        out = translator.translate_text(req.text, req.source, req.target, req.beams)
    except Exception as exc:
        log.exception("translation failed")
        raise HTTPException(500, detail=f"Translation failed: {exc}") from exc
    n = sum(len(p) for p in split_sentences(req.text))
    return TranslateResponse(translation=out, source=req.source, target=req.target, sentences=n,
                             time_ms=int((time.perf_counter() - t0) * 1000))


@app.post("/api/translate/batch", response_model=BatchResponse, tags=["translation"])
def translate_batch(req: BatchRequest):
    """Translate up to 32 texts in one call."""
    _require_ready()
    t0 = time.perf_counter()
    outs = [translator.translate_text(t, req.source, req.target, req.beams) if t.strip() else "" for t in req.texts]
    return BatchResponse(translations=outs, source=req.source, target=req.target,
                         time_ms=int((time.perf_counter() - t0) * 1000))


@app.get("/api/examples", tags=["content"])
def examples():
    """Example sentences from the test set, for the frontend."""
    return json.loads((DATA_DIR / "examples.json").read_text())


@app.get("/api/model-info", tags=["content"])
def model_info():
    """Evaluation results and known limitations of the deployed model."""
    return json.loads((DATA_DIR / "model_info.json").read_text())


# ------------------------------------------------------------------------ frontend
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
