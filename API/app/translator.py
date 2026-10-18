"""Kidaw'ida <-> Kiswahili translation with the fine-tuned NLLB-200 model (experiment E6).

The model expects exactly the format used in training (notebook 03):
    encoder input : [src_lang_tag] subword tokens </s>
    decoder start : forced to the target language tag
Kidaw'ida has its own added tag, ``dav_Latn``; Kiswahili uses NLLB's ``swh_Latn``.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path

log = logging.getLogger("translator")

LANGUAGES = {
    "swa": {"code": "swa", "name": "Kiswahili", "nllb": "swh_Latn"},
    "dav": {"code": "dav", "name": "Kidaw'ida", "nllb": "dav_Latn"},
}
DEFAULT_MAX_LEN = 64  # subword tokens; training sentences are at most ~40 tokens

# Split on sentence-final punctuation followed by whitespace. The model was trained
# on short single sentences (mean 4.5 words), so long inputs are translated
# sentence by sentence and joined back together.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[list[str]]:
    """Return paragraphs -> sentences, keeping the paragraph structure of the input."""
    paragraphs = []
    for line in text.replace("\r\n", "\n").split("\n"):
        line = line.strip()
        paragraphs.append([s for s in _SENT_SPLIT.split(line) if s.strip()] if line else [])
    return paragraphs


class Translator:
    """Loads the model once and translates batches of sentences.

    ``model_id`` is a local folder or a Hugging Face Hub repo id.
    With ``backend="mock"`` no model is loaded (used for tests and frontend work).
    """

    def __init__(self, model_id: str, backend: str = "nllb", device: str | None = None,
                 num_threads: int | None = None):
        self.model_id = model_id
        self.backend = backend
        self.requested_device = device
        self.num_threads = num_threads
        self.status = "not_loaded"      # not_loaded | loading | ready | error
        self.error: str | None = None
        self.device = "cpu"
        self.load_seconds: float | None = None
        self.lang_tags = {k: v["nllb"] for k, v in LANGUAGES.items()}
        self.max_len = DEFAULT_MAX_LEN
        self._lock = threading.Lock()   # one generation at a time: CPU inference is the bottleneck
        self._tok = self._model = None
        self._cache: OrderedDict[tuple, str] = OrderedDict()
        self.cache_size = 4096
        self.batch_size = 16

    # ------------------------------------------------------------------ loading
    def load(self) -> None:
        self.status, t0 = "loading", time.time()
        try:
            if self.backend == "mock":
                self.device = "none"
            else:
                self._load_nllb()
            self.load_seconds = round(time.time() - t0, 1)
            self.status = "ready"
            log.info("Model ready (%s, %ss)", self.device, self.load_seconds)
        except Exception as exc:  # surfaced through /api/health
            log.exception("Model failed to load")
            self.status, self.error = "error", f"{type(exc).__name__}: {exc}"

    def _load_nllb(self) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if self.num_threads:
            torch.set_num_threads(self.num_threads)
        self.device = self.requested_device or ("cuda" if torch.cuda.is_available() else "cpu")

        cfg = self._read_mt_config()
        self.lang_tags.update({k: v for k, v in cfg.get("lang_codes", {}).items() if k in LANGUAGES})
        self.max_len = int(cfg.get("max_len", DEFAULT_MAX_LEN))

        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
        # Weights are stored in fp16; CPUs need fp32 for speed and numerical support.
        model = model.half() if self.device == "cuda" else model.float()
        self._model = model.to(self.device).eval()

        unk = self._tok.unk_token_id
        for lang, tag in self.lang_tags.items():
            if self._tok.convert_tokens_to_ids(tag) in (None, unk):
                raise ValueError(f"Language tag {tag!r} for {lang} is missing from the tokenizer")

    def _read_mt_config(self) -> dict:
        local = Path(self.model_id) / "mt_config.json"
        if local.exists():
            return json.loads(local.read_text())
        try:
            from huggingface_hub import hf_hub_download
            return json.loads(Path(hf_hub_download(self.model_id, "mt_config.json")).read_text())
        except Exception:
            log.warning("mt_config.json not found for %s; using default language tags", self.model_id)
            return {}

    # -------------------------------------------------------------- translating
    def translate_text(self, text: str, source: str, target: str, beams: int = 4) -> str:
        """Translate free text: split into sentences, translate as one batch, rebuild paragraphs."""
        paragraphs = split_sentences(text)
        flat = [s for p in paragraphs for s in p]
        if not flat:
            return ""
        out = iter(self.translate_sentences(tuple(flat), source, target, beams))
        return "\n".join(" ".join(next(out) for _ in p) for p in paragraphs)

    def translate_sentences(self, sentences: tuple[str, ...], source: str, target: str,
                            beams: int = 4) -> list[str]:
        """Translate sentences in batches; sentences seen before come from an LRU cache."""
        keys = [(s, source, target, beams) for s in sentences]
        missing = list(dict.fromkeys(k[0] for k in keys if k not in self._cache))
        for i in range(0, len(missing), self.batch_size):
            chunk = missing[i:i + self.batch_size]
            for src, out in zip(chunk, self._generate(chunk, source, target, beams)):
                self._cache[(src, source, target, beams)] = out
        result = []
        for k in keys:
            self._cache.move_to_end(k)
            result.append(self._cache[k])
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return result

    def _generate(self, sentences: list[str], source: str, target: str, beams: int) -> list[str]:
        if self.status != "ready":
            raise RuntimeError(f"model is {self.status}")
        if self.backend == "mock":
            return [f"[{LANGUAGES[target]['name']}] {s}" for s in sentences]

        import torch
        tok, model = self._tok, self._model
        src_id = tok.convert_tokens_to_ids(self.lang_tags[source])
        tgt_id = tok.convert_tokens_to_ids(self.lang_tags[target])
        ids = tok(list(sentences), add_special_tokens=False)["input_ids"]
        ids = [[src_id] + x[: self.max_len - 2] + [tok.eos_token_id] for x in ids]
        width = max(map(len, ids))
        inp = torch.tensor([x + [tok.pad_token_id] * (width - len(x)) for x in ids], device=self.device)
        with self._lock, torch.inference_mode():
            gen = model.generate(input_ids=inp, attention_mask=(inp != tok.pad_token_id).long(),
                                 forced_bos_token_id=tgt_id, num_beams=beams,
                                 max_new_tokens=self.max_len)
        return [t.strip() for t in tok.batch_decode(gen, skip_special_tokens=True)]


def from_env() -> Translator:
    """Build the translator from environment variables (see README)."""
    return Translator(
        model_id=os.getenv("MODEL_ID", "models/nllb_dav_swa_final"),
        backend=os.getenv("MT_BACKEND", "nllb"),
        device=os.getenv("DEVICE") or None,
        num_threads=int(os.getenv("NUM_THREADS")) if os.getenv("NUM_THREADS") else None,
    )
