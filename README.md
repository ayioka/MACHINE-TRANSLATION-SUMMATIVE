---
title: Kidaw'ida Translator
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Kidaw'ida ↔ Kiswahili Translator

A FastAPI service and web frontend for translating between **Kidaw'ida** (Taita, ISO 639-3 `dav`) and **Kiswahili**.
It serves NLLB-200-distilled-600M fine-tuned on both directions (experiment E6 of the project):
**54.4 chrF++** Kiswahili → Kidaw'ida and **57.2 chrF++** Kidaw'ida → Kiswahili on 5,058 held-out sentence pairs.

```
kidawida-translator/
├── app/
│   ├── main.py            FastAPI app: API routes + serves the frontend
│   ├── translator.py      model loading, sentence splitting, batched + cached generation
│   └── data/              example sentences and evaluation results shown on the page
├── static/                frontend: index.html, styles.css, app.js (no build step)
├── tests/test_api.py      API tests (use a mock model, run in under a second)
├── colab/upload_model_to_hub.py   publishes the trained model from Drive to the Hugging Face Hub
├── Dockerfile             container for Hugging Face Spaces or any Docker host
└── requirements.txt
```

## API

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/api/health` | model status: `loading`, `ready` or `error` |
| `POST` | `/api/translate` | translate one text (multi-sentence input is split, batched and re-joined) |
| `POST` | `/api/translate/batch` | translate up to 32 texts |
| `GET`  | `/api/languages` | supported languages |
| `GET`  | `/api/examples` | example sentences |
| `GET`  | `/api/model-info` | evaluation results and limitations |
| `GET`  | `/docs` | interactive OpenAPI documentation |

```bash
curl -X POST http://localhost:7860/api/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Unaenda wapi kesho asubuhi?", "source": "swa", "target": "dav"}'
# {"translation":"Kwaw'eenda hao kesho luma lwa kesho?","source":"swa","target":"dav","sentences":1,"time_ms":...}
```

Request fields: `text` (≤ 1,000 characters), `source` and `target` (`"swa"` or `"dav"`, must differ), `beams` (1–5, default 4).
Invalid input returns `422` with a message; requests while the model is loading return `503`.

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_ID` | `models/nllb_dav_swa_final` | local folder **or** Hugging Face Hub repo id |
| `MT_BACKEND` | `nllb` | `mock` echoes input without loading a model (frontend work, tests) |
| `DEVICE` | auto | `cpu` or `cuda` |
| `NUM_THREADS` | torch default | CPU threads for inference |

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# Option A: use the model from Google Drive — download the folder
#   MyDrive/kidawida-mt/models/nllb_dav_swa_final  into  ./models/nllb_dav_swa_final
uvicorn app.main:app --port 7860

# Option B: use the model from the Hugging Face Hub (after running colab/upload_model_to_hub.py)
MODEL_ID=YOUR_HF_USERNAME/nllb-600m-kidawida-swahili uvicorn app.main:app --port 7860

# Frontend only, no model:
MT_BACKEND=mock uvicorn app.main:app --port 7860

pytest -q
```

Open http://localhost:7860. The first start downloads ≈1.2 GB of weights and loading takes about a minute; the page shows
"Loading the model…" until it is ready. Needs ≈4 GB of free RAM.

## Deploy to Hugging Face Spaces (free CPU)

1. Publish the model: paste `colab/upload_model_to_hub.py` into a Colab cell and run it (needs a **write** token).
2. Create a Space at https://huggingface.co/new-space → SDK **Docker** → template **Blank** → hardware **CPU basic (free)**.
3. Push this folder to the Space:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_HF_USERNAME/kidawida-translator space
   cp -r app static Dockerfile requirements.txt README.md .dockerignore space/
   cd space && git add . && git commit -m "Kidaw'ida translator" && git push
   ```
4. In the Space → Settings → **Variables**, add `MODEL_ID = YOUR_HF_USERNAME/nllb-600m-kidawida-swahili`.
5. The build takes ~5 minutes; the first translation after a cold start waits for the model to load.

On the free 2-vCPU machine a short sentence takes roughly 1–3 s with beam 4. Free Spaces sleep after 48 h without visitors and wake on the next visit.

## How translation works

The model expects exactly the training format from notebook 03: the encoder input is `[src_tag] subwords </s>` and the
decoder is forced to start with the target tag (`dav_Latn` for Kidaw'ida, `swh_Latn` for Kiswahili; read from `mt_config.json`).
Because the model was trained on short sentences (4.5 words on average), longer input is split at `.`, `!` and `?`,
translated as one batch and joined back, keeping line breaks. Sentences already translated are served from an in-memory LRU cache,
and a lock keeps CPU inference to one request at a time.

## Credits

Data: *Low-Resource Language Data: Parallel Corpora for Kiswahili and Kidaw'ida, Kalenjin, and Dholuo* — Mbogho, Kipkebut,
Wanzare, Awuor, Oloo & Lugano (2024), Lacuna Fund, CC BY 4.0. Base model: NLLB-200 (NLLB Team, 2022), CC BY-NC 4.0.
