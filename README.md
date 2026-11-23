# 🛡️ Godi Detector

> *"Don't just consume information. Verify it."*

A dual-engine fake-news and claim verifier. It combines a supervised style
classifier with live evidence retrieval, so a verdict reflects both *how* a text
is written and *whether* public records and news archives support it.

| Engine | What it does | Backed by |
|---|---|---|
| **Engine 1 — Style** | Predicts `real` vs `fake/clickbait` from vocabulary and phrasing, and reports the tokens that drove the decision | TF-IDF (1–2 grams, 50k features) + Logistic Regression, trained on 39,095 articles — 97.84% test accuracy |
| **Engine 2 — Evidence** | Retrieves live Wikipedia summaries and Google News RSS items, then checks whether one authoritative sentence actually contains the claim | Wikipedia REST/Search API, Google News RSS |

Accuracy and corpus size are read from `models/model_meta.json` at request time,
so they always match the last training run — nothing is hardcoded.

Three front ends share one pipeline: a **FastAPI web app** (`static/`), a
**Streamlit app** (`app.py`), and a **terminal verifier** (`predict.py`).

---

## 🧭 How a verification runs

```
input (headline / excerpt / article URL)
  │
  ├─ URL?  ──► scraper.py: fetch, extract title + first 12 paragraphs
  │
  ├─► Engine 1  classifier.predict_proba → real % / fake % + contributing tokens
  │
  ├─► Engine 2  fact_checker.verify_claim_evidence
  │      1. direct contradiction check on authoritative event records
  │      2. Wikipedia search (up to 4 pages) + Google News RSS (up to 5 items)
  │      3. sentence-level co-occurrence scoring of the claim against evidence
  │
  └─► synthesis → verdict, category (real / fake / warn), confidence, explanation
```

Evidence overrides style: a claim that contradicts a known record is reported as
debunked even if it is written in formal newsroom prose.

### Verdicts

| Verdict | Category | Trigger |
|---|---|---|
| `DEBUNKED / FACTUALLY FALSE` | fake | Engine 2 found a direct contradiction (≥ 96% confidence) |
| `VERIFIED REAL NEWS` | real | Engine 2 corroborated the claim (≥ 88%) |
| `LIKELY REAL (PARTIALLY CORROBORATED)` | real | Partial evidence and formal tone (real ≥ 60%) |
| `UNVERIFIED / REQUIRES CAUTION` | warn | Partial evidence but informal tone |
| `PARTIALLY VERIFIED (BELOW STRICTNESS THRESHOLD)` | warn | Streamlit only: corroborated evidence under the sidebar strictness slider |
| `LIKELY FAKE NEWS / CLICKBAIT` | fake | No evidence and fake-style probability ≥ 70% |
| `UNVERIFIED CLAIM (FORMAL JOURNALISTIC TONE)` | warn | No evidence, real ≥ 75% |
| `AMBIGUOUS / UNVERIFIED` | warn | No evidence, neutral style (50%) |

Engine 2 statuses are `CORROBORATED` (match ≥ 0.55), `PARTIALLY_CORROBORATED`
(≥ 0.35), `NO_EVIDENCE`, and Engine 1's probabilities are always returned
separately so the two signals can be read independently.

---

## 📁 Project structure

```
Godi Detector/
├── api.py               # FastAPI backend: /api/status, /api/verify, /api/scrape, serves static/
├── app.py               # Streamlit UI (port 8501)
├── predict.py           # Terminal verifier (interactive + one-shot argument)
├── fact_checker.py      # Engine 2: Wikipedia + Google News retrieval and scoring
├── scraper.py           # News URL scraper + SSRF guard (_safe_get / validate_public_url)
├── utils.py             # decide_verdict(), explain_style(), load_model_meta() — shared by all front ends
├── train_model.py       # Training pipeline → .pkl snapshots + model_meta.json
├── prepare_data.py      # Cleaning, leakage removal, deduplication
├── explore_data.py      # Quick pandas look at the raw CSVs
├── requirements.txt
├── .env.example         # Copy to .env
├── static/              # Web frontend served by FastAPI
│   ├── index.html
│   ├── style.css
│   └── app.js
├── models/
│   ├── godi_news_classifier.pkl          # Latest model (gitignored)
│   ├── godi_news_classifier_<ts>.pkl     # Versioned snapshots (gitignored)
│   └── model_meta.json                   # accuracy, sizes, trained_at (tracked)
└── data/                # Not in git — see "Dataset" below
    ├── raw/             # True.csv, Fake.csv
    └── processed/       # news_dataset.csv (~189 MB)
```

Everything in the repo is path-relative and must be run from the project root —
`models/…`, `data/…` and `static/…` are resolved against the current directory.

---

## ⚙️ Setup

### Requirements
- **Python 3.12 or 3.13.** Python 3.14 fails on machines with Windows Smart App
  Control enabled: it blocks `pydantic_core`'s compiled extension
  (`_pydantic_core.cp314-win_amd64.pyd`). Use 3.13 unless Smart App Control is off.
- ~900 MB of disk for the environment, plus ~300 MB for the dataset if you retrain.

### 1. Create and activate a virtual environment

The folder is `.venv`, not `venv`:

```bash
# Windows
py -3.13 -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3.13 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the environment

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Defaults work for local development. `.env` is gitignored — never commit it.

### 4. Get a model

A trained model is not committed (`.pkl` files are gitignored). Either use the
one already in `models/`, or retrain — see [Training](#-training).

---

## 🚀 Running

### Option A — FastAPI web app (recommended)

```bash
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

Open <http://localhost:8000>.

Use `127.0.0.1` for local-only access. `--host 0.0.0.0` exposes the API to your
whole network — see [Security](#-security-notes) first.

> **First start after a fresh install can take up to a minute.** The first
> `joblib.load` makes Windows scan the newly installed scikit-learn/numpy
> binaries (slower with Smart App Control on). Afterwards the model unpickles in
> ~1.5 s and the server is ready in 5–10 s. Wait for `Model loaded successfully!`
> and `Application startup complete.` before sending requests.

The UI has two tabs — **Direct Text / Headline** and **News Article Link (URL)** —
a fact-check toggle, sample chips, and per-engine result panels (style
probabilities, contributing tokens, evidence source, match ratio, snippet, link).

### Option B — Streamlit UI

```bash
streamlit run app.py
```

Serves on <http://localhost:8501>. Same engines, sidebar shows live model metadata.

### Option C — Terminal verifier

```bash
# Interactive loop (type 'exit' or Ctrl+C to quit)
python predict.py

# One-shot
python predict.py "NASA rover Perseverance landed on Mars in February 2021"
```

The CLI classifies the text you give it; it does not fetch URLs. Use the API or
the web UI for links.

### Utilities

```bash
python scraper.py https://www.bbc.com/news    # test article extraction
python fact_checker.py "claim to check"       # test Engine 2 alone
python explore_data.py                        # peek at the raw CSVs
```

---

## 🌐 API

Base URL `http://localhost:8000`. No authentication — intended for local use.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Serves `static/index.html` |
| `GET` | `/static/*` | Frontend assets |
| `GET` | `/api/status` | Health check + model metadata |
| `POST` | `/api/verify` | Verify text or a URL with both engines |
| `POST` | `/api/scrape` | Extract an article's title and body from a URL |
| `GET` | `/docs` | Generated OpenAPI docs (FastAPI default) |

### `GET /api/status`

```json
{
  "status": "online",
  "model_loaded": true,
  "engine_1": "TF-IDF + Logistic Regression",
  "engine_2": "Real-Time Wikipedia & Google News Retrieval",
  "dataset_articles": 39095,
  "model_accuracy": 97.84
}
```

`dataset_articles` and `model_accuracy` are `null` when `model_meta.json` is
missing or unreadable.

### `POST /api/verify`

```json
{
  "text": "NASA rover Perseverance landed on Mars in February 2021",
  "check_facts": true
}
```

- `text` — 3 to 20,000 characters; a headline, excerpt, claim, or `http(s)` URL.
  A URL is scraped first, then classified from title + body. Longer input is
  rejected with `422` rather than vectorised.
- `check_facts` — default `true`. Set `false` for a fast Engine 1-only pass.

```bash
curl -X POST http://localhost:8000/api/verify \
  -H "Content-Type: application/json" \
  -d '{"text": "NASA rover Perseverance landed on Mars in February 2021", "check_facts": true}'
```

Response:

```json
{
  "verdict": "VERIFIED REAL NEWS",
  "category": "real",
  "confidence": 88.0,
  "explanation": "Corroborated by Wikipedia. Authoritative public records confirm this claim.",
  "scraped_article": null,
  "engine_1_style": {
    "real_probability": 91.2,
    "fake_probability": 8.8,
    "real_signals": ["said", "percent"],
    "fake_signals": ["shocking", "watch"]
  },
  "engine_2_facts": {
    "status": "CORROBORATED",
    "confidence": 0.92,
    "top_source": "Wikipedia",
    "top_snippet": "...",
    "top_url": "https://en.wikipedia.org/wiki/...",
    "matched_keywords": ["nasa", "rover"],
    "missing_keywords": [],
    "match_ratio": 0.67,
    "total_evidence_count": 9
  }
}
```

`scraped_article` is populated (same shape as `/api/scrape`, minus `error`) when
the input was a URL. `engine_2_facts` is `null` when `check_facts` is `false`.

### `POST /api/scrape`

```json
{ "url": "https://www.bbc.com/news/articles/..." }
```

Returns `url`, `domain`, `title`, `content` (first 12 paragraphs), `preview`
(50 words), and `word_count`. It strips `script`/`style`/`nav`/`header`/`footer`
boilerplate and drops paragraphs under 40 characters or containing cookie,
subscribe or newsletter text.

The body is streamed and truncated at **5 MB**, so a huge or malicious page
cannot exhaust server memory; the text an article needs sits near the top of the
document. Non-200 responses are never read.

### Errors

| Status | When |
|---|---|
| `400` | Invalid or unsafe URL, non-200 from the target site, no extractable text, empty text |
| `422` | Request body failed validation (`text` under 3 or over 20,000 characters, a URL over 2,048 characters) |
| `500` | `/api/verify` called while no model is loaded — run `train_model.py` |

---

## 🧠 Training

Only needed to retrain on updated data. Both steps read and write fixed paths
from the project root.

```bash
python prepare_data.py    # data/raw/*.csv  → data/processed/news_dataset.csv
python train_model.py     # → models/*.pkl + models/model_meta.json
```

`prepare_data.py` removes label leakage before training: Reuters datelines and
wire tags from real articles, and image/photo credits from fake articles. It
then combines title + body, drops duplicates and articles under 20 characters,
and shuffles with `random_state=42`.

`train_model.py` splits 80/20 stratified, fits TF-IDF (`stop_words="english"`,
`max_features=50000`, `ngram_range=(1,2)`, `min_df=2`) into Logistic Regression
(`max_iter=1000`), prints accuracy, a classification report and a confusion
matrix, then writes:

- `models/godi_news_classifier.pkl` — always the latest
- `models/godi_news_classifier_<YYYYMMDD_HHMMSS>.pkl` — immutable snapshot
- `models/model_meta.json` — `trained_at`, `accuracy`, `dataset_size`, `train_size`, `test_size`

The API and Streamlit read `model_meta.json` per request, so numbers update
without a restart. The `.pkl` itself is loaded once at startup — restart
`uvicorn` after retraining.

---

## 🔧 Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins. Use `*` for local dev only |
| `MODEL_PATH` | `models/godi_news_classifier.pkl` | Model file; `model_meta.json` is read from the same directory |
| `LOG_LEVEL` | `info` | Uvicorn log level: `debug`, `info`, `warning`, `error` |

---

## 📊 Dataset

The [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
(ISOT), split into:

- `data/raw/True.csv` — real articles (Reuters wire)
- `data/raw/Fake.csv` — fake and clickbait articles

Both are gitignored, as is the generated `data/processed/news_dataset.csv`
(~189 MB). Download the two CSVs into `data/raw/` and run `prepare_data.py`.

---

## 🔒 Security notes

- **SSRF guard on `/api/scrape` and URL input to `/api/verify`.** Because the
  server fetches a caller-chosen URL, `scraper.py` validates every hop:
  `http`/`https` only, hostname blocklist (`localhost`, `*.local`, `*.internal`,
  cloud metadata names), and the resolved IP must be globally routable.
  Embedded IPv4 (NAT64 `64:ff9b::/96`, 6to4 `2002::/16`) and integer/hex forms
  like `2130706433` are unwrapped and checked. Redirects are followed manually
  (max 5) so a public URL cannot bounce to a private one. Blocked requests
  return `400` with a plain-language reason.
- **No authentication.** The API is meant for `127.0.0.1`. Binding to `0.0.0.0`
  publishes an unauthenticated scraping and verification endpoint to your LAN.
- **CORS.** `ALLOWED_ORIGINS=*` is for development. Set the exact origin before
  exposing the service.
- **Secrets.** `.env` is gitignored. It currently holds no credentials; if you
  add an API key later, keep it there and out of version control.

### Known limitation

The guard validates the IP at request time but does not pin the resolved address
for the actual connection, so a hostile DNS server could in principle rebind
between check and connect (TOCTOU). This matters only for untrusted multi-user
deployments; for local use it is acceptable.

---

## 🧯 Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `DLL load failed` on `pydantic_core` or `numpy` | Windows Smart App Control blocked the extension. Use Python 3.13 (see Setup). |
| `Warning: Model file '…' not found` | `models/godi_news_classifier.pkl` is missing. Run `train_model.py`, or set `MODEL_PATH`. |
| `/api/verify` returns `500` | Same as above — the classifier did not load; check the startup log. |
| First request takes a minute | One-time Windows antivirus scan of freshly installed binaries. Subsequent starts are seconds. |
| `error while attempting to bind on address` | Port 8000 is already in use — stop the other server or pass `--port 8001`. |
| `/api/scrape` says the site blocked the request | Non-200 response (paywall, bot check, redirect loop). The scraper sends browser headers but no JS execution. |
| `/api/scrape` returns `400` "local, private, or internal address" | The URL failed the SSRF guard. Intentional. |
| Verdict says `NO_EVIDENCE` for a true claim | Wikipedia/Google News returned nothing usable, or the claim's wording does not co-occur in one authoritative sentence. Rephrase with the key entities. |
| Streamlit shows `unknown` accuracy | `models/model_meta.json` is missing or invalid; retrain or restore it. |

---

## ⚠️ Limitations

- **Engine 1 classifies style, not truth.** It was trained on 2016–2017 Reuters
  wire copy versus clickbait sites, so it flags tone and register. Formal prose
  can still be false, and terse true statements can look suspicious.
- **Engine 2 covers what Wikipedia and Google News happen to index.** Recent,
  local or niche claims often return no usable evidence, which maps to
  `AMBIGUOUS / UNVERIFIED` rather than a confident answer.
- Verdicts are decision support, not fact-checking. Confirm anything consequential
  against a primary source.

---

## 📄 License

Built by **Rupam** · Godi Detector Project
