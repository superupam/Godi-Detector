# 🛡️ Godi Detector

> *"Don't just consume information. Verify it."*

A dual-engine AI system for detecting fake news and verifying claims using:
- **Engine 1** — TF-IDF + Logistic Regression trained on 39,095 articles (97.8% accuracy)
- **Engine 2** — Real-time cross-referencing against Wikipedia and Google News RSS

Built with FastAPI (REST backend), Streamlit (notebook UI), and vanilla JS (web frontend).

---

## 📁 Project Structure

```
Godi Detector/
├── api.py              # FastAPI backend server
├── app.py              # Streamlit interactive UI
├── predict.py          # CLI tool for terminal-based verification
├── fact_checker.py     # Engine 2: Real-time Wikipedia & News retrieval
├── scraper.py          # News article URL scraper
├── utils.py            # Shared utilities (explain_style)
├── train_model.py      # ML model training pipeline
├── prepare_data.py     # Data cleaning & preparation pipeline
├── explore_data.py     # EDA helper script
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── data/
│   ├── raw/            # Original Fake.csv and True.csv datasets
│   └── processed/      # Cleaned, deduplicated news_dataset.csv
├── models/
│   ├── godi_news_classifier.pkl       # Latest trained model
│   ├── godi_news_classifier_<ts>.pkl  # Versioned snapshots
│   └── model_meta.json                # Training metadata
├── static/             # FastAPI frontend (HTML + CSS + JS)
│   ├── index.html
│   ├── style.css
│   └── app.js
└── notebooks/          # Jupyter exploration notebooks
```

---

## ⚙️ Setup

### 1. Create & activate virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
```bash
# Copy the example config and edit as needed
copy .env.example .env
```

---

## 🚀 Running the Project

### Option A — FastAPI Web App (Recommended)
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```
Then open http://localhost:8000 in your browser.

### Option B — Streamlit UI
```bash
streamlit run app.py
```

### Option C — Command-Line Verifier
```bash
# Interactive mode
python predict.py

# Single claim (pass as argument)
python predict.py "NASA rover Perseverance landed on Mars in February 2021"
```

---

## 🧠 Training the Model

> Only needed if you want to retrain on updated data.

```bash
# Step 1: Clean and prepare the raw dataset
python prepare_data.py

# Step 2: Train and save the model
python train_model.py
```

After training, two files are saved:
- `models/godi_news_classifier.pkl` — always the latest
- `models/godi_news_classifier_<YYYYMMDD_HHMMSS>.pkl` — versioned snapshot
- `models/model_meta.json` — accuracy, timestamp, dataset size

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serves the web frontend |
| GET | `/api/status` | Health check + model metadata |
| POST | `/api/verify` | Verify a headline/claim/URL |
| POST | `/api/scrape` | Extract article text from a URL |

### Example: Verify a claim
```bash
curl -X POST http://localhost:8000/api/verify \
  -H "Content-Type: application/json" \
  -d '{"text": "NASA rover Perseverance landed on Mars in February 2021", "check_facts": true}'
```

---

## 🔧 Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins (use `*` for dev only) |
| `MODEL_PATH` | `models/godi_news_classifier.pkl` | Path to the trained model file |
| `LOG_LEVEL` | `info` | Uvicorn log level |

---

## 📊 Dataset

The model is trained on the [LIAR-PLUS / FakeNewsNet](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) dataset:
- `data/raw/True.csv` — Real news articles (Reuters wire)
- `data/raw/Fake.csv` — Fake / clickbait articles

Preprocessing steps (see `prepare_data.py`):
1. Remove Reuters dateline leakage from real articles
2. Strip photo credits and social links from fake articles
3. Combine title + body text
4. Deduplicate and shuffle

---

## 📄 License

Built by **Rupam** · Godi Detector Project
