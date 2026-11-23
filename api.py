# ==========================================
# GODI DETECTOR
# FastAPI Backend Server
# ==========================================

import os
import sys
import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; fall back to real env vars

# Import the Fact Checking Engine
from fact_checker import verify_claim_evidence
from scraper import is_valid_url, scrape_news_article
from utils import decide_verdict, explain_style, load_model_meta

MODEL_PATH = os.getenv("MODEL_PATH", "models/godi_news_classifier.pkl")
# Training stats live next to the model file, written by train_model.py
MODEL_META_PATH = os.path.join(os.path.dirname(MODEL_PATH), "model_meta.json")

# Upper bound on user-supplied input. The classifier vectorises the whole string,
# so an unbounded body is a cheap way to make the server allocate a lot of memory.
MAX_INPUT_CHARS = 20_000
MAX_URL_CHARS = 2_048

# ==========================================
# Load ML model at startup using FastAPI lifespan
# ==========================================

classifier_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier_model
    if os.path.exists(MODEL_PATH):
        print(f"Loading ML model from '{MODEL_PATH}'...")
        classifier_model = joblib.load(MODEL_PATH)
        print("Model loaded successfully!")
    else:
        print(f"Warning: Model file '{MODEL_PATH}' not found. Run 'python train_model.py' to generate it.")
    yield  # App runs here
    # Cleanup on shutdown (nothing needed for this model)


# Parse allowed origins from env (comma-separated, default to localhost only)
_origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost")
allowed_origins = [o.strip() for o in _origins_env.split(",") if o.strip()]

app = FastAPI(
    title="Godi Detector API",
    description="Dual-Engine Misinformation & News Verification System",
    version="1.0.0",
    lifespan=lifespan
)

# CORS — controlled via ALLOWED_ORIGINS in .env
# allow_credentials must stay False: the app uses no cookies or auth, and a
# credentialed response is invalid for a wildcard origin, so browsers reject it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class VerifyRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=3,
        max_length=MAX_INPUT_CHARS,
        description=f"Headline, news claim, or article URL to verify (max {MAX_INPUT_CHARS:,} characters)",
    )
    check_facts: bool = Field(default=True, description="Whether to query live Wikipedia/News archives")


class ScrapeRequest(BaseModel):
    url: str = Field(
        ...,
        max_length=MAX_URL_CHARS,
        description="Web URL of the news article to scrape and extract",
    )



@app.get("/api/status")
def get_status():
    """Health check and model metadata."""
    meta = load_model_meta(MODEL_META_PATH)
    accuracy = meta.get("accuracy")
    return {
        "status": "online",
        "model_loaded": classifier_model is not None,
        "engine_1": "TF-IDF + Logistic Regression",
        "engine_2": "Real-Time Wikipedia & Google News Retrieval",
        "dataset_articles": meta.get("dataset_size"),
        "model_accuracy": round(float(accuracy), 2) if accuracy is not None else None,
    }


@app.post("/api/scrape")
def scrape_article_endpoint(req: ScrapeRequest):
    """Scrape and extract news article title and text from a URL."""
    result = scrape_news_article(req.url)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/api/verify")
def verify_claim(req: VerifyRequest):
    """Verify a news headline, claim, or news article URL using both engines."""
    if not classifier_model:
        raise HTTPException(status_code=500, detail="ML Classifier model is not loaded.")
    
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    scraped_info = None
    # Auto-detect if input is a URL
    if is_valid_url(text):
        scrape_res = scrape_news_article(text)
        if not scrape_res["success"]:
            raise HTTPException(status_code=400, detail=scrape_res["error"])
        scraped_info = scrape_res
        # Formulate text from scraped title + content
        text = f"{scrape_res['title']}. {scrape_res['content']}".strip()

    # --- Engine 1: ML Style Analysis ---
    probs = classifier_model.predict_proba([text])[0]
    style_real_prob = round(float(probs[0]) * 100, 1)
    style_fake_prob = round(float(probs[1]) * 100, 1)
    real_signals, fake_signals = explain_style(classifier_model, text)

    # --- Engine 2: Real-Time Fact Retrieval ---
    fact_result = None
    if req.check_facts:
        fact_result = verify_claim_evidence(text)

    # --- Synthesize Hybrid Verdict (shared logic in utils.decide_verdict) ---
    decision = decide_verdict(style_real_prob, style_fake_prob, fact_result)

    return {
        "verdict": decision["verdict"],
        "category": decision["category"],
        "confidence": decision["confidence"],
        "explanation": decision["explanation"],
        "scraped_article": scraped_info,
        "engine_1_style": {
            "real_probability": style_real_prob,
            "fake_probability": style_fake_prob,
            "real_signals": real_signals,
            "fake_signals": fake_signals
        },
        "engine_2_facts": fact_result
    }


# Mount Static Files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_index():
    """Serve the single-page HTML frontend."""
    return FileResponse("static/index.html")
