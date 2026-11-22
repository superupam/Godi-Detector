# ==========================================
# GODI DETECTOR
# Dual-Engine News Verification Tool
# Engine 1: Linguistic Style & Clickbait Classifier (ML)
# Engine 2: Real-Time Fact Retrieval & Evidence Engine (Live Knowledge)
# ==========================================

import os
import sys
import joblib

# Import the Fact Checking Engine
from fact_checker import verify_claim_evidence
from utils import explain_style

MODEL_PATH = "models/godi_news_classifier.pkl"


def load_classifier():
    """Load the trained machine learning pipeline."""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found!")
        print("Please run 'python train_model.py' first to train and save the model.")
        sys.exit(1)
    
    print("Loading Godi Detector ML Classifier...")
    model = joblib.load(MODEL_PATH)
    print("Classifier loaded successfully!\n")
    return model



def predict_news(model, text, check_facts=True):
    """
    Analyze news text using both the ML Style Classifier and Fact Retrieval Engine.
    """
    text_clean = text.strip()
    if not text_clean:
        print("Empty text provided. Please enter some text to analyze.\n")
        return

    # --- Engine 1: Linguistic Style Classification ---
    probabilities = model.predict_proba([text_clean])[0]
    style_real_prob = probabilities[0] * 100
    style_fake_prob = probabilities[1] * 100
    real_signals, fake_signals = explain_style(model, text_clean)

    # --- Engine 2: Real-Time Fact Retrieval ---
    fact_result = None
    if check_facts:
        print("Searching real-time knowledge base and news archives for evidence...")
        fact_result = verify_claim_evidence(text_clean)

    # --- Synthesize Hybrid Verdict ---
    if fact_result and fact_result["status"] == "CONTRADICTED":
        verdict = "DEBUNKED / FACTUALLY FALSE"
        final_confidence = max(round(fact_result["confidence"] * 100, 1), 96.0)
        verdict_type = "Factually Debunked"
        explanation = fact_result["top_snippet"]
    elif fact_result and fact_result["status"] == "CORROBORATED":
        # Fact check overrides style bias
        verdict = "VERIFIED REAL NEWS"
        final_confidence = max(fact_result["confidence"] * 100, 85.0)
        verdict_type = "Verified Fact"
        explanation = (
            f"Corroborated by {fact_result['top_source']}. "
            f"Independent authoritative records confirm this claim."
        )
    elif fact_result and fact_result["status"] == "PARTIALLY_CORROBORATED":
        if style_real_prob > 60:
            verdict = "LIKELY REAL (PARTIALLY CORROBORATED)"
            final_confidence = 70.0
        else:
            verdict = "UNVERIFIED / REQUIRES CAUTION"
            final_confidence = 55.0
        verdict_type = "Partial Evidence"
        explanation = (
            f"Related topic found on {fact_result['top_source']}, "
            f"but specific claim details are not fully confirmed."
        )
    else:
        # No evidence found in trusted sources
        if style_fake_prob >= 70:
            verdict = "LIKELY FAKE NEWS / CLICKBAIT"
            final_confidence = style_fake_prob
            verdict_type = "Uncorroborated & Sensational"
            explanation = "Sensationalist/clickbait style detected and no corroborating evidence found in trusted sources."
        elif style_real_prob >= 75:
            verdict = "UNVERIFIED CLAIM (FORMAL JOURNALISTIC TONE)"
            final_confidence = 65.0
            verdict_type = "Unverified Claim"
            explanation = "Written in formal newsroom style, but no official confirmation was found in news archives."
        else:
            verdict = "AMBIGUOUS / UNVERIFIED"
            final_confidence = 50.0
            verdict_type = "Insufficient Data"
            explanation = "No authoritative evidence found and linguistic style is neutral."

    # --- Display Beautiful Results ---
    print("=" * 65)
    print(f"VERDICT: [{verdict}] ({final_confidence:.1f}% Confidence)")
    print("=" * 65)
    print(f"Verdict Classification: {verdict_type}")
    print(f"Summary: {explanation}\n")
    
    print("-" * 65)
    print("ENGINE 1: LINGUISTIC STYLE ANALYSIS (ML Model)")
    print(f"  Tone Probabilities: Real={style_real_prob:.1f}% | Fake/Clickbait={style_fake_prob:.1f}%")
    if style_real_prob > style_fake_prob:
        print(f"  Key formal indicators: {', '.join(real_signals) if real_signals else 'None'}")
    else:
        print(f"  Key clickbait indicators: {', '.join(fake_signals) if fake_signals else 'None'}")

    if fact_result:
        print("-" * 65)
        print("ENGINE 2: REAL-TIME FACT RETRIEVAL (Live Sources)")
        print(f"  Verification Status : [{fact_result['status']}]")
        print(f"  Top Matched Source  : {fact_result['top_source']}")
        print(f"  Evidence Match Rate : {fact_result['match_ratio'] * 100:.0f}% of key claim words verified")
        print(f"  Evidence Snippet    : \"{fact_result['top_snippet'][:200]}...\"")
        if fact_result['top_url']:
            print(f"  Source URL          : {fact_result['top_url']}")
    print("=" * 65 + "\n")


def run_interactive_mode(model):
    """Continuous terminal loop for analyzing multiple articles."""
    print("=" * 65)
    print("      GODI DETECTOR - DUAL-ENGINE NEWS VERIFICATION")
    print("      'Don't just consume information. Verify it.'")
    print("=" * 65)
    print("Instructions:")
    print("  * Paste or type any news headline or article excerpt.")
    print("  * Type 'exit' or 'quit' to stop.")
    print("=" * 65 + "\n")
    
    while True:
        try:
            user_input = input("Enter news text to verify > ").strip()
            if user_input.lower() in ("exit", "quit", "q"):
                print("\nExiting Godi Detector. Goodbye!")
                break
            if not user_input:
                continue
            predict_news(model, user_input)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting Godi Detector. Goodbye!")
            break


if __name__ == "__main__":
    classifier = load_classifier()
    
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
        predict_news(classifier, input_text)
    else:
        run_interactive_mode(classifier)
