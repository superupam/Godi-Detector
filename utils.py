# ==========================================
# GODI DETECTOR
# Shared Utilities — imported by api.py, app.py, predict.py
# ==========================================

import json


def load_model_meta(path: str = "models/model_meta.json") -> dict:
    """
    Read the training metadata written by train_model.py.

    Returns an empty dict when the file is missing or unreadable, so callers
    can fall back to "unknown" instead of showing stale hardcoded numbers.

    Args:
        path: Location of model_meta.json

    Returns:
        dict with keys such as accuracy, dataset_size, train_size, test_size
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


# ==========================================
# Verdict synthesis — shared by api.py, app.py and predict.py
# ==========================================
# These thresholds used to be duplicated in all three front ends and had
# drifted apart (the corroboration floor was 88 in the API and 85 elsewhere).
# Engine 2 statuses come from fact_checker.verify_claim_evidence.

CONTRADICTED_FLOOR = 96.0
CORROBORATED_FLOOR = 88.0
PARTIAL_STYLE_THRESHOLD = 60.0
PARTIAL_REAL_CONFIDENCE = 72.0
PARTIAL_WARN_CONFIDENCE = 58.0
CLICKBAIT_THRESHOLD = 70.0
FORMAL_TONE_THRESHOLD = 75.0
FORMAL_TONE_CONFIDENCE = 65.0
AMBIGUOUS_CONFIDENCE = 50.0


def decide_verdict(style_real_prob, style_fake_prob, fact_result=None, min_confidence=None):
    """
    Combine Engine 1 (style) and Engine 2 (evidence) into a single verdict.

    Evidence overrides style: a claim that contradicts an authoritative record
    is reported as debunked even when it is written in formal newsroom prose.

    Args:
        style_real_prob: Engine 1 probability of "real", 0-100
        style_fake_prob: Engine 1 probability of "fake/clickbait", 0-100
        fact_result:    dict from fact_checker.verify_claim_evidence, or None
                        when live fact checking is disabled
        min_confidence: optional strictness override (0-100) used by the
                        Streamlit sidebar; None applies the defaults above

    Returns:
        dict with keys:
            verdict     - display string, e.g. "VERIFIED REAL NEWS"
            category    - "real" | "fake" | "warn"
            confidence  - 0-100, rounded to one decimal
            explanation - human-readable reason
            label       - short classification used by the CLI
    """
    # The sidebar slider is a per-user strictness setting; without it the
    # module defaults apply, so the API and CLI agree with each other.
    fake_threshold = CLICKBAIT_THRESHOLD if min_confidence is None else min_confidence
    real_threshold = FORMAL_TONE_THRESHOLD if min_confidence is None else min_confidence

    def result(verdict, category, confidence, explanation, label):
        return {
            "verdict": verdict,
            "category": category,
            "confidence": round(float(confidence), 1),
            "explanation": explanation,
            "label": label,
        }

    status = fact_result["status"] if fact_result else None
    source = fact_result["top_source"] if fact_result else None

    if status == "CONTRADICTED":
        return result(
            "DEBUNKED / FACTUALLY FALSE",
            "fake",
            max(fact_result["confidence"] * 100, CONTRADICTED_FLOOR),
            fact_result["top_snippet"],
            "Factually Debunked",
        )

    if status == "CORROBORATED":
        raw_confidence = fact_result["confidence"] * 100
        if min_confidence is not None and raw_confidence < min_confidence:
            return result(
                "PARTIALLY VERIFIED (BELOW STRICTNESS THRESHOLD)",
                "warn",
                raw_confidence,
                f"Evidence found on {source}, but confidence "
                f"({raw_confidence:.0f}%) is below your strictness setting of "
                f"{min_confidence:g}%.",
                "Below Strictness Threshold",
            )
        return result(
            "VERIFIED REAL NEWS",
            "real",
            max(raw_confidence, CORROBORATED_FLOOR),
            f"Corroborated by {source}. "
            "Authoritative public records confirm this claim.",
            "Verified Fact",
        )

    if status == "PARTIALLY_CORROBORATED":
        partial_explanation = (
            f"Related coverage found on {source}, "
            "but specific claim details are not fully confirmed."
        )
        if style_real_prob >= PARTIAL_STYLE_THRESHOLD:
            return result(
                "LIKELY REAL (PARTIALLY CORROBORATED)",
                "real",
                PARTIAL_REAL_CONFIDENCE,
                partial_explanation,
                "Partial Evidence",
            )
        return result(
            "UNVERIFIED / REQUIRES CAUTION",
            "warn",
            PARTIAL_WARN_CONFIDENCE,
            partial_explanation,
            "Partial Evidence",
        )

    # No corroborating evidence: fact checking was off, or Engine 2 returned
    # NO_EVIDENCE for this claim.
    if style_fake_prob >= fake_threshold:
        return result(
            "LIKELY FAKE NEWS / CLICKBAIT",
            "fake",
            style_fake_prob,
            "Sensationalist/clickbait style detected and no corroborating evidence "
            "found in trusted sources.",
            "Uncorroborated & Sensational",
        )
    if style_real_prob >= real_threshold:
        return result(
            "UNVERIFIED CLAIM (FORMAL JOURNALISTIC TONE)",
            "warn",
            FORMAL_TONE_CONFIDENCE,
            "Written in formal newsroom style, but no official confirmation was "
            "found in news archives.",
            "Unverified Claim",
        )
    return result(
        "AMBIGUOUS / UNVERIFIED",
        "warn",
        AMBIGUOUS_CONFIDENCE,
        "No authoritative evidence found and linguistic style is neutral.",
        "Insufficient Data",
    )


def explain_style(model, text: str, top_k: int = 5):
    """
    Extract the strongest positive/negative contributing tokens from the
    TF-IDF + Logistic Regression pipeline.

    Args:
        model:  Trained sklearn Pipeline with named steps 'tfidf' and 'classifier'
        text:   Raw input string to explain
        top_k:  Number of top signals to return for each direction

    Returns:
        real_signals (list[str]): Words pushing toward REAL / formal news
        fake_signals (list[str]): Words pushing toward FAKE / clickbait
    """
    tfidf = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    vec = tfidf.transform([text])
    feature_names = tfidf.get_feature_names_out()

    _, cols = vec.nonzero()
    contributions = []
    for col in cols:
        weight = float(vec[0, col]) * float(classifier.coef_[0, col])
        contributions.append((feature_names[col], weight))

    contributions.sort(key=lambda x: x[1])

    real_signals = [word for word, score in contributions if score < 0][:top_k]
    fake_signals = [word for word, score in reversed(contributions) if score > 0][:top_k]

    return real_signals, fake_signals
