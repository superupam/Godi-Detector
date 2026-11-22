# ==========================================
# GODI DETECTOR
# Shared Utilities — imported by api.py, app.py, predict.py
# ==========================================


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
