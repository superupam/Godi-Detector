# LinkedIn publication material — Godi Detector

Publication is manual. Review, replace the placeholders, publish.

---

## Post copy

**🛡️ Godi Detector — detecting fake news with ML + live evidence**

Misinformation outruns manual fact-checking. I built Godi Detector to test a
simple idea: combine how a text is written with whether the public record
actually supports it.

**The problem**
Readers judge credibility from surface cues — authoritative tone, sensational
headline — and both are trivial to fake. Style alone is not truth; evidence
retrieval alone returns nothing for recent or niche claims.

**What I built**
A dual-engine verifier:
• Engine 1 — TF-IDF (1–2 grams, 50k features) + a supervised classifier that
  flags formal newsroom prose vs clickbait, and shows the exact words behind
  the decision.
• Engine 2 — live retrieval from Wikipedia and Google News, scoring whether one
  authoritative sentence actually contains the claim. Evidence overrides style:
  a claim that contradicts a known record is reported as debunked.

**The experiment**
I trained and compared three algorithms on one identical dataset, split and
feature space (39,095 cleaned articles, 80/20 stratified, seed 42):

| Model | Accuracy | Weighted F1 | Brier |
|---|---|---|---|
| Linear SVM | 98.68% | 98.68% | — |
| Logistic Regression | 97.84% | 97.84% | 0.0218 |
| Multinomial Naive Bayes | 94.82% | 94.82% | 0.0399 |

SVM wins on raw accuracy, but it only outputs an uncalibrated margin. Because
the app shows a confidence score, I deployed Logistic Regression — genuine
probabilities (Brier 0.0218), interpretable coefficients for token-level
explanations, 0.84 points behind on accuracy. Trade-off documented, not hidden.

**The data work that mattered most**
Removing label leakage. Real articles all started with a Reuters dateline and
wire tags; fake articles ended with CMS image credits. Without cleaning those,
the model learns the publisher, not the style. After leakage removal,
deduplication and filtering: 44,898 → 39,095 articles (54% real / 46% fake).

**Stack**
Python · scikit-learn · pandas/NumPy · Matplotlib · FastAPI · Streamlit ·
BeautifulSoup · pytest (99 tests)

**What it is not**
Not a fact-checker. Verdicts are decision support, and the UI says so.

🔗 GitHub: `<REPO_URL>`
🎥 60-second demo: `<VIDEO_URL>`

#MachineLearning #NLP #FakeNewsDetection #DataScience #Python #ScikitLearn
#FastAPI #AI #TextClassification #InformationIntegrity #Portfolio

---

## 60–90 second demo storyboard

| Time | On screen | Voice-over / caption |
|---|---|---|
| 0:00–0:08 | Title card: "Godi Detector — Don't just consume information. Verify it." | "Fake news spreads faster than we can fact-check it. Here's how I built a verifier that combines writing style with real evidence." |
| 0:08–0:20 | Terminal: `python train_model.py` output scrolling to the comparison table, then the results chart | "Three algorithms — Logistic Regression, Multinomial Naive Bayes and Linear SVM — trained on 39,095 cleaned articles, same split, same features. SVM hit 98.7%; I deployed Logistic Regression at 97.8% because it gives real probabilities." |
| 0:20–0:30 | Results folder + confusion matrices | "Every number comes from the saved run: metrics JSON, comparison table, confusion matrices, per-class reports." |
| 0:30–0:45 | Browser: paste the clickbait sample, click Verify | "Paste any headline. Engine 1 flags sensational phrasing and shows the exact tokens — here: 'shocking', 'secret', 'watch'." |
| 0:45–0:58 | Result banner: LIKELY FAKE / CLICKBAIT, 98.7% confidence | "Clickbait style with no corroborating evidence means likely fake — and it tells you why." |
| 0:58–1:12 | Paste a real claim, verify, show Engine 2 panel | "A real claim gets cross-checked against Wikipedia and Google News. Evidence can override style entirely: a false claim written formally is still debunked." |
| 1:12–1:25 | Switch to URL tab, scrape a live article | "Or drop an article link. The scraper extracts the body, with an SSRF guard so the server can't be pointed at internal addresses." |
| 1:25–1:30 | Repo + closing card | "Reproducible, tested, documented. GitHub link in the comments." |

**Recording tips:** 1080p, browser zoom 110–125%, cut loading spinners, show the
real result values (no mocked numbers), keep the caveat on screen — "decision
support, not proof".
