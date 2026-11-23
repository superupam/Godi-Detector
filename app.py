# ==========================================
# GODI DETECTOR
# Streamlit Web Application
# "Don't just consume information. Verify it."
# ==========================================

import os
import joblib
import streamlit as st

# Import the Fact Checking Engine
from fact_checker import verify_claim_evidence
from utils import decide_verdict, explain_style, load_model_meta

MODEL_PATH = "models/godi_news_classifier.pkl"
MODEL_META_PATH = "models/model_meta.json"

# Set Page Configuration
st.set_page_config(
    page_title="Godi Detector - AI News Verification",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .verdict-box {
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border-left: 8px solid;
    }
    .verdict-real {
        background-color: #ECFDF5;
        border-color: #10B981;
        color: #065F46;
    }
    .verdict-fake {
        background-color: #FEF2F2;
        border-color: #EF4444;
        color: #991B1B;
    }
    .verdict-warn {
        background-color: #FFFBEB;
        border-color: #F59E0B;
        color: #92400E;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .badge-real { background-color: #D1FAE5; color: #065F46; }
    .badge-fake { background-color: #FEE2E2; color: #991B1B; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load and cache the trained ML pipeline."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)



# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/shield.png", width=64)
    st.title("Godi Detector")
    st.caption("AI Misinformation & Claim Verification System")
    st.divider()

    st.subheader("⚙️ Verification Settings")
    enable_fact_check = st.toggle("Enable Live Fact Checking", value=True, help="Queries Wikipedia & Google News for corroboration")
    min_confidence = st.slider("Strictness Threshold", min_value=50, max_value=90, value=70, step=5)

    st.divider()
    st.subheader("📊 Model Information")
    _meta = load_model_meta(MODEL_META_PATH)
    _size = _meta.get("dataset_size")
    _acc = _meta.get("accuracy")
    _size_str = f"{_size:,} sanitized & deduplicated articles" if isinstance(_size, int) else "unknown (run `train_model.py`)"
    _acc_str = f"{_acc:.2f}%" if isinstance(_acc, (int, float)) else "unknown"
    st.markdown(f"""
    * **Engine 1:** TF-IDF + Logistic Regression (Style & Tone)
    * **Engine 2:** Real-Time Knowledge Retrieval (Wikipedia & News RSS)
    * **Dataset:** {_size_str}
    * **Model Accuracy:** {_acc_str}
    """)
    st.caption("Developed by Rupam | Godi Detector Project")


# --- Main Application Header ---
st.markdown('<div class="main-header">🛡️ Godi Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">"Don\'t just consume information. Verify it." &mdash; Dual-Engine Misinformation Detection</div>', unsafe_allow_html=True)

model = load_model()
if model is None:
    st.error("Model file not found! Please run `python train_model.py` to train and save the model.")
    st.stop()

# --- Sample Buttons for Easy Demo ---
st.markdown("**Try sample headlines:**")
col_s1, col_s2, col_s3, col_s4 = st.columns(4)

default_text = ""
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

if col_s1.button("📌 Real Political News"):
    st.session_state.input_text = "The Senate on Wednesday passed a bipartisan $1.2 trillion infrastructure bill with overwhelming support from both parties."

if col_s2.button("🚨 Sensational Clickbait"):
    st.session_state.input_text = "BREAKING: Shocking video exposes secret plot by corrupt politicians to rig the election! WATCH HERE!"

if col_s3.button("⚽ Real Sports Event"):
    st.session_state.input_text = "Lionel Messi won the FIFA World Cup in December, 2022"

if col_s4.button("🪐 Science Discovery"):
    st.session_state.input_text = "NASA rover Perseverance landed on Mars in February 2021 to search for ancient microbial life."

# --- Input Area ---
user_text = st.text_area(
    "Enter a news headline, statement, or article excerpt to verify:",
    value=st.session_state.input_text,
    height=120,
    placeholder="Type or paste any news text here..."
)

verify_button = st.button("🔍 Verify Claim", type="primary", use_container_width=True)

# --- Analysis & Output ---
if verify_button and user_text.strip():
    with st.spinner("Analyzing text style and searching real-time knowledge bases..."):
        # Engine 1: ML Tone
        probabilities = model.predict_proba([user_text])[0]
        style_real_prob = probabilities[0] * 100
        style_fake_prob = probabilities[1] * 100
        real_signals, fake_signals = explain_style(model, user_text)

        # Engine 2: Fact Checking
        fact_result = None
        if enable_fact_check:
            fact_result = verify_claim_evidence(user_text)

        # Hybrid Decision Logic (shared with the API and CLI)
        decision = decide_verdict(
            style_real_prob,
            style_fake_prob,
            fact_result,
            min_confidence=min_confidence,
        )
        verdict_title = decision["verdict"]
        confidence = decision["confidence"]
        summary_msg = decision["explanation"]
        verdict_class = {
            "real": "verdict-real",
            "fake": "verdict-fake",
        }.get(decision["category"], "verdict-warn")

    # --- Render Verdict Banner ---
    st.markdown(f"""
    <div class="verdict-box {verdict_class}">
        <h2 style="margin: 0; font-size: 1.6rem;">{verdict_title} &bull; {confidence:.1f}% Confidence</h2>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.05rem;">{summary_msg}</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Two Columns for Dual Engines ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Engine 1: Linguistic Style Analysis")
        st.caption("Evaluates writing style, clickbait patterns, and vocabulary tone")
        
        st.write(f"**Formal Newsroom Tone:** {style_real_prob:.1f}%")
        st.progress(int(style_real_prob))

        st.write(f"**Sensational / Clickbait Tone:** {style_fake_prob:.1f}%")
        st.progress(int(style_fake_prob))

        st.markdown("**Linguistic Indicators — Real/Formal signals:**")
        if real_signals:
            for w in real_signals:
                st.markdown(f'<span class="badge badge-real">✓ {w}</span>', unsafe_allow_html=True)
        else:
            st.caption("None detected.")

        st.markdown("**Linguistic Indicators — Sensational/Fake signals:**")
        if fake_signals:
            for w in fake_signals:
                st.markdown(f'<span class="badge badge-fake">⚠ {w}</span>', unsafe_allow_html=True)
        else:
            st.caption("None detected.")

    with col2:
        st.subheader("🌐 Engine 2: Real-Time Fact Retrieval")
        st.caption("Live cross-referencing against Wikipedia and Google News")
        
        if fact_result:
            status = fact_result["status"]
            if status == "CORROBORATED":
                st.success(f"Status: **CORROBORATED** ({fact_result['match_ratio']*100:.0f}% claim match)")
            elif status == "PARTIALLY_CORROBORATED":
                st.warning(f"Status: **PARTIALLY CORROBORATED** ({fact_result['match_ratio']*100:.0f}% match)")
            else:
                st.error(f"Status: **NO EVIDENCE FOUND** ({fact_result['match_ratio']*100:.0f}% match)")

            st.markdown(f"**Top Source:** {fact_result['top_source']}")
            st.markdown(f"**Evidence Snippet:**\n> \"{fact_result['top_snippet'][:240]}...\"")
            
            if fact_result.get("top_url"):
                st.markdown(f"[🔗 View Original Source Article]({fact_result['top_url']})")
        else:
            st.info("Live Fact Checking is disabled in the sidebar.")

elif verify_button:
    st.warning("Please enter some text to verify.")
