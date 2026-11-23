import streamlit as st
import joblib
import re
from pathlib import Path

MODEL_PATH = Path("fake_news_pipeline.joblib")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def label_display(label):
    value = str(label).strip().lower()
    if value in {"0", "fake", "false"}:
        return "Fake"
    if value in {"1", "real", "true"}:
        return "Real"
    return f"Unknown label ({label})"

# These are transparent text-pattern indicators, not fact-checking rules.
SIGNAL_PATTERNS = {
    "Urgency or deadline language": r"\b(urgent|breaking|tomorrow|tonight|midnight|deadline|immediately|act now)\b",
    "Forward/share pressure": r"\b(forward|share this|send this|spread the word)\b",
    "Unspecified insider/source claim": r"\b(insiders?|sources?)\s+(say|claim|leak|revealed|reported)\b|\b(leaked memo|secret memo)\b",
    "Alarmist language": r"\b(shocking|unbelievable|alert|warning|exposed|they don't want you to know)\b",
    "Unusually absolute language": r"\b(all|every|never|guaranteed|automatically)\b",
}

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")
st.title("📰 Fake News Detection")
st.caption("Educational NLP classifier — not a factual verification tool.")
st.info("This app detects learned text patterns. It cannot verify claims against reliable sources or determine truth.")

if not MODEL_PATH.exists():
    st.warning("Model file not found. Train the model first using: python train_model.py")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
news = st.text_area(
    "Enter a news headline or article text",
    height=220,
    placeholder="Paste a headline or article here..."
)

if st.button("Classify text", type="primary", use_container_width=True):
    if not news.strip():
        st.warning("Please enter text before classifying.")
    else:
        cleaned = clean_text(news)
        prediction = model.predict([cleaned])[0]
        result = label_display(prediction)

        st.subheader("Model prediction")
        if result == "Fake":
            st.error(f"Prediction: {result}")
        elif result == "Real":
            st.success(f"Prediction: {result}")
        else:
            st.warning(result)

        st.caption(f"Raw model label: {prediction}")

        # LinearSVC decision_function is a signed margin, NOT a probability.
        classifier = model.named_steps.get("classifier") if hasattr(model, "named_steps") else None
        if classifier is not None and hasattr(classifier, "decision_function"):
            try:
                margin = float(model.decision_function([cleaned])[0])
                st.metric("Decision margin (not probability)", f"{margin:.3f}")
                absolute_margin = abs(margin)
                if absolute_margin < 0.25:
                    st.warning("The model's decision is close to its boundary. Treat this prediction as especially uncertain.")
                elif absolute_margin < 0.75:
                    st.warning("The model has a moderate decision margin. Verify the claim independently.")
                else:
                    st.caption("The model's decision margin is relatively large on this input, but it is not a confidence probability and does not prove correctness.")
            except Exception:
                pass

        # Transparent, separate heuristic signals; they do not change the ML prediction.
        matched_signals = [
            name for name, pattern in SIGNAL_PATTERNS.items()
            if re.search(pattern, news, flags=re.IGNORECASE)
        ]
        st.subheader("Text-pattern signals")
        st.caption("These rule-based signals are separate from the ML prediction and do not establish whether a claim is false.")
        if matched_signals:
            for signal in matched_signals:
                st.write(f"⚠️ {signal}")
            st.warning("This text contains patterns often associated with persuasive or alarmist messages. Check the claim using official sources.")
        else:
            st.success("No signals from this simple pattern list were detected. This is not evidence that the text is true.")

        st.divider()
        st.markdown("### Verification checklist")
        st.markdown("""
        - Find the claim on the relevant official website.
        - Check the publication date and original source.
        - Search for independent reporting from established outlets.
        - Avoid forwarding messages before verifying them.
        """)
        st.caption("Predictions and pattern signals can be incorrect. Verify claims independently.")
