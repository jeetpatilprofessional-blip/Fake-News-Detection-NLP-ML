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

st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detection")
st.caption("Educational NLP classifier — not a factual verification tool.")

if not MODEL_PATH.exists():
    st.warning("Model file not found. Run: python train_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)
news = st.text_area("Enter a news headline or text", height=160, placeholder="Paste news text here...")
if st.button("Classify", type="primary"):
    if not news.strip():
        st.info("Please enter text first.")
    else:
        prediction = model.predict([clean_text(news)])[0]
        if prediction == 0:
            st.error("Model prediction: Fake")
        elif prediction == 1:
            st.success("Model prediction: Real")
        else:
            st.write(f"Model prediction: {prediction}")
        st.caption("Predictions may be incorrect. Verify claims independently.")
