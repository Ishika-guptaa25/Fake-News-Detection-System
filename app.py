# app.py
import streamlit as st
import pickle
import gzip
import os
import numpy as np

MODEL_PATH = "model.pkl.gz"
VECT_PATH = "vectorizer.pkl.gz"

st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detector")
st.markdown("""
Paste the news text or headline below and click **Predict**.
The model was trained using a PassiveAggressiveClassifier with TF-IDF features.
""")

# load model & vectorizer
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECT_PATH):
        st.error(f"Model or vectorizer not found. Looking for: {MODEL_PATH} and {VECT_PATH}")
        st.info(f"Current files in directory: {os.listdir('.')}")
        return None, None

    with gzip.open(MODEL_PATH, "rb") as f:  # ✅ Use variable
        model = pickle.load(f)
    with gzip.open(VECT_PATH, "rb") as f:  # ✅ Use variable
        vectorizer = pickle.load(f)

    return model, vectorizer

model, vectorizer = load_model()

text_input = st.text_area("Enter news text:", height=200, placeholder="Paste news article or headline here...")

col1, col2 = st.columns(2)
with col1:
    if st.button("Predict") and model is not None and text_input.strip() != "":
        X = vectorizer.transform([text_input])
        pred = model.predict(X)[0]
        pred_proba = None
        try:
            score = model.decision_function(X)[0]
            prob = 1 / (1 + np.exp(-abs(score)))
            pred_proba = prob
        except Exception:
            pred_proba = None

        label = "Fake" if pred==1 else "Real / True"
        st.markdown(f"### Prediction: **{label}**")
        if pred_proba is not None:
            st.write(f"Confidence (approx.): {pred_proba:.3f}")
        else:
            st.write("Confidence: N/A")

with col2:
    if st.button("Clear"):
        st.rerun()

st.markdown("---")
st.markdown("**How to use:** paste the text and click Predict. For best results, paste the full article or headline + short description.")