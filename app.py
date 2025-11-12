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
The model was trained using a **Logistic Regression** with TF-IDF features on 39,000+ news articles.
""")


# Add cache_data instead of cache_resource to force refresh
@st.cache_data
def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECT_PATH):
        st.error(f"Model or vectorizer not found. Looking for: {MODEL_PATH} and {VECT_PATH}")
        st.info(f"Current files in directory: {os.listdir('.')}")
        return None, None

    try:
        with gzip.open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with gzip.open(VECT_PATH, "rb") as f:
            vectorizer = pickle.load(f)

        # Debug info
        st.sidebar.success("✅ Model loaded successfully!")
        st.sidebar.info(f"Model type: {type(model).__name__}")

        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


model, vectorizer = load_model()

text_input = st.text_area(
    "Enter news text:",
    height=200,
    placeholder="Paste the FULL article here (at least 3-4 sentences for best results)..."
)

col1, col2 = st.columns(2)
with col1:
    if st.button("Predict") and model is not None and text_input.strip() != "":
        X = vectorizer.transform([text_input])
        pred = model.predict(X)[0]

        # For LogisticRegression we can get probabilities
        try:
            pred_proba = model.predict_proba(X)[0]
            confidence = pred_proba[pred]
        except:
            # Fallback for other models
            try:
                score = model.decision_function(X)[0]
                confidence = 1 / (1 + np.exp(-abs(score)))
            except:
                confidence = None

        # pred = 0 means TRUE, pred = 1 means FAKE
        label = "Fake" if pred == 1 else "Real / True"

        if pred == 1:
            st.markdown(f"### Prediction: **:red[{label}]**")
        else:
            st.markdown(f"### Prediction: **:green[{label}]**")

        if confidence is not None:
            st.write(f"Confidence: {confidence:.3f}")

with col2:
    if st.button("Clear"):
        st.rerun()

st.markdown("---")
st.markdown(
    "**How to use:** paste the full article text and click Predict. The model works best with complete articles (multiple sentences).")