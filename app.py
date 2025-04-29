import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
import io
import streamlit as st
import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained("./tokenizer")
    model = DistilBertForSequenceClassification.from_pretrained(
        "./",
        num_labels=2
    )
    model.eval()
    return tokenizer, model


def reset_text():
    """Callback to clear the text area."""
    st.session_state.news_box = ""

# ── Main ──────────────────────────────────────────────────────────────────────

# Page config
st.set_page_config(
    page_title="Fake-News Detector (DistilBERT)",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Load once
tokenizer, model = load_model()

# Keep our text in session state so Clear works
if "news_box" not in st.session_state:
    st.session_state.news_box = ""

# Header
st.markdown(
    "<h1 style='text-align:center; font-size:2.5rem;'>"
    "📰 Fake-News Detector (DistilBERT)</h1>",
    unsafe_allow_html=True,
)

# Banner image
st.image("banner.png", use_container_width=True)

st.markdown(
    "Paste a headline or short article below and click **Predict**. "
    "The model will say whether it’s *Fake* or *Real*, with confidence."
)

# Text area
news = st.text_area(
    "📝 News text",
    value=st.session_state.news_box,
    key="news_box",
    height=200,
    placeholder="e.g. 'WASHINGTON, April 12 (Reuters) – The U.S. Labor Department reported…'",
)

# Two‐column layout for buttons
col1, col2 = st.columns([1, 1])
download_csv_bytes = None

with col1:
    if st.button("Predict"):
        if not news.strip():
            st.warning("Please enter some text first.")
        else:
            # Tokenize & predict
            inputs = tokenizer(news, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]
            label_id = torch.argmax(probs).item()
            label = "Fake" if label_id == 0 else "Real"
            confidence = probs[label_id].item() * 100

            # Display
            icon = "❌" if label == "Fake" else "✅"
            color = "red" if label == "Fake" else "green"
            st.markdown(
                f"### {icon} Prediction: <span style='color:{color}'>{label}</span>",
                unsafe_allow_html=True,
            )
            st.write(f"**Confidence:** {confidence:.1f}%")

            # Build one‐row DataFrame
            df_one = pd.DataFrame({
                "text": [news.replace("\n", " ")],
                "prediction": [label],
                "confidence": [f"{confidence:.1f}%"],
            })
            csv_buffer = io.StringIO()
            df_one.to_csv(csv_buffer, index=False)
            download_csv_bytes = csv_buffer.getvalue().encode("utf-8")

with col2:
    st.button("Clear", on_click=reset_text)

st.markdown("---")

# If we just predicted, show download button for that row
if download_csv_bytes is not None:
    st.download_button(
        label="⬇️ Download This Prediction (CSV)",
        data=download_csv_bytes,
        file_name="my_prediction.csv",
        mime="text/csv",
    )

# Footer
st.markdown(
    "<small>⚡ Model: DistilBERT fine-tuned on ISOT fake/real news • "
    "Demo © 2025 Mokshitha</small>",
    unsafe_allow_html=True,
)
