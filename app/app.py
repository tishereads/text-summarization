"""Streamlit interface: summarize any text with a fine-tuned or pretrained model."""
import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

st.set_page_config(page_title="Abstractive Summarizer", layout="wide")
st.title("📝 Abstractive Text Summarizer")
st.caption("PEGASUS / T5 / BART — fine-tuned on CNN/DailyMail and XSum")

@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if torch.cuda.is_available():
        model = model.to("cuda")
    return tokenizer, model

DEFAULT_MODELS = {
    "PEGASUS (CNN/DailyMail)": "google/pegasus-cnn_dailymail",
    "PEGASUS (XSum)": "google/pegasus-xsum",
    "BART (CNN/DailyMail)": "facebook/bart-large-cnn",
    "T5 (small)": "t5-small",
}

with st.sidebar:
    st.header("Settings")
    choice = st.selectbox("Model", list(DEFAULT_MODELS.keys()))
    custom = st.text_input("Or paste a local/HF model path (overrides above)", "")
    model_name = custom.strip() if custom.strip() else DEFAULT_MODELS[choice]

    max_len = st.slider("Max summary length", 30, 300, 128)
    num_beams = st.slider("Beams", 1, 8, 4)

    st.divider()
    st.caption("Tip: point this at your fine-tuned model, e.g. ./output/google-pegasus-large/final")

tokenizer, model = load_model(model_name)

text = st.text_area("Paste the article to summarize", height=250)

if st.button("Summarize", type="primary") and text.strip():
    with st.spinner("Generating summary..."):
        inputs = tokenizer(
            [text], max_length=512, truncation=True, padding=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_length=max_len, num_beams=num_beams,
                length_penalty=0.8, no_repeat_ngram_size=3,
            )
        summary = tokenizer.decode(out[0], skip_special_tokens=True)
    st.subheader("Summary")
    st.success(summary)
    st.subheader("Word counts")
    st.write(f"Input: {len(text.split())} words → Summary: {len(summary.split())} words")
