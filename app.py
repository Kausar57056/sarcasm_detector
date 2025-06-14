import os

# Try to import torch, install if missing
try:
    import torch
except ImportError:
    os.system("pip install torch==2.5.1")
    import torch

import streamlit as st
import requests
import traceback
from transformers import AutoTokenizer
from sentimixturenet import SentimixtureNet

# Hugging Face Model URL (update with your username)
HF_MODEL_URL = "https://huggingface.co/kausar57056/urdu-sarcasm-model/resolve/main/sentimixture_model.pt"

# Catch-all error handler for better debugging
def catch_all_errors():
    try:
        run_app()
    except Exception as e:
        st.error("❌ An unexpected error occurred:")
        st.code(str(e))
        st.text("Traceback:")
        st.text(traceback.format_exc())

# Main app logic in a separate function
def run_app():
    st.title("🤖 Urdu Sarcasm Detection")
    st.markdown("Enter an Urdu tweet and I will tell you if it's sarcastic or not.")

    # Load model + tokenizer
    st.write("🚀 Loading model...")
    model, tokenizer, device = load_model()

    # User input
    tweet = st.text_area("✍️ Enter Urdu Tweet:", height=100)

    if st.button("🔍 Predict"):
        if not tweet.strip():
            st.warning("Please enter a tweet to continue.")
            return

        encoding = tokenizer(tweet, return_tensors="pt", truncation=True, padding=True)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            prediction = torch.argmax(output, dim=1).item()

        if prediction == 1:
            st.success("😏 This tweet is **Sarcastic**!")
        else:
            st.info("🙂 This tweet is **Not Sarcastic**.")

# Model loader
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "sentimixture_model.pt"

    try:
        st.write("📥 Checking if model is already downloaded...")
        if not os.path.exists(model_path):
            st.write("⬇️ Downloading model from Hugging Face...")
            response = requests.get(HF_MODEL_URL)
            if response.status_code != 200:
                raise RuntimeError(f"Download failed: HTTP {response.status_code}")
            with open(model_path, "wb") as f:
                f.write(response.content)
            st.write("✅ Download complete.")

        st.write("🔧 Loading model structure...")
        model = SentimixtureNet()
        st.write("📦 Loading model weights...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        st.write("🧠 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        st.write("✅ Model and tokenizer loaded.")

        return model.to(device), tokenizer, device

    except Exception as e:
        st.error("❌ Error during model loading:")
        st.code(str(e))
        st.text("📄 Traceback:")
        st.text(traceback.format_exc())
        raise e
