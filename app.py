import os
import torch
import requests
import streamlit as st
import traceback
from transformers import AutoTokenizer
from sentimixturenet import SentimixtureNet

# Hugging Face model URL
HF_MODEL_URL = "https://huggingface.co/kausar57056/urdu-sarcasm-model/resolve/main/fixed_sentimixture_model.pt"
MODEL_FILENAME = "fixed_sentimixture_model.pt"

# Catch and display any errors in the app
def catch_all_errors():
    try:
        run_app()
    except Exception as e:
        st.error("❌ An unexpected error occurred:")
        st.code(str(e))
        st.text("📄 Traceback:")
        st.text(traceback.format_exc())

# Main app logic
def run_app():
    st.set_page_config(page_title="Urdu Sarcasm Detection", page_icon="😏")
    st.title("🤖 Urdu Sarcasm Detection")
    st.markdown("Enter an Urdu tweet and I will tell you if it's **sarcastic** or not.")

    st.write("🚀 Loading model and tokenizer...")
    model, tokenizer, device = load_model()

    tweet = st.text_area("✍️ Enter Urdu Tweet:", height=100)

    if st.button("🔍 Predict"):
        if not tweet.strip():
            st.warning("Please enter a tweet to continue.")
            return

        encoding = tokenizer(tweet, return_tensors="pt", truncation=True, padding=True)
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            prediction = torch.argmax(output, dim=1).item()

        labels = {0: "Not Sarcastic", 1: "Sarcastic"}
        emojis = {0: "🙂", 1: "😏"}
        colors = {0: st.info, 1: st.success}
        colors[prediction](f"{emojis[prediction]} This tweet is **{labels[prediction]}**.")

# Cached model loading function
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Download model if not present
    if not os.path.exists(MODEL_FILENAME):
        st.info("⬇️ Downloading model from Hugging Face...")
        response = requests.get(HF_MODEL_URL)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to download model. HTTP {response.status_code}")
        with open(MODEL_FILENAME, "wb") as f:
            f.write(response.content)
        st.success("✅ Model downloaded.")

    # Load the model
    try:
        st.write("📦 Initializing model...")
        model = SentimixtureNet()
        state_dict = torch.load(MODEL_FILENAME, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        st.success("✅ Model initialized.")
    except Exception as e:
        st.error("❌ Failed to initialize the model.")
        st.code(str(e))
        st.text("📄 Traceback:")
        st.text(traceback.format_exc())
        st.stop()

    # Load the tokenizer
    try:
        st.write("📦 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        st.success("✅ Tokenizer loaded.")
    except Exception as e:
        st.error("❌ Failed to load tokenizer.")
        st.code(str(e))
        st.text("📄 Traceback:")
        st.text(traceback.format_exc())
        st.stop()

    return model, tokenizer, device

# Run the app
if __name__ == "__main__":
    catch_all_errors()
