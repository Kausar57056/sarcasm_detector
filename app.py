import os
import streamlit as st
import requests
import traceback
from sentimixturenet import SentimixtureNet

HF_MODEL_URL = "https://huggingface.co/kausar57056/urdu-sarcasm-model/resolve/main/sentimixture_model.pt"

def catch_all_errors():
    try:
        run_app()
    except Exception as e:
        st.error("❌ An unexpected error occurred:")
        st.code(str(e))
        st.text("Traceback:")
        st.text(traceback.format_exc())

def run_app():
    st.title("🤖 Urdu Sarcasm Detection")
    st.markdown("Enter an Urdu tweet and I will tell you if it's sarcastic or not.")

    st.write("🚀 Loading model...")
    model, tokenizer, device = load_model()

    tweet = st.text_area("✍️ Enter Urdu Tweet:", height=100)

    if st.button("🔍 Predict"):
        if not tweet.strip():
            st.warning("Please enter a tweet to continue.")
            return

        encoding = tokenizer(tweet, return_tensors="pt", truncation=True, padding=True)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        import torch
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            prediction = torch.argmax(output, dim=1).item()

        if prediction == 1:
            st.success("😏 This tweet is **Sarcastic**!")
        else:
            st.info("🙂 This tweet is **Not Sarcastic**.")

@st.cache_resource
def load_model():
    try:
        import torch
    except ImportError:
        os.system("pip install torch==2.5.1")
        import torch

    try:
        from transformers import AutoTokenizer
    except ImportError:
        os.system("pip install transformers==4.37.2")
        from transformers import AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "sentimixture_model.pt"

    if not os.path.exists(model_path):
        st.info("⬇️ Downloading model from Hugging Face...")
        response = requests.get(HF_MODEL_URL)
        with open(model_path, "wb") as f:
            f.write(response.content)

    model = SentimixtureNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    return model.to(device), tokenizer, device

# Run
if __name__ == "__main__":
    catch_all_errors()
