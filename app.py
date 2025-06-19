import os
import torch
import streamlit as st
import requests
import traceback
from transformers import AutoTokenizer
from sentimixturenet import SentimixtureNet

MODEL_URL = "https://huggingface.co/kausar57056/urdu-sarcasm-model/resolve/main/best_sarcasm_epoch5.pt"
MODEL_FILE = "best_sarcasm_epoch5.pt"

@st.cache_resource
def load_model_and_tokenizer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(MODEL_FILE):
        with st.spinner("Downloading model..."):
            response = requests.get(MODEL_URL)
            if response.status_code != 200:
                raise RuntimeError(f"Failed to download model. HTTP {response.status_code}")
            with open(MODEL_FILE, "wb") as f:
                f.write(response.content)

    model = SentimixtureNet()
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    return model, tokenizer, device

def main():
    st.title("🤖 Urdu Sarcasm Detection")
    st.markdown("Enter an Urdu tweet to detect if it's sarcastic or not.")

    model, tokenizer, device = load_model_and_tokenizer()
    tweet = st.text_area("📝 Enter Urdu Tweet", height=100)

    if st.button("🔍 Detect"):
        if tweet.strip() == "":
            st.warning("Please enter a tweet.")
            return

        encoding = tokenizer(tweet, return_tensors="pt", truncation=True, padding=True, max_length=128)
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            sarcasm_score = probs[0][1].item()

        if sarcasm_score >= 0.4:
            st.success(f"😏 Sarcastic (Confidence: {sarcasm_score:.2f})")
        else:
            st.info(f"🙂 Not Sarcastic (Confidence: {1 - sarcasm_score:.2f})")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("❌ Something went wrong!")
        st.code(str(e))
        st.text(traceback.format_exc())
