# app.py

import streamlit as st
import torch
import os
import requests
from transformers import AutoTokenizer
from sentimixturenet import SentimixtureNet

# ✅ Hugging Face URL to your model file
HF_MODEL_URL = "https://huggingface.co/kausar57056/urdu-sarcasm-model/resolve/main/sentimixture_model.pt"

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Download model if not present
    model_path = "sentimixture_model.pt"
    if not os.path.exists(model_path):
        with open(model_path, "wb") as f:
            f.write(requests.get(HF_MODEL_URL).content)

    model = SentimixtureNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    return model.to(device), tokenizer, device

model, tokenizer, device = load_model()

# Streamlit UI
st.title("🤖 Urdu Sarcasm Detection")
st.markdown("Enter an Urdu tweet below and click **Predict** to check for sarcasm.")

text_input = st.text_area("✍️ Enter Urdu Tweet Here:", height=120)

if st.button("🔍 Predict"):
    if not text_input.strip():
        st.warning("Please enter a tweet.")
    else:
        encoding = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = torch.argmax(output, dim=1).item()
