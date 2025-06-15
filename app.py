import os
import streamlit as st
import requests
import traceback
import torch

# dynamically ensure correct transformers version
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    st.warning("Installing compatible transformers…")
    os.system("pip install transformers==4.37.2 --quiet")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

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

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            prediction = torch.argmax(output.logits, dim=1).item()

        if prediction == 1:
            st.success("😏 This tweet is **Sarcastic**!")
        else:
            st.info("🙂 This tweet is **Not Sarcastic**.")

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained("kausar57056/urdu-sarcasm-model")
    tokenizer = AutoTokenizer.from_pretrained("kausar57056/urdu-sarcasm-model")
    model.to(device)
    model.eval()
    return model, tokenizer, device

if __name__ == "__main__":
    catch_all_errors()
