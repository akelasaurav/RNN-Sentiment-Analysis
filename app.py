import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------
# Load resources safely
# ------------------------------
@st.cache_resource
def load_resources():
    try:
        # Try loading modern format first
        try:
            model = load_model("model.keras", compile=False)
        except:
            model = load_model("model.keras", compile=False)

        # Load tokenizer
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)

        # Load max length
        with open("max_len.pkl", "rb") as f:
            max_len = pickle.load(f)

        # Create reverse word index (FAST lookup)
        index_to_word = {index: word for word, index in tokenizer.word_index.items()}

        return model, tokenizer, max_len, index_to_word

    except Exception as e:
        st.error(f"❌ Error loading resources: {e}")
        return None, None, None, None


model, tokenizer, max_len, index_to_word = load_resources()

# ------------------------------
# Prediction function
# ------------------------------
def predict_next_word(text):
    try:
        sequence = tokenizer.texts_to_sequences([text])[0]

        if len(sequence) == 0:
            return "No valid input"

        sequence = pad_sequences([sequence], maxlen=max_len - 1, padding='pre')

        preds = model.predict(sequence, verbose=0)
        predicted_index = np.argmax(preds)

        return index_to_word.get(predicted_index, "Not found")

    except Exception as e:
        return f"Error: {e}"

# ------------------------------
# UI
# ------------------------------
st.set_page_config(page_title="Next Word Prediction", layout="centered")

st.title("🧠 Next Word Prediction (LSTM)")
st.write("Type a sentence and predict the **next word** using Deep Learning.")

user_input = st.text_input("✍️ Enter text:")

if st.button("Predict Next Word"):
    if model is None:
        st.error("❌ Model not loaded properly")
    elif user_input.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        result = predict_next_word(user_input)
        st.success(f"👉 Predicted Next Word: {result}")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("🚀 LSTM-based Next Word Prediction App")