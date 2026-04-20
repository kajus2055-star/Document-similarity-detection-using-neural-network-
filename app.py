import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = tf.keras.models.load_model("model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 10

def predict_similarity(text1, text2):
    seq1 = tokenizer.texts_to_sequences([text1])
    seq2 = tokenizer.texts_to_sequences([text2])

    p1 = pad_sequences(seq1, maxlen=max_len)
    p2 = pad_sequences(seq2, maxlen=max_len)

    score = model.predict([p1, p2])[0][0]
    return score

# UI
st.title("📄 Text Similarity Detector (LSTM)")

text1 = st.text_area("Enter Document 1")
text2 = st.text_area("Enter Document 2")

if st.button("Check Similarity"):
    if text1 and text2:
        score = predict_similarity(text1, text2)

        st.write(f"### Similarity Score: {score:.2f}")

        if score > 0.5:
            st.success("✅ Documents are SIMILAR")
        else:
            st.error("❌ Documents are NOT SIMILAR")
    else:
        st.warning("⚠️ Please enter both texts")