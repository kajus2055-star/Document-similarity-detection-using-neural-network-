import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Dataset
data = {
    "doc1": [
        "AI is transforming healthcare",
        "Machine learning is powerful",
        "I love football",
        "The sky is blue"
    ],
    "doc2": [
        "Artificial intelligence is changing medicine",
        "ML is useful",
        "Football is my favorite sport",
        "Sky looks blue"
    ],
    "label": [1, 1, 1, 1]
}

df = pd.DataFrame(data)

# Add negative samples
extra = {
    "doc1": ["I love pizza", "Dogs are animals"],
    "doc2": ["Sky is blue", "Cars are vehicles"],
    "label": [0, 0]
}

df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)

# Tokenization
texts = list(df['doc1']) + list(df['doc2'])
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

seq1 = tokenizer.texts_to_sequences(df['doc1'])
seq2 = tokenizer.texts_to_sequences(df['doc2'])

max_len = 10
X1 = pad_sequences(seq1, maxlen=max_len)
X2 = pad_sequences(seq2, maxlen=max_len)

y = np.array(df['label'])

# Model
vocab_size = len(tokenizer.word_index) + 1

input1 = Input(shape=(max_len,))
input2 = Input(shape=(max_len,))

embedding = Embedding(vocab_size, 50)
lstm = LSTM(64)

x1 = lstm(embedding(input1))
x2 = lstm(embedding(input2))

merged = Concatenate()([x1, x2])
dense = Dense(64, activation='relu')(merged)
output = Dense(1, activation='sigmoid')(dense)

model = Model([input1, input2], output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
model.fit([X1, X2], y, epochs=10, batch_size=2)

# Save
model.save("model.h5")

print("✅ Model trained & saved!")