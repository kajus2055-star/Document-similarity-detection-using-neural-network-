# 📄 Text Similarity Detection using LSTM

---

## 🔹 Project Overview

This project implements a **Text Document Similarity Detection system** using a **Siamese Neural Network with LSTM**.
It takes two input texts and predicts how similar they are by generating a **similarity score between 0 and 1**.

The system consists of:

* A **backend API** for prediction
* A **frontend UI** for user interaction
* A trained **deep learning model**

---

## 🔹 Key Features

* 🔍 Detect semantic similarity between two texts
* 🧠 Uses LSTM-based Siamese Neural Network
* ⚡ Fast prediction using API (FastAPI backend)
* 🎨 Interactive frontend UI for user input
* 📊 Outputs similarity score with interpretation
* 💻 Easy to run locally

---

## 🔹 System Architecture

The system follows a **client-server architecture**:

1. User enters two text inputs in the frontend
2. Frontend sends request to backend API
3. Backend processes text using:

   * Tokenization
   * Padding
   * LSTM Model
4. Model computes similarity score
5. Backend returns result to frontend
6. Frontend displays similarity result

--

## 🔹 Technologies Used

* Python
* TensorFlow / Keras
* FastAPI (Backend)
* Streamlit (Frontend)
* NumPy, Pandas

---

## 🔹 Local Installation and Setup

Follow these steps to run the project on your system:

---

### 🔸 1. Clone the Repository

```bash
git clone https://github.com/your-username/text-similarity-project.git
cd text-similarity-project
```

---

### 🔸 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

* Windows:

```bash
venv\Scripts\activate
```

* Mac/Linux:

```bash
source venv/bin/activate
```

---

### 🔸 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 🔸 4. Run the FastAPI Backend Server

```bash
uvicorn main:app --reload
```

Backend will run at:

```
http://127.0.0.1:8000
```

---

### 🔸 5. Launch the Frontend UI

```bash
streamlit run app.py
```

Frontend will open in your browser automatically.

---

## 🔹 API Endpoint (Backend)

* **POST /predict**
* Input:

```json
{
  "text1": "Your first document",
  "text2": "Your second document"
}
```

* Output:

```json
{
  "similarity_score": 0.82
}
```

---

## 🔹 Applications

* Plagiarism Detection
* Chatbots
* Search Engines
* Recommendation Systems

---

## 🔹 Conclusion

This project demonstrates how **LSTM-based neural networks** can effectively capture semantic meaning and measure similarity between text documents.
It provides a scalable solution using **modern backend (FastAPI)** and **interactive frontend UI**.

---
