# embedder.py
import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

def embed_texts(texts):
    if not texts:
        return np.array([])

    return model.encode(
        texts,
        convert_to_numpy=True,   # 🔑 FORCE NumPy
        show_progress_bar=False,
        normalize_embeddings=True
    )

def embed_query(query):
    return model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]
