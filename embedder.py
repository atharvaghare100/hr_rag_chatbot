# embedder.py
import numpy as np
import streamlit as st
from openai import OpenAI

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def embed_texts(texts):
    if not texts:
        return np.array([])

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    return np.array([e.embedding for e in response.data])

def embed_query(query):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    return np.array(response.data[0].embedding)
