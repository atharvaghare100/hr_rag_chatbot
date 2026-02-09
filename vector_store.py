# vector_store.py
import numpy as np
import pickle
import os
from sklearn.neighbors import NearestNeighbors

DB_PATH = "vector_db/store.pkl"

def save_vector_store(embeddings, metadata):
    os.makedirs("vector_db", exist_ok=True)
    with open(DB_PATH, "wb") as f:
        pickle.dump((embeddings, metadata), f)

def load_vector_store():
    if not os.path.exists(DB_PATH):
        return None, None
    with open(DB_PATH, "rb") as f:
        return pickle.load(f)

def create_vector_store(embeddings, metadata):
    return embeddings, metadata

def search_vector_store(query_embedding, top_k=3):
    embeddings, metadata = load_vector_store()
    if embeddings is None:
        return []

    nn = NearestNeighbors(n_neighbors=top_k, metric="cosine")
    nn.fit(embeddings)

    distances, indices = nn.kneighbors([query_embedding])
    return [metadata[i] for i in indices[0]]
