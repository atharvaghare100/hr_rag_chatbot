# vector_store.py
import faiss
import numpy as np
import os
import pickle

VECTOR_DB_PATH = "vector_db/faiss.index"
META_PATH = "vector_db/meta.pkl"

def save_vector_store(index, metadata):
    faiss.write_index(index, VECTOR_DB_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)

def load_vector_store():
    if not os.path.exists(VECTOR_DB_PATH):
        return None, None
    index = faiss.read_index(VECTOR_DB_PATH)
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def create_vector_store(embeddings, metadata):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index
