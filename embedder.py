# embedder.py
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts: list[str]):
    return model.encode(texts, show_progress_bar=False)

def embed_query(query: str):
    return model.encode([query])[0]
