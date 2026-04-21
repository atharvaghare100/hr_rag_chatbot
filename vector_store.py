import streamlit as st
import numpy as np
import faiss
from pinecone import Pinecone, ServerlessSpec

# ---- PINECONE SETUP ----
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)


def save_vector_store(embeddings, chunks):
    # Save to Pinecone (database)
    vectors = []
    for i, (emb, text) in enumerate(zip(embeddings, chunks)):
        vectors.append({
            "id": str(i),
            "values": emb.tolist(),
            "metadata": {"text": text}
        })
    index.upsert(vectors)


def search_vector_store(query_embedding, top_k=3):
    # Step 1: Fetch all vectors from Pinecone
    stats = index.describe_index_stats()
    total = stats["total_vector_count"]

    if total == 0:
        return []

    all_ids = [str(i) for i in range(total)]
    fetch_result = index.fetch(ids=all_ids)

    # Step 2: Build FAISS index from fetched vectors
    vectors = []
    texts = []
    for id_, vec in fetch_result.vectors.items():
        vectors.append(vec.values)
        texts.append(vec.metadata["text"])
        
    vectors_np = np.array(vectors).astype(np.float32)
    dim = vectors_np.shape[1]

    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(vectors_np)

    # Step 3: Search using FAISS
    query = query_embedding.astype(np.float32).reshape(1, -1)
    _, indices = faiss_index.search(query, top_k)

    return [texts[i] for i in indices[0]]