import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import numpy as np

# Initialize Pinecone
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])

INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]

# Create index if not exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # embedding size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)


def save_vector_store(embeddings, metadata):
    vectors = []

    for i, (emb, text) in enumerate(zip(embeddings, metadata)):
        vectors.append({
            "id": str(i),
            "values": emb.tolist(),
            "metadata": {"text": text}
        })

    index.upsert(vectors)


def search_vector_store(query_embedding, top_k=3):
    results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )

    return [match["metadata"]["text"] for match in results["matches"]]