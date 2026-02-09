# rag.py
from openai import OpenAI
from embedder import embed_query
from vector_store import load_vector_store
import numpy as np
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def retrieve_context(query, top_k=3):
    index, metadata = load_vector_store()
    if index is None:
        return []

    query_embedding = embed_query(query)
    distances, indices = index.search(
        np.array([query_embedding]), top_k
    )

    return [metadata[i] for i in indices[0]]

def generate_answer(query):
    contexts = retrieve_context(query)

    if not contexts:
        return "No policy documents found. Please upload documents first.", []

    context_text = "\n\n".join(contexts)

    prompt = f"""
You are an HR assistant.
Answer ONLY using the policy content below.
If the answer is not present, say "Information not found in policy".

Policy Content:
{context_text}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content, contexts
