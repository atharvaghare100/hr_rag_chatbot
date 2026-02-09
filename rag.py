# rag.py
from openai import OpenAI
import streamlit as st
from embedder import embed_query
from vector_store import search_vector_store

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def generate_answer(query):
    contexts = search_vector_store(embed_query(query))

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
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content, contexts
