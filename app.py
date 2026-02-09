# app.py
import streamlit as st
import os
import numpy as np
from ingest import load_pdf_text, chunk_text
from embedder import embed_texts
from vector_store import create_vector_store, save_vector_store
from rag import generate_answer

st.set_page_config(page_title="HR Policy RAG Chatbot")

st.title("📄 HR Policy Assistant (RAG)")

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("vector_db", exist_ok=True)

st.sidebar.header("Upload HR Policy PDFs")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True,
    key="hr_policy_uploader"
)


if st.sidebar.button("Process Documents", key="process_docs_btn"):

    all_chunks = []
    metadata = []

    for file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())

        text = load_pdf_text(file_path)
        chunks = chunk_text(text)

        all_chunks.extend(chunks)
        metadata.extend(chunks)

    embeddings = embed_texts(all_chunks)
    index = create_vector_store(np.array(embeddings), metadata)
    save_vector_store(index, metadata)

    st.sidebar.success("Documents processed successfully!")

st.divider()

query = st.text_input("Ask a question about HR policies:")

if st.button("Get Answer"):
    answer, sources = generate_answer(query)

    st.subheader("Answer")
    st.write(answer)

    with st.expander("Source Policy Chunks"):
        for src in sources:
            st.write(src)
