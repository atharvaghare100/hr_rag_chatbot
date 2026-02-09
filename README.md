# HR Policy Assistant using Retrieval Augmented Generation (RAG)

This project implements a Retrieval Augmented Generation (RAG) based chatbot that allows employees to ask questions about company HR policies and receive accurate, document-grounded answers.

## Features
- Upload HR policy PDFs (leave policy, WFH rules, guidelines)
- Semantic search using vector embeddings
- Retrieval Augmented Generation to reduce hallucinations
- Source transparency for each answer
- Streamlit-based web interface

## Tech Stack
- Python
- Streamlit
- FAISS
- Sentence-Transformers
- OpenAI API

## How it Works
1. HR policy documents are uploaded and processed into text chunks.
2. Chunks are converted into embeddings and stored in a vector database.
3. User queries are embedded and matched with relevant policy sections.
4. Retrieved content is injected into an LLM prompt to generate grounded answers.

## Deployment
The application is deployed using Streamlit Cloud and GitHub integration.

## Note
API keys are managed securely using environment variables / Streamlit secrets.
