import numpy as np
import time
from embedder import embed_query


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def evaluate_rag(query, answer, contexts):
    start_time = time.time()

    query_emb = embed_query(query)
    answer_emb = embed_query(answer)

    # 🔹 Context Recall
    context_scores = []
    for ctx in contexts:
        ctx_emb = embed_query(ctx)
        context_scores.append(cosine_similarity(query_emb, ctx_emb))

    context_recall = np.mean(context_scores) if context_scores else 0

    # 🔹 Answer Relevance
    answer_relevance = cosine_similarity(query_emb, answer_emb)

    # 🔹 Faithfulness
    faithfulness_scores = []
    for ctx in contexts:
        ctx_emb = embed_query(ctx)
        faithfulness_scores.append(cosine_similarity(answer_emb, ctx_emb))

    faithfulness = np.mean(faithfulness_scores) if faithfulness_scores else 0

    latency = time.time() - start_time

    return {
        "context_recall": round(float(context_recall), 3),
        "answer_relevance": round(float(answer_relevance), 3),
        "faithfulness": round(float(faithfulness), 3),
        "latency_sec": round(latency, 2)
    }