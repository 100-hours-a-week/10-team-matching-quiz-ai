import numpy as np
from vector_db.utils import embed_texts
from vector_db.chroma_client import collection

if __name__ != "__main__":
    import os
    from vector_db.init_data import init_vector_store_from_csv

    current_dir = os.path.dirname(__file__)
    csv_path = os.path.join(current_dir, "question_data.csv")
    init_vector_store_from_csv(csv_path)

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def rag_retriever(main_question: str, keyword: str, top_k: int = 6, sim_threshold: float = 0.6):
    keyword_embedding = embed_texts([keyword])[0]
    results = collection.query(
        query_embeddings=[keyword_embedding],
        n_results=top_k * 2,
        include=["documents", "embeddings"]
    )

    all_questions = results["documents"][0]
    all_embeddings = results["embeddings"][0]

    filtered = []
    for q, q_emb in zip(all_questions, all_embeddings):
        sim = cosine_similarity(keyword_embedding, q_emb)
        if sim >= sim_threshold:
            filtered.append({"question": q, "similarity": round(sim, 4)})

    filtered.sort(key=lambda x: -x["similarity"])
    return filtered[:top_k]