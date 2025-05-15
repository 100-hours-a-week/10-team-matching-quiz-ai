from typing import List, Dict
import torch
import logging
import numpy as np
from vector_db.utils import embed_texts
from vector_db.chroma_client import get_all_documents_with_vectors
from sentence_transformers.util import cos_sim


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))

def rag_retriever(
    main_question: str,
    keyword: str = None,
    top_k: int = 4,
    sim_threshold: float = 0.7,
    question_weight: float = 0.12,
    keyword_weight: float = 0.88
) -> List[Dict[str, float]]:
    try:
        if not keyword:
            question_weight, keyword_weight = 1.0, 0.0

        q_vec = embed_texts([main_question])[0]
        k_vec = embed_texts([keyword])[0] if keyword else None

        all_docs = get_all_documents_with_vectors()
        results = []

        for _, doc_text, doc_vec in all_docs:
            q_sim = cosine_similarity(np.array(q_vec), np.array(doc_vec))
            k_sim = cosine_similarity(np.array(k_vec), np.array(doc_vec)) if k_vec else 0.0
            final_sim = question_weight * q_sim + keyword_weight * k_sim
            if final_sim >= sim_threshold:
                results.append({
                    "question": doc_text,
                    "similarity": round(final_sim, 4)
                })

        sorted_results = sorted(results, key=lambda x: x["similarity"], reverse=True)
        return sorted_results[:top_k]

    except Exception as e:
        logging.warning(f"RAG 검색 실패: {e}")
        return []
