import torch
from vector_db.utils import embed_texts
from vector_db.chroma_client import collection
from typing import List, Dict
import logging

def cosine_similarity_gpu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a / torch.norm(a, dim=-1, keepdim=True)
    b = b / torch.norm(b, dim=-1, keepdim=True)
    return torch.matmul(a, b.T)  

def rag_retriever(
    main_question: str,
    keyword: str,
    top_k: int = 6,
    sim_threshold: float = 0.6
) -> List[Dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        keyword_emb = embed_texts([keyword])[0] 
        keyword_tensor = torch.tensor(keyword_emb, dtype=torch.float32, device=device).unsqueeze(0)

        results = collection.query(
            query_embeddings=[keyword_emb],  
            n_results=top_k * 5, 
            include=["documents", "embeddings"]
        )

        docs = results.get("documents", [[]])[0]
        embs = results.get("embeddings", [[]])[0]

        if len(docs) == 0 or len(embs) == 0:
            return []

        embedding_tensor = torch.tensor(embs, dtype=torch.float32, device=device)

        similarities = cosine_similarity_gpu(keyword_tensor, embedding_tensor).squeeze(0)

        top_sim_values, top_indices = similarities.topk(k=top_k * 2)

        filtered = [
            {
                "question": docs[i],
                "similarity": round(sim, 4)
            }
            for i, sim in zip(top_indices.tolist(), top_sim_values.tolist())
            if sim >= sim_threshold
        ]

        return filtered[:top_k]

    except Exception as e:
        logging.warning(f"RAG 검색 실패: {e}")
        return []
