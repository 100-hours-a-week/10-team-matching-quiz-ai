import numpy as np
from embedding import embed_texts
from vector_store import collection

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def rag_retriever(main_question: str, keyword: str, top_k: int = 6, sim_threshold: float = 0.5):
    query_embedding = embed_texts([main_question], keyword=keyword)[0]
    keyword_embedding = embed_texts([keyword])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
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

# 단독 실행 테스트용
if __name__ == "__main__":
    main_q = "FastAPI의 비동기 처리 방식은 무엇인가요?"
    keyword = "비동기"
    results = rag_retriever(main_q, keyword)

    print(f"\n 입력 질문: {main_q} / 키워드: {keyword}")
    for i, item in enumerate(results, 1):
        print(f"{i}. {item['question']} (유사도: {item['similarity']:.4f})")
