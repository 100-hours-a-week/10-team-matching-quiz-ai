# vector_db/query_pipeline.py

from embedding import embed_texts
from vector_store import collection
from init_data import init_vector_store_from_csv

def query_similar_questions(query: str, top_k: int = 4):
    """
    입력 질문에 대해 유사 질문을 검색
    """
    query_embedding = embed_texts([query])
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    questions = results["documents"][0]
    distances = results["distances"][0]
    return list(zip(questions, distances))

# 실행 예시
if __name__ == "__main__":
    # 질문 DB 초기화
    init_vector_store_from_csv("./dataset/questions.csv")

    query = "FastAPI는 어떤 특징이 있나요?"
    results = query_similar_questions(query)

    print(f"\n🔍 입력 질문: {query}\n")
    for i, (q, dist) in enumerate(results, 1):
        print(f"{i}. {q} (유사도 거리: {dist:.4f})")
