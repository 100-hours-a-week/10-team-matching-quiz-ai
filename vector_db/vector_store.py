from chromadb import EphemeralClient

# 메모리 기반 클라이언트 (세션 종료 시 사라짐)
chroma_client = EphemeralClient()

# 질문 컬렉션 생성 or 로드
collection = chroma_client.get_or_create_collection(name="questions")

def save_to_vectorstore(questions, embeddings):
    """
    질문과 임베딩된 벡터를 벡터 DB에 저장한다.
    """
    ids = [f"id_{i}" for i in range(len(questions))]
    collection.add(
        documents=questions,
        embeddings=embeddings,
        ids=ids
    )
