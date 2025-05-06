import chromadb
import hashlib

DB_PATH = "./chroma_db"
COLLECTION_NAME = "questions"

chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

def generate_id(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()

# 벡터 저장 함수 (중복 방지 + 빠른 저장)
def save_to_vectorstore(questions, embeddings, batch_size=5000):
    ids = [generate_id(q) for q in questions]

    for i in range(0, len(questions), batch_size):
        collection.upsert(
            documents=questions[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )
