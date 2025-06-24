import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
import hashlib
from app.vector_db.config import DB_PATH, FOLLOW_UP_COLLECTION_NAME, QUIZ_COLLECTION_NAME

chroma_client = chromadb.PersistentClient(path=DB_PATH)
follow_up_collection = chroma_client.get_or_create_collection(FOLLOW_UP_COLLECTION_NAME)
quiz_collection = chroma_client.get_or_create_collection(QUIZ_COLLECTION_NAME)

def generate_id(text: str) -> str:
    """텍스트를 MD5 해시로 변환하여 고유 ID 생성"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def save_to_question_vectorstore(questions, embeddings, batch_size=1024):
    """꼬리질문 생성용 벡터스토어에 저장"""
    ids = [generate_id(q) for q in questions]

    for i in range(0, len(questions), batch_size):
        follow_up_collection.upsert(
            documents=questions[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )

def get_all_question_documents_with_vectors():
    """꼬리질문 생성용 벡터스토어에서 모든 문서 조회"""
    results = follow_up_collection.get(include=["embeddings", "documents"])
    return list(zip(results["ids"], results["documents"], results["embeddings"]))

def save_to_quiz_vectorstore(documents, embeddings, metadatas=None, batch_size=1024):
    """퀴즈 생성용 벡터스토어에 저장"""
    ids = [generate_id(doc) for doc in documents]

    for i in range(0, len(documents), batch_size):
        quiz_collection.upsert(
            documents=documents[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )
        
def get_all_quiz_documents_with_vectors():
    """퀴즈 생성용 벡터스토어에서 모든 문서 조회"""
    results = quiz_collection.get(include=["embeddings", "documents"])
    return list(zip(results["ids"], results["documents"], results["embeddings"]))
