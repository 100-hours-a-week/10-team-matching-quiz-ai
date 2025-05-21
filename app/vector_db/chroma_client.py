import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
import hashlib
from app.vector_db.config import DB_PATH, COLLECTION_NAME

chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

def generate_id(text: str) -> str:
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def save_to_vectorstore(questions, embeddings, batch_size=1024):
    ids = [generate_id(q) for q in questions]

    for i in range(0, len(questions), batch_size):
        collection.upsert(
            documents=questions[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )

def get_all_documents_with_vectors():
    results = collection.get(include=["embeddings", "documents"])
    return list(zip(results["ids"], results["documents"], results["embeddings"]))
