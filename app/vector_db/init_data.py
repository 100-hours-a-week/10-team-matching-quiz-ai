import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from app.vector_db.utils import embed_texts
from app.vector_db.chroma_client import (
    save_to_question_vectorstore, 
    save_to_quiz_vectorstore,
    follow_up_collection,
    quiz_collection,
    get_collection_stats
)
from app.vector_db.config import EMBEDDING_MODEL_NAME

def init_question_vector_store_from_csv(csv_path: str):
    """꼬리질문 생성용 벡터스토어 초기화"""
    if follow_up_collection.count() > 0:
        print("꼬리질문 벡터스토어에 이미 데이터가 존재합니다. 초기화를 건너뜁니다.")
        return

    df = pd.read_csv(csv_path, quotechar='"', on_bad_lines='skip')
    questions = df["question"].dropna().tolist()

    print(f"총 {len(questions)}개의 질문을 임베딩 중...")
    embeddings = embed_texts(questions,content_type = "question")
    save_to_question_vectorstore(questions, embeddings)

    print(f"꼬리질문 벡터스토어에 총 {len(questions)}개의 질문이 저장되었습니다.")

def init_quiz_vector_store_from_json(json_path: str):
    """퀴즈 생성용 벡터스토어 초기화"""
    if quiz_collection.count() > 0:
        print("퀴즈 벡터스토어에 이미 데이터가 존재합니다. 초기화를 건너뜁니다.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        rag_data = json.load(f)

    documents = []

    for i, item in enumerate(rag_data):
        doc_parts = []
        
        if item.get('definition'):
            doc_parts.append(f"정의: {item.get('definition', '').strip()}")
        if item.get('how_it_works'):
            doc_parts.append(f"동작 원리: {item.get('how_it_works', '').strip()}")
        
        comparison = item.get("comparison", None)
        if comparison and comparison.strip():
            doc_parts.append(f"비교: {comparison.strip()}")

        full_doc = "\n\n".join(doc_parts)
        
        documents.append(full_doc)

    print(f"총 {len(documents)}개의 퀴즈 문서를 임베딩 중...")
    embeddings = embed_texts(documents)
    save_to_quiz_vectorstore(documents, embeddings)

    print(f"퀴즈 벡터스토어에 총 {len(documents)}개의 문서가 저장되었습니다.")

def init_all_vector_stores():
    """모든 벡터스토어 초기화"""
    print("=== 벡터스토어 초기화 시작 ===")
    
    # 꼬리질문용 벡터스토어 초기화
    init_question_vector_store_from_csv("app/vector_db/question_data.csv")
    
    # 퀴즈용 벡터스토어 초기화
    init_quiz_vector_store_from_json("rag_data.json")
    
    print("=== 벡터스토어 초기화 완료 ===")

if __name__ == "__main__":
    init_all_vector_stores()