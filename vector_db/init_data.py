# vector_db/init_data.py

import pandas as pd
from embedding import embed_texts
from vector_store import save_to_vectorstore

def init_vector_store_from_csv(csv_path: str):
    """
    CSV 파일에서 질문을 읽어 임베딩 후 벡터 DB에 저장.
    """
    df = pd.read_csv(csv_path)
    questions = df["question"].dropna().tolist()

    embeddings = embed_texts(questions)
    save_to_vectorstore(questions, embeddings)

    print(f"✅ 총 {len(questions)}개의 질문이 저장되었습니다.")

# 단독 실행 가능
if __name__ == "__main__":
    init_vector_store_from_csv("./dataset/questions.csv")
