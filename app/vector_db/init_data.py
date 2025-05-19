__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import pandas as pd
from utils import embed_texts
from chroma_client import save_to_vectorstore, collection
import pysqlite3
import sys

def init_vector_store_from_csv(csv_path: str):
    if collection.count() > 0:
        print("이미 데이터가 존재합니다. 초기화를 건너뜁니다.")
        return

    df = pd.read_csv(csv_path, quotechar='"', on_bad_lines='skip')
    questions = df["question"].dropna().tolist()

    embeddings = embed_texts(questions)
    save_to_vectorstore(questions, embeddings)

    print(f"총 {len(questions)}개의 질문이 저장되었습니다.")

if __name__ == "__main__":
    init_vector_store_from_csv("vector_db/question_data.csv")