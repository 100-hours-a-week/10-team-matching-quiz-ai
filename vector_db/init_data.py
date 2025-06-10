import pandas as pd
from vector_db.utils import embed_texts
from vector_db.chroma_client import save_to_vectorstore, collection

def init_vector_store_from_csv(csv_path: str):
    if collection.count() > 0:
        print("이미 데이터가 존재합니다. 초기화를 건너뜁니다.")
        return

    df = pd.read_csv(csv_path, quotechar='"', on_bad_lines='skip')
    questions = df["question"].dropna().tolist()

    embeddings = embed_texts(questions)
    save_to_vectorstore(questions, embeddings)

    print(f"총 {len(questions)}개의 질문이 저장되었습니다.")