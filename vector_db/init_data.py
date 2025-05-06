import pandas as pd
from embedding import embed_texts
from vector_store import save_to_vectorstore, collection

def init_vector_store_from_csv(csv_path: str):
    # 이미 데이터가 들어있으면 중복 저장하지 않음
    if collection.count() > 0:
        print("이미 벡터스토어에 데이터가 존재합니다. 초기화를 건너뜁니다.")
        return

    df = pd.read_csv(csv_path, quotechar='"', on_bad_lines='skip')
    questions = df["question"].dropna().tolist()

    embeddings = embed_texts(questions)
    save_to_vectorstore(questions, embeddings)

    print(f"총 {len(questions)}개의 질문이 저장되었습니다.")

