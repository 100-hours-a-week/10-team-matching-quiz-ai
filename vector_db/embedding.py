from sentence_transformers import SentenceTransformer

_model = None

def get_model(model_name='intfloat/e5-small-v2'):
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer(model_name)
            print(f"모델 '{model_name}' 로딩 완료")
        except Exception as e:
            print("모델 로딩 실패:", e)
            raise
    return _model

# 문장 디테일 보강
def enrich_question(question: str, keyword: str = None):
    if keyword:
        return f"면접 질문: {question} 관련 키워드: {keyword}"
    return f"면접 질문: {question}"

# embed_texts 내부에서 enrich_question 적용
def embed_texts(texts, keyword: str = None):
    model = get_model()
    enriched = [enrich_question(text, keyword) for text in texts]
    return model.encode(enriched, normalize_embeddings=True).tolist()

# 단독 실행 테스트용
if __name__ == "__main__":
    sample = ["FastAPI는 어떤 프레임워크인가요?", "Python의 특징은 무엇인가요?"]
    vectors = embed_texts(sample, keyword="Python")
    print("벡터 수:", len(vectors))
