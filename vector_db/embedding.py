from sentence_transformers import SentenceTransformer
import os # os 모듈 임포트
import logging

logger = logging.getLogger(__name__)

_model = None

DEFAULT_EMBEDDING_MODEL = 'intfloat/e5-small-v2'
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', DEFAULT_EMBEDDING_MODEL)

def get_model(model_name=EMBEDDING_MODEL_NAME):
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
