from sentence_transformers import SentenceTransformer
import logging
import warnings
from vector_db.config import EMBEDDING_MODEL_NAME

# transformers 관련 경고 숨기기
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

_model = None

def get_model(model_name=EMBEDDING_MODEL_NAME):
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer(model_name, trust_remote_code=True)
            print(f"Embedding 모델 '{model_name}' 로딩 완료")
        except Exception as e:
            print("모델 로딩 실패:", e)
            raise
    return _model

def enrich_question(question: str, keyword: str = None):
    keyword = keyword.lower().strip() if keyword else ""
    enriched = f"면접 질문: {question}"
    if keyword:
        enriched += f"키워드: {keyword}"
    return enriched

def embed_texts(texts, keyword: str = None):
    model = get_model()
    enriched = [enrich_question(text, keyword) for text in texts]
    return model.encode(enriched, normalize_embeddings=True).tolist()
