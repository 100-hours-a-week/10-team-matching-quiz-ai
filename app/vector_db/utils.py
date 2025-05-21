from sentence_transformers import SentenceTransformer
import logging
import warnings
from app.vector_db.config import EMBEDDING_MODEL_NAME, KEYBERT_MODEL_NAME
import numpy as np
from keybert import KeyBERT
import re

# transformers 관련 경고 숨기기
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

_model = None
_kw_model = None  


def get_embedding_model(model_name=EMBEDDING_MODEL_NAME):
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer(model_name, trust_remote_code=True)
            print(f"Embedding 모델 '{model_name}' 로딩 완료")
        except Exception as e:
            print("모델 로딩 실패:", e)
            raise
    return _model


def get_keyword_model():
    global _kw_model
    if _kw_model is None:
        try:
            _kw_model = KeyBERT(model=KEYBERT_MODEL_NAME)
            print("✅ KeyBERT 모델 로딩 완료")
        except Exception as e:
            print("❌ KeyBERT 모델 로딩 실패:", e)
            raise
    return _kw_model


def enrich_question(question: str, keyword: str = None):
    keyword = keyword.lower().strip() if keyword else ""
    enriched = f"면접 질문: {question}"
    if keyword:
        enriched += f"키워드: {keyword}"
    return enriched


def embed_texts(texts, keyword: str = None):
    model = get_embedding_model()
    enriched = [enrich_question(text, keyword) for text in texts]
    return model.encode(enriched, normalize_embeddings=True).tolist()


def clean_keyword_phrase(text: str) -> str:
    import re

    text = text.lower().strip()

    # 문장 끝의 의미 없는 표현 제거
    stopwords_end = [
        "무엇인가요", "무엇인가", "무엇이며", "어떤 것인가요", "차이점은", "에 대해서",
        "인가요", "있나요", "설명해주세요", "해주세요", "하세요", "입니까", "했나요"
    ]
    for stop in stopwords_end:
        text = re.sub(rf"{re.escape(stop)}[\s\?\.\!]*$", "", text)

    # 의미 없는 조사 제거 (짧은 조사만)
    text = re.sub(r"\b(은|는|이|가|을|를|에|의)\b", "", text)

    # 특수문자 제거
    text = re.sub(r"[^\w\s가-힣A-Za-z/\-]", "", text)

    # 괄호 안 의미없는 문자 제거
    text = re.sub(r"\([^가-힣ㄱ-ㅎㅏ-ㅣ]+\)", "", text)

    # 중복 공백 제거
    text = re.sub(r"\s+", " ", text)

    return text.strip()
