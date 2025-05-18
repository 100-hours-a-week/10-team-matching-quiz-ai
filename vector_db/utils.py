from sentence_transformers import SentenceTransformer
import logging
import warnings
from vector_db.config import EMBEDDING_MODEL_NAME
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
            _kw_model = KeyBERT(model="snunlp/KR-SBERT-V40K-klueNLI-augSTS")
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
    # ✅ 제거해도 되는 진짜 불용어만
    korean_stopwords = [
        "무엇이며", "어떻게", "왜", "이란", "차이점은", "에 대해서", "사용되는가",
        "필요한다", "무엇인가", "있나요", "인가요", "에 대해", "설명해주세요",
        "하세요", "있으신", "해주세요", "요", "까", "말씀"
    ]

    text = text.lower().strip()

    # 문장 끝 불용어 제거
    for stop in korean_stopwords:
        text = re.sub(rf"{re.escape(stop)}[\s\?\.\!]*$", "", text)

    # 조사 제거
    text = re.sub(r"(지만|까지|부터|마다|에서|으로|와|과|과의|에게|에|의|은|는|을|를|이|가|입니까|인가요|었나요|였나요|입니|인가|었나|였나)\b", "", text)

    # 어미/동사형 제거
    text = re.sub(r"(해보신|해봤던|사용해보신|경험해보신|했던|습니다|주세요|해주세요|있습니다|했습니다)", "", text)
    
    # 의미없는 단어 제거
    text = re.sub(r"(대해|대하여)", "", text)
    
    # 특수문자 제거 (/ - 유지)
    text = re.sub(r"[^\w\s가-힣A-Za-z/\-]", "", text)
    
    # 괄호 안 영어/숫자/기호 포함 제거
    text = re.sub(r"\([^가-힣ㄱ-ㅎㅏ-ㅣ]+\)", "", text)


    return text.strip()