from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.config.model_config import ENVIRONMENT, ENABLED_MODELS
import logging
from datetime import datetime
from app.api.question_generator.question_generator_api import router as generate_router
import time
import os

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logger.info(f"감지된 환경: {ENVIRONMENT}")
logger.info(f"활성화된 모델들: {ENABLED_MODELS}")

MAX_VLLM_RETRIES = int(os.getenv("MAX_VLLM_RETRIES", "5"))
VLLM_RETRY_DELAY = int(os.getenv("VLLM_RETRY_DELAY", "10"))

class EmbeddingModelManager:
    def __init__(self):
        self.embedding_model = None
        self.keyword_model = None
        self.model_loaded = False
        
    def load_embedding_model(self):
        if self.model_loaded:
            logger.info('임베딩 모델 이미 로드완료')
            return True 
    
        try:
            logger.info('임베딩 모델 로드 시작')
            from app.vector_db.utils import get_embedding_model, get_keyword_model
            self.embedding_model = get_embedding_model()
            logger.info('임베딩 모델 로드 완료')
            self.keyword_model = get_keyword_model()
            logger.info('keybert 모델 로드 완료')
            self.model_loaded = True 
            logger.info('모든 임베딩 모델 로드 완료')
            return True  
        except Exception as e:
            logger.error(f'임베딩 모델 로드 실패:{e}') 
            return False
    
    def get_model(self):
        """임베딩 모델 반환"""
        if not self.model_loaded:  
            self.load_embedding_model()
        return self.embedding_model 
    
    def get_keyword_model(self):
        """KeyBERT 모델 반환"""
        if not self.model_loaded:
            self.load_embedding_model()
        return self.keyword_model
    
    def is_ready(self):
        """모델 준비 상태"""
        return self.model_loaded  
    
    def cleanup(self):
        """리소스 정리"""
        self.embedding_model = None 
        self.keyword_model = None  
        self.model_loaded = False   
        logger.info("임베딩 모델 정리 완료")

class ModelManager:
    def __init__(self):
        self._question_generator_ready = False
        self._vllm_client = None

    def initialize_models(self) -> bool:
        """vLLM 모델 초기화 (재시도 로직 포함)"""
        if "question_generator" not in ENABLED_MODELS:
            return True

        for attempt in range(1, MAX_VLLM_RETRIES + 1):
            try:
                from app.api.question_generator.question_generator_model import initialize_llm, get_llm_engine

                logger.info(f"question_generator 초기화 시도 {attempt}/{MAX_VLLM_RETRIES}...")
                
                if initialize_llm():
                    self._vllm_client = get_llm_engine()
                    if self._vllm_client:
                        self._question_generator_ready = True
                        logger.info("question_generator API 클라이언트 초기화 완료")
                        return True

                logger.warning(f"question_generator 초기화 실패 (시도 {attempt}/{MAX_VLLM_RETRIES})")
                
                if attempt < MAX_VLLM_RETRIES:
                    logger.info(f"{VLLM_RETRY_DELAY}초 후 재시도...")
                    time.sleep(VLLM_RETRY_DELAY)

            except Exception as e:
                logger.error(f"question_generator 초기화 오류 (시도 {attempt}/{MAX_VLLM_RETRIES}): {e}")
                
                if attempt < MAX_VLLM_RETRIES:
                    logger.info(f"{VLLM_RETRY_DELAY}초 후 재시도...")
                    time.sleep(VLLM_RETRY_DELAY)

        logger.error(f"question_generator 초기화 최종 실패 ({MAX_VLLM_RETRIES}회 시도)")
        return False

    # ✅ 누락된 메서드들 추가
    def get_available_models(self):
        """사용 가능한 모델 목록 반환"""
        available = []
        if self._question_generator_ready:
            available.append("question_generator")
        return available
    
    def is_model_available(self, model_name: str) -> bool:
        """특정 모델 사용 가능 여부 확인"""
        if model_name == "question_generator":
            return self._question_generator_ready
        return False
    
    def cleanup(self):
        """리소스 정리"""
        self._question_generator_ready = False
        self._vllm_client = None
        logger.info("ModelManager 리소스 정리 완료")

# 전역 모델 매니저
model_manager = ModelManager()
embedding_manager = EmbeddingModelManager()

def initialize_vector_db():
    """벡터 데이터베이스 초기화"""
    try:
        from app.vector_db.init_data import init_all_vector_stores
        init_all_vector_stores()
        logger.info("벡터 데이터베이스 초기화 완료")
        return True
    except Exception as e:
        logger.error(f"벡터 데이터베이스 초기화 실패: {e}")
        return False

async def get_system_status():
    """시스템 상태 확인 (health check용)"""
    available_models = model_manager.get_available_models()
    
    # vLLM API 상태 확인
    vllm_api_status = False
    try:
        from app.api.question_generator.question_generator_model import check_vllm_health
        vllm_api_status = await check_vllm_health()
    except Exception as e:
        logger.error(f"vLLM API 상태 확인 실패: {e}")

    # 벡터 데이터베이스 상태 확인
    vector_db_status = {}
    try:
        from app.vector_db.chroma_client import follow_up_collection, quiz_collection
        vector_db_status = {
            "follow_up_questions": follow_up_collection.count() > 0,
            "quiz_data": quiz_collection.count() > 0,
            "follow_up_count": follow_up_collection.count(),
            "quiz_count": quiz_collection.count(),
        }
    except Exception as e:
        logger.error(f"벡터 데이터베이스 상태 확인 실패: {e}")
        vector_db_status = {"error": str(e)}

    # 시스템 상태 결정
    enabled_count = len(ENABLED_MODELS)
    available_count = len(available_models)
    embedding_ready = embedding_manager.is_ready()
    
    if available_count == 0:
        system_status = "unhealthy"
    elif available_count < enabled_count:
        system_status = "degraded"
    else:
        system_status = "healthy"

    return {
        "status": system_status,
        "timestamp": datetime.now().isoformat(),
        "environment": ENVIRONMENT,
        "enabled_models": ENABLED_MODELS,
        "available_models": available_models,
        "vllm_api_status": vllm_api_status,
        "vector_db_status": vector_db_status,
        "embedding_model_status": { 
            "ready": embedding_ready,
        },
        "system_info": {
            "total_enabled": enabled_count,
            "total_available": available_count,
            "availability_ratio": f"{available_count}/{enabled_count}",
        },
    }

@asynccontextmanager  # ✅ 하나만!
async def lifespan(app: FastAPI):
    """애플리케이션 라이프사이클 관리"""
    logger.info(f"애플리케이션 시작 (환경: {ENVIRONMENT})")

    # 1. 임베딩 모델 프리로딩 (KeyBERT 포함)
    logger.info("임베딩 모델 프리로딩 시작...")
    embedding_success = embedding_manager.load_embedding_model()
    if embedding_success:
        logger.info("임베딩 모델 프리로딩 완료")
    else:
        logger.warning("임베딩 모델 프리로딩 실패")

    # 2. 벡터 데이터베이스 초기화
    logger.info("벡터 데이터베이스 초기화 시작...")
    vector_success = initialize_vector_db()

    # 3. vLLM 모델 초기화 (재시도 로직 포함)
    logger.info("LLM 모델 초기화 시작...")
    llm_success = model_manager.initialize_models()

    logger.info(f"초기화 완료 - Embedding: {embedding_success}, Vector DB: {vector_success}, LLM: {llm_success}")
    
    available_models = model_manager.get_available_models()
    logger.info(f"사용 가능한 모델들: {available_models}")
    logger.info("🚀 API 요청 대기 중...")

    yield

    logger.info("애플리케이션 종료: 리소스 정리 중...")
    model_manager.cleanup()
    embedding_manager.cleanup()
    logger.info("리소스 정리 완료")

# FastAPI 앱 생성
app = FastAPI(
    title="Team Matching Quiz AI",
    description=f"AI 기반 팀 매칭 퀴즈 시스템\n환경: {ENVIRONMENT}\n활성화된 모델: {', '.join(ENABLED_MODELS)}",
    version="1.0.0",
    lifespan=lifespan,
)

# 유틸리티 함수들 (API에서 사용)
def is_model_available(model_name: str) -> bool:
    """모델 사용 가능 여부 확인"""
    return model_manager.is_model_available(model_name)

def get_available_models():
    """사용 가능한 모델 목록"""
    return model_manager.get_available_models()

# 전역 임베딩 모델 접근 함수 (다른 모듈에서 사용)
def get_embedding_model():
    """전역 임베딩 모델 반환"""
    return embedding_manager.get_model()

def is_embedding_ready():
    """임베딩 모델 준비 상태"""
    return embedding_manager.is_ready()

def get_keyword_model():
    """전역 KeyBERT 모델 반환"""
    return embedding_manager.get_keyword_model()

# 라우터 등록
app.include_router(generate_router, prefix="/interview", tags=["question-generator"])

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "service": "Team Matching Quiz AI",
        "version": "1.0.0",
        "environment": ENVIRONMENT,
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """시스템 상태 확인"""
    return await get_system_status()