from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.config.model_config import ENVIRONMENT, ENABLED_MODELS
import logging
from datetime import datetime
from app.api.question_generator.question_generator_api import router as generate_router
import os
import asyncio
from typing import Optional

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
            logger.info('KeyBERT 모델 로드 완료')
            
            self.model_loaded = True 
            logger.info('모든 임베딩 모델 로드 완료')
            return True  
        except Exception as e:
            logger.error(f'임베딩 모델 로드 실패: {e}') 
            return False
    
    def get_model(self):
        if not self.model_loaded:  
            self.load_embedding_model()
        return self.embedding_model 
    
    def get_keyword_model(self):
        if not self.model_loaded:
            self.load_embedding_model()
        return self.keyword_model
    
    def is_ready(self):
        return self.model_loaded  
    
    def cleanup(self):
        self.embedding_model = None 
        self.keyword_model = None  
        self.model_loaded = False   
        logger.info("임베딩 모델 정리 완료")


class ModelManager:
    def __init__(self):
        self._question_generator_ready = False
        self._vllm_client = None
        self._retry_task: Optional[asyncio.Task] = None
        self._should_retry = True

    async def initialize_models_with_continuous_retry(self) -> bool:
        if "question_generator" not in ENABLED_MODELS:
            return True

        if await self._try_connect():
            return True

        if self._should_retry:
            self._retry_task = asyncio.create_task(self._continuous_retry())
            logger.info("백그라운드에서 vLLM 연결을 계속 시도합니다...")

        return False

    async def _try_connect(self) -> bool:
        try:
            from app.api.question_generator.question_generator_model import (
                initialize_llm, get_llm_engine, check_vllm_health
            )
            
            logger.debug("vLLM 연결 시도 중...")
            
            if not initialize_llm():
                logger.debug("vLLM 클라이언트 초기화 실패")
                return False
                
            self._vllm_client = get_llm_engine()
            if not self._vllm_client:
                logger.debug("vLLM 엔진 가져오기 실패")
                return False
            
            if await check_vllm_health():
                self._question_generator_ready = True
                logger.info("vLLM 연결 성공!")
                return True
            else:
                logger.debug("vLLM 서버 응답 없음")
                return False
                
        except Exception as e:
            logger.debug(f"vLLM 연결 실패: {e}")
            return False

    async def _continuous_retry(self):
        retry_count = 1
        
        while self._should_retry and not self._question_generator_ready:
            try:
                logger.info(f"vLLM 재연결 시도 #{retry_count}")
                
                if await self._try_connect():
                    logger.info(f"vLLM 연결 성공! (재시도 #{retry_count})")
                    break
                
                retry_count += 1
                logger.debug(f"{VLLM_RETRY_DELAY}초 후 재시도...")
                await asyncio.sleep(VLLM_RETRY_DELAY)
                
            except Exception as e:
                logger.error(f"재연결 시도 중 오류: {e}")
                await asyncio.sleep(VLLM_RETRY_DELAY)

    def stop_retry(self):
        self._should_retry = False
        if self._retry_task and not self._retry_task.done():
            self._retry_task.cancel()

    def get_available_models(self):
        return ["question_generator"] if self._question_generator_ready else []
    
    def is_model_available(self, model_name: str) -> bool:
        if model_name == "question_generator":
            return self._question_generator_ready
        return False

    def cleanup(self):
        self.stop_retry()
        self._question_generator_ready = False
        self._vllm_client = None
        logger.info("ModelManager 리소스 정리 완료")


model_manager = ModelManager()
embedding_manager = EmbeddingModelManager()


def initialize_vector_db():
    try:
        from app.vector_db.init_data import init_all_vector_stores
        init_all_vector_stores()
        logger.info("벡터 데이터베이스 초기화 완료")
        return True
    except Exception as e:
        logger.error(f"벡터 데이터베이스 초기화 실패: {e}")
        return False


async def get_system_status():
    available_models = model_manager.get_available_models()
    
    vllm_api_status = False
    try:
        from app.api.question_generator.question_generator_model import check_vllm_health
        vllm_api_status = await check_vllm_health()
    except Exception as e:
        logger.error(f"vLLM API 상태 확인 실패: {e}")

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

    enabled_count = len(ENABLED_MODELS)
    available_count = len(available_models)
    
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
        "embedding_model_status": {"ready": embedding_manager.is_ready()},
        "system_info": {
            "total_enabled": enabled_count,
            "total_available": available_count,
            "availability_ratio": f"{available_count}/{enabled_count}",
        },
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"애플리케이션 시작 (환경: {ENVIRONMENT})")

    logger.info("임베딩 모델 프리로딩 시작...")
    embedding_success = embedding_manager.load_embedding_model()

    logger.info("벡터 데이터베이스 초기화 시작...")
    vector_success = initialize_vector_db()

    logger.info("LLM 모델 초기화 시작...")
    llm_success = await model_manager.initialize_models_with_continuous_retry()

    if llm_success:
        logger.info("vLLM 즉시 연결 성공")
    else:
        logger.warning("vLLM 연결 실패 - 백그라운드에서 계속 시도 중...")

    logger.info(f"초기화 완료 - Embedding: {embedding_success}, Vector DB: {vector_success}, LLM: {llm_success}")
    logger.info("API 요청 대기 중...")

    yield

    logger.info("애플리케이션 종료: 리소스 정리 중...")
    model_manager.cleanup()
    embedding_manager.cleanup()
    logger.info("리소스 정리 완료")


app = FastAPI(
    title="Team Matching Quiz AI",
    description=f"AI 기반 팀 매칭 퀴즈 시스템\n환경: {ENVIRONMENT}\n활성화된 모델: {', '.join(ENABLED_MODELS)}",
    version="1.0.0",
    lifespan=lifespan,
)


def is_model_available(model_name: str) -> bool:
    return model_manager.is_model_available(model_name)


def get_available_models():
    return model_manager.get_available_models()


def get_embedding_model():
    return embedding_manager.get_model()


def is_embedding_ready():
    return embedding_manager.is_ready()


def get_keyword_model():
    return embedding_manager.get_keyword_model()


app.include_router(generate_router, prefix="/interview", tags=["question-generator"])


@app.get("/")
async def root():
    return {
        "service": "Team Matching Quiz AI",
        "version": "1.0.0",
        "environment": ENVIRONMENT,
        "status": "running"
    }


@app.get("/health")
async def health_check():
    return await get_system_status()