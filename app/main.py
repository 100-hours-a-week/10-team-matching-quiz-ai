from fastapi import FastAPI, APIRouter
from contextlib import asynccontextmanager
from app.config.model_config import ENVIRONMENT, ENABLED_MODELS
import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from enum import Enum
from datetime import datetime
from app.api.question_generator.question_generator_api import router as generate_router
# Quiz API 라우터 제거 - Worker에서만 처리
# from app.api.quiz_generator.quiz_generator_api import router as quiz_router
from app import rabbitmq_producer  # RabbitMQ producer import

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logger.info(f"감지된 환경: {ENVIRONMENT}")
logger.info(f"활성화된 모델들: {ENABLED_MODELS}")


class ModelType(Enum):
    VLLM = "vllm"
    TRANSFORMERS = "transformers"


class ModelStatus(Enum):
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    LAZY_READY = "lazy_ready"


class BaseModelWrapper(ABC):
    """모든 모델의 기본 래퍼 클래스"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.status = ModelStatus.UNINITIALIZED
        self._model_data = None

    @abstractmethod
    def initialize(self) -> bool:
        pass

    @abstractmethod
    def cleanup(self):
        pass

    def is_available(self) -> bool:
        return self.status == ModelStatus.READY and self._model_data is not None

    def get_model_data(self):
        return self._model_data


class VLLMApiWrapper(BaseModelWrapper):
    """vLLM API 래퍼 - HTTP API 호출 방식"""

    def initialize(self) -> bool:
        self.status = ModelStatus.INITIALIZING
        try:
            # vLLM API 서버 상태 확인
            import asyncio
            from app.api.question_generator.question_generator_model_api import check_vllm_api_health
            
            logger.info(f"{self.model_name} (vLLM API) 연결 확인 중...")
            
            # 비동기 함수를 동기적으로 실행
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                is_healthy = loop.run_until_complete(check_vllm_api_health())
                if is_healthy:
                    self._model_data = "vllm_api_client"  # API 클라이언트 표시용
                    self.status = ModelStatus.READY
                    logger.info(f"{self.model_name} API 연결 성공")
                    return True
                else:
                    self.status = ModelStatus.ERROR
                    logger.error(f"{self.model_name} API 서버에 연결할 수 없습니다")
                    return False
            finally:
                loop.close()

        except Exception as e:
            self.status = ModelStatus.ERROR
            logger.error(f"{self.model_name} API 연결 오류: {e}")
            return False

    def cleanup(self):
        # API 클라이언트는 별도 정리가 필요 없음
        logger.info(f"{self.model_name} API 클라이언트 정리 완료")


class ModelManager:
    """Question Generator API 모델 관리자 클래스"""

    def __init__(self):
        self._models: Dict[str, BaseModelWrapper] = {
            "question_generator": VLLMApiWrapper("question_generator"),
        }

    def initialize_models(self) -> bool:
        """활성화된 모델들을 초기화"""
        success = True
        if "question_generator" in ENABLED_MODELS:
            if not self._models["question_generator"].initialize():
                logger.error("question_generator API 연결 실패")
                success = False
        return success

    def get_model(self, model_name: str) -> Optional[Any]:
        wrapper = self._models.get(model_name)
        if not wrapper:
            return None

        return wrapper.get_model_data() if wrapper.is_available() else None

    def is_model_available(self, model_name: str) -> bool:
        wrapper = self._models.get(model_name)
        return wrapper.is_available() if wrapper else False

    def get_available_models(self) -> list:
        return [
            name for name, wrapper in self._models.items() if wrapper.is_available()
        ]

    def cleanup_all_models(self):
        for wrapper in self._models.values():
            if wrapper.status == ModelStatus.READY:
                wrapper.cleanup()


model_manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 라이프사이클 관리"""
    logger.info(f"애플리케이션 시작 (환경: {ENVIRONMENT})")

    # RabbitMQ 연결 초기화
    logger.info("RabbitMQ 연결 초기화 시작...")
    try:
        await rabbitmq_producer.get_rabbitmq_connection()
        await rabbitmq_producer.get_rabbitmq_channel()  # 채널 및 Exchange 선언 포함
        logger.info("RabbitMQ 연결 초기화 완료.")
    except Exception as e:
        logger.error(f"RabbitMQ 연결 초기화 실패: {e}")

    # 벡터 데이터베이스 초기화 (Quiz Worker와 공유)
    logger.info("벡터 데이터베이스 초기화 시작...")
    try:
        from app.vector_db.init_data import init_all_vector_stores

        init_all_vector_stores()
        logger.info("벡터 데이터베이스 초기화 완료 (Quiz Worker와 공유)")
    except Exception as e:
        logger.error(f"벡터 데이터베이스 초기화 실패: {e}")

    # Question Generator API 초기화
    logger.info("Question Generator API 연결 확인...")
    initialization_success = model_manager.initialize_models()

    if initialization_success:
        logger.info("Question Generator API 연결 성공")
    else:
        logger.warning("Question Generator API 연결 실패")

    available_models = model_manager.get_available_models()
    logger.info(f"API 서버에서 사용 가능한 모델들: {available_models}")

    yield

    logger.info("애플리케이션 종료: 리소스 정리 중...")
    model_manager.cleanup_all_models()

    # RabbitMQ 연결 종료
    logger.info("RabbitMQ 연결 종료 중...")
    try:
        await rabbitmq_producer.close_rabbitmq_connection()
        logger.info("RabbitMQ 연결 종료 완료.")
    except Exception as e:
        logger.error(f"RabbitMQ 연결 종료 실패: {e}")

    logger.info("리소스 정리 완료")


app = FastAPI(
    title="Team Matching Quiz AI - Question Generator API",
    description=f"vLLM OpenAI 호환 API 기반 질문 생성 서비스\n환경: {ENVIRONMENT}\n활성화된 모델: {', '.join(ENABLED_MODELS)}",
    version="2.0.0",
    lifespan=lifespan,
)


# 유틸리티 함수들
def get_model(model_name: str):
    return model_manager.get_model(model_name)


def is_model_available(model_name: str) -> bool:
    return model_manager.is_model_available(model_name)


def get_available_models():
    return model_manager.get_available_models()


# API 라우터 포함 (Question Generator만)
app.include_router(generate_router, prefix="/interview", tags=["question-generator"])
# Quiz API는 Worker에서만 처리하므로 제거
# app.include_router(quiz_router, prefix="/quiz", tags=["quiz"])


@app.get("/health")
async def health_check():
    """시스템 상태 확인"""
    available_models = get_available_models()

    # 벡터 데이터베이스 상태 확인 (Quiz Worker와 공유)
    vector_db_status = {}
    try:
        from app.vector_db.chroma_client import follow_up_collection, quiz_collection

        vector_db_status = {
            "follow_up_questions": follow_up_collection.count() > 0,
            "quiz_data": quiz_collection.count() > 0,
            "follow_up_count": follow_up_collection.count(),
            "quiz_count": quiz_collection.count(),
            "shared_with_workers": True,
        }
    except Exception as e:
        logger.error(f"벡터 데이터베이스 상태 확인 실패: {e}")
        vector_db_status = {
            "follow_up_questions": False,
            "quiz_data": False,
            "error": str(e),
        }

    # Worker 상태 확인 (RabbitMQ 큐 기반)
    worker_status = {}
    try:
        # RabbitMQ 큐 상태로 Worker 상태 추정
        quiz_worker_status = "active"  # 실제로는 큐 메시지 수나 컨슈머 수로 판단
        stt_worker_status = "active"   # 실제로는 큐 메시지 수나 컨슈머 수로 판단
        
        worker_status = {
            "quiz_worker": quiz_worker_status,
            "stt_worker": stt_worker_status,
        }
    except Exception as e:
        logger.error(f"Worker 상태 확인 실패: {e}")
        worker_status = {
            "quiz_worker": "unknown",
            "stt_worker": "unknown",
            "error": str(e),
        }

    # 시스템 상태 결정 (API 서버 모델만 고려)
    api_enabled_models = [model for model in ENABLED_MODELS if model != "quiz_generator"]
    enabled_count = len(api_enabled_models)
    available_count = len(available_models)

    system_status = "healthy"
    if available_count == 0 and enabled_count > 0:
        system_status = "unhealthy"
    elif available_count < enabled_count:
        system_status = "degraded"

        return {
        "status": system_status,
        "timestamp": datetime.now().isoformat(),
        "environment": ENVIRONMENT,
        "service": "question-generator-api",
        "architecture": "vllm_openai_compatible_api",
        "enabled_models": ENABLED_MODELS,
        "available_models": available_models,
        "models_status": {
            "question_generator": is_model_available("question_generator"),
        },
        "vector_db_status": vector_db_status,
        "system_info": {
            "enabled": enabled_count,
            "available": available_count,
            "version": "2.0.0",
        },
    }