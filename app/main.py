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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logger.info(f"감지된 환경: {ENVIRONMENT}")
logger.info(f"활성화된 모델들: {ENABLED_MODELS}")


class ModelType(Enum):
    VLLM = "vllm"


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
            
            try:
                # 현재 실행 중인 이벤트 루프가 있는지 확인
                try:
                    loop = asyncio.get_running_loop()
                    # 이미 실행 중인 루프가 있다면 task로 실행
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, check_vllm_api_health())
                        is_healthy = future.result(timeout=30)
                except RuntimeError:
                    # 실행 중인 루프가 없다면 새로 생성
                    is_healthy = asyncio.run(check_vllm_api_health())
                
                if is_healthy:
                    self._model_data = "vllm_api_client"  # API 클라이언트 표시용
                    self.status = ModelStatus.READY
                    logger.info(f"{self.model_name} API 연결 성공")
                    return True
                else:
                    self.status = ModelStatus.ERROR
                    logger.error(f"{self.model_name} API 서버에 연결할 수 없습니다")
                    return False
            except Exception as e:
                self.status = ModelStatus.ERROR
                logger.error(f"{self.model_name} API 연결 중 오류: {e}")
                return False

        except Exception as e:
            self.status = ModelStatus.ERROR
            logger.error(f"{self.model_name} API 연결 오류: {e}")
            return False

    def cleanup(self):
        """vLLM API 클라이언트 정리"""
        try:
            if self._model_data:
                logger.info(f"{self.model_name} API 연결 정리 중...")
                self._model_data = None
                self.status = ModelStatus.UNINITIALIZED
                logger.info(f"{self.model_name} API 연결 정리 완료")
        except Exception as e:
            logger.error(f"{self.model_name} API 정리 중 오류: {e}")


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

    # 벡터 데이터베이스 초기화
    logger.info("벡터 데이터베이스 초기화 시작...")
    try:
        from app.vector_db.init_data import init_all_vector_stores

        init_all_vector_stores()
        logger.info("벡터 데이터베이스 초기화 완료")
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
    logger.info("리소스 정리 완료")


app = FastAPI(
    title="Team Matching Quiz AI - Question Generator API",
    description=f"vLLM OpenAI 호환 API 기반 질문 생성 서비스\n환경: {ENVIRONMENT}",
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


# API 라우터 포함
app.include_router(generate_router, prefix="/interview", tags=["question-generator"])


@app.get("/health")
async def health_check():
    """시스템 상태 확인"""
    available_models = get_available_models()

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
        vector_db_status = {
            "follow_up_questions": False,
            "quiz_data": False,
            "error": str(e),
        }

    # 시스템 상태 결정
    enabled_count = len(ENABLED_MODELS)
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