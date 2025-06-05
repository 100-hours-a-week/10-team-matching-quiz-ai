from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api import router
from app.config.model_config import ENVIRONMENT, ENABLED_MODELS
import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from enum import Enum

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 환경 정보 로깅
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
        """모델 초기화"""
        pass

    @abstractmethod
    def cleanup(self):
        """모델 정리"""
        pass

    def is_available(self) -> bool:
        """모델 사용 가능 여부"""
        return self.status == ModelStatus.READY and self._model_data is not None

    def get_model_data(self):
        """모델 데이터 반환"""
        return self._model_data


class VLLMModelWrapper(BaseModelWrapper):
    """vLLM 모델 래퍼 - 즉시 로딩"""

    def initialize(self) -> bool:
        """vLLM 모델 즉시 초기화"""
        self.status = ModelStatus.INITIALIZING
        try:
            # GCP 환경 최적화
            if ENVIRONMENT.startswith("gcp-"):
                os.environ["VLLM_GPU_MEMORY_UTILIZATION"] = "0.7"
                os.environ["VLLM_MAX_MODEL_LEN"] = "2048"

            from app.api.question_generator.question_generator_model import (
                initialize_llm,
                get_llm_engine,
            )

            logger.info(f"{self.model_name} (vLLM) 초기화 시도...")
            initialized_engine = initialize_llm()

            if initialized_engine:
                current_engine = get_llm_engine()
                if current_engine is not None:
                    self._model_data = current_engine
                    self.status = ModelStatus.READY
                    logger.info(f"{self.model_name} 초기화 완료")
                    return True

            self.status = ModelStatus.ERROR
            return False

        except Exception as e:
            self.status = ModelStatus.ERROR
            logger.error(f"{self.model_name} 초기화 오류: {e}")
            return False

    def cleanup(self):
        """vLLM 모델 정리"""
        if self._model_data and hasattr(self._model_data, "shutdown_background_loop"):
            try:
                self._model_data.shutdown_background_loop()
                logger.info(f"{self.model_name} 정리 완료")
            except Exception as e:
                logger.error(f"{self.model_name} 정리 오류: {e}")


class LazyTransformersModelWrapper(BaseModelWrapper):
    """Transformers 모델 래퍼 - 지연 로딩"""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.status = ModelStatus.LAZY_READY

    def initialize(self) -> bool:
        """Transformers 모델 지연 초기화"""
        if self.status == ModelStatus.READY:
            return True

        self.status = ModelStatus.INITIALIZING
        try:
            from app.api.quiz_generator.quiz_generator_model import (
                initialize_quiz_model,
            )

            logger.info(f"{self.model_name} (Transformers) 지연 초기화 시도...")
            model, tokenizer = initialize_quiz_model()
            
            self._model_data = {
                "model": model,
                "tokenizer": tokenizer,
                "type": "transformers",
            }
            self.status = ModelStatus.READY
            logger.info(f"{self.model_name} 초기화 완료")
            return True

        except Exception as e:
            self.status = ModelStatus.ERROR
            logger.error(f"{self.model_name} 초기화 오류: {e}")
            return False

    def get_model_data_with_lazy_init(self):
        """지연 초기화 후 모델 데이터 반환"""
        if self.status == ModelStatus.READY:
            return self._model_data

        if self.initialize():
            return self._model_data
        return None

    def is_available(self) -> bool:
        """지연 로딩 모델의 사용 가능 여부"""
        return (
            self.status == ModelStatus.READY and self._model_data is not None
        ) or self.status == ModelStatus.LAZY_READY

    def cleanup(self):
        """Transformers 모델 정리"""
        if self._model_data:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info(f"{self.model_name} 정리 완료")
            except Exception as e:
                logger.error(f"{self.model_name} 정리 오류: {e}")


class ModelManager:
    """모델 관리자 클래스"""

    def __init__(self):
        self._models: Dict[str, BaseModelWrapper] = {}
        # 모델 래퍼 초기화
        self._models["question_generator"] = VLLMModelWrapper("question_generator")
        self._models["quiz_generator"] = LazyTransformersModelWrapper("quiz_generator")

    def initialize_immediate_models(self) -> bool:
        """즉시 로딩 모델들만 초기화 (vLLM)"""
        success = True
        
        # question_generator (vLLM) - 즉시 초기화
        if "question_generator" in ENABLED_MODELS:
            if not self._models["question_generator"].initialize():
                logger.error("question_generator 초기화 실패")
                success = False

        # quiz_generator (Transformers) - 지연 로딩이므로 초기화 표시만
        if "quiz_generator" in ENABLED_MODELS:
            logger.info("quiz_generator는 지연 로딩으로 설정됨")

        return success

    def get_model(self, model_name: str) -> Optional[Any]:
        """모델 데이터 반환 (지연 로딩 지원)"""
        wrapper = self._models.get(model_name)
        if not wrapper:
            return None

        # LazyTransformersModelWrapper인 경우 지연 로딩
        if isinstance(wrapper, LazyTransformersModelWrapper):
            return wrapper.get_model_data_with_lazy_init()

        return wrapper.get_model_data() if wrapper.is_available() else None

    def is_model_available(self, model_name: str) -> bool:
        """모델 사용 가능 여부 확인"""
        wrapper = self._models.get(model_name)
        return wrapper.is_available() if wrapper else False

    def get_available_models(self) -> list:
        """사용 가능한 모델 목록 반환"""
        return [
            name for name, wrapper in self._models.items() if wrapper.is_available()
        ]

    def cleanup_all_models(self):
        """모든 모델 정리"""
        for wrapper in self._models.values():
            if wrapper.status == ModelStatus.READY:
                wrapper.cleanup()


# 전역 모델 관리자
model_manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 라이프사이클 관리"""
    # 시작 로직
    logger.info(f"애플리케이션 시작 (환경: {ENVIRONMENT})")
    
    # 즉시 로딩 모델들 초기화
    logger.info("모델 초기화 시작...")
    initialization_success = model_manager.initialize_immediate_models()

    if initialization_success:
        logger.info("모델 초기화 성공")
    else:
        logger.warning("일부 모델 초기화 실패")

    available_models = model_manager.get_available_models()
    logger.info(f"사용 가능한 모델들: {available_models}")

    # 애플리케이션 실행
    yield

    # 종료 로직
    logger.info("애플리케이션 종료: 리소스 정리 중...")
    model_manager.cleanup_all_models()
    logger.info("리소스 정리 완료")


# FastAPI 애플리케이션 생성
app = FastAPI(
    title="Team Matching Quiz AI",
    description=f"AI 기반 팀 매칭 퀴즈 시스템\n환경: {ENVIRONMENT}\n활성화된 모델: {', '.join(ENABLED_MODELS)}",
    version="1.0.0",
    lifespan=lifespan,
)

# 라우터 등록
app.include_router(router)


# =============================================================================
# 유틸리티 함수들
# =============================================================================

def get_model(model_name: str):
    """
    특정 모델 반환
    
    Args:
        model_name (str): 모델 이름 ('question_generator' 또는 'quiz_generator')
        
    Returns:
        모델 객체 또는 None
    """
    return model_manager.get_model(model_name)


def is_model_available(model_name: str) -> bool:
    """
    모델 사용 가능 여부 확인
    
    Args:
        model_name (str): 모델 이름
        
    Returns:
        bool: 모델 사용 가능 여부
    """
    return model_manager.is_model_available(model_name)


def get_available_models():
    """
    사용 가능한 모델 목록 반환
    
    Returns:
        list: 사용 가능한 모델 이름 목록
    """
    return model_manager.get_available_models()


@app.get("/health")
async def health_check():
    """시스템 상태 확인"""
    available_models = get_available_models()
    
    return {
        "status": "healthy",
        "timestamp": "2025-06-05",
        "environment": ENVIRONMENT,
        "enabled_models": ENABLED_MODELS,
        "available_models": available_models,
        "models_status": {
            model: is_model_available(model) for model in ["question_generator", "quiz_generator"]
        },
        "system_info": {
            "total_enabled": len(ENABLED_MODELS),
            "total_available": len(available_models),
        }
    }