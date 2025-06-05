from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api import router
from app.config.model_config import ModelConfig
import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, TYPE_CHECKING
from enum import Enum
import sys

# TYPE_CHECKING을 사용하여 순환 import 방지
if TYPE_CHECKING:
    from app.vector_db.utils import get_embedding_model, get_keyword_model

logger_sys = logging.getLogger("sys_path_check")
logger_sys.info(f"Current sys.path: {sys.path}")

# 환경에 따른 로깅 레벨 설정
log_level = (
    logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO
)

logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        # 필요시 파일 핸들러 추가 가능
    ],
)
logger = logging.getLogger(__name__)

# 환경에 따른 자동 모델 선택
ENVIRONMENT = ModelConfig.get_environment()
ENABLED_MODELS = ModelConfig.get_enabled_models()
MEMORY_LIMIT = ModelConfig.get_model_memory_limit()

logger.info(f"감지된 환경: {ENVIRONMENT}")
logger.info(f"활성화된 모델들: {ENABLED_MODELS}")
logger.info(f"메모리 제한: {MEMORY_LIMIT}")


class ModelType(Enum):
    """모델 타입 정의"""

    VLLM = "vllm"
    TRANSFORMERS = "transformers"
    OPENAI_API = "openai_api"


class ModelStatus(Enum):
    """모델 상태 정의"""

    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    LAZY_READY = "lazy_ready"  # 지연 로딩 준비 상태


class BaseModelWrapper(ABC):
    """모든 모델의 기본 래퍼 클래스"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.status = ModelStatus.UNINITIALIZED
        self._model_data = None
        self._error_message = None

    @abstractmethod
    def initialize(self) -> bool:
        """모델 초기화"""
        pass

    @abstractmethod
    def cleanup(self):
        """모델 정리"""
        pass

    @property
    @abstractmethod
    def model_type(self) -> ModelType:
        """모델 타입 반환"""
        pass

    def is_available(self) -> bool:
        """모델 사용 가능 여부"""
        return self.status == ModelStatus.READY and self._model_data is not None

    def get_model_data(self):
        """모델 데이터 반환"""
        return self._model_data

    def get_status_info(self) -> Dict[str, Any]:
        """모델 상태 정보 반환"""
        return {
            "name": self.model_name,
            "type": self.model_type.value,
            "status": self.status.value,
            "available": self.is_available(),
            "error": self._error_message,
        }


class VLLMModelWrapper(BaseModelWrapper):
    """vLLM 모델 래퍼 - 즉시 로딩"""

    @property
    def model_type(self) -> ModelType:
        return ModelType.VLLM

    def initialize(self) -> bool:
        """vLLM 모델 즉시 초기화"""
        self.status = ModelStatus.INITIALIZING
        try:
            # 환경별 최적화 설정 적용
            vllm_config = ModelConfig.get_vllm_config()
            if ModelConfig.is_gcp_environment():
                os.environ["VLLM_GPU_MEMORY_UTILIZATION"] = str(
                    vllm_config["gpu_memory_utilization"]
                )
                os.environ["VLLM_MAX_MODEL_LEN"] = str(vllm_config["max_model_len"])

            from app.api.question_generator.question_generator_model import (
                initialize_llm,
                llm as global_llm_engine,
            )

            logger.info(f"{self.model_name} (vLLM) 즉시 초기화를 시도합니다...")
            initialize_llm()

            if global_llm_engine:
                self._model_data = global_llm_engine
                self.status = ModelStatus.READY
                logger.info(
                    f"{self.model_name} 즉시 초기화가 성공적으로 완료되었습니다."
                )
                return True
            else:
                self._error_message = "초기화 후 global_llm_engine이 None입니다."
                self.status = ModelStatus.ERROR
                logger.error(f"{self.model_name}: {self._error_message}")
                return False

        except Exception as e:
            self._error_message = str(e)
            self.status = ModelStatus.ERROR
            logger.error(
                f"{self.model_name} 즉시 초기화 중 오류 발생: {e}", exc_info=True
            )
            return False

    def cleanup(self):
        """vLLM 모델 정리"""
        if self._model_data:
            try:
                if hasattr(self._model_data, "shutdown_background_loop"):
                    self._model_data.shutdown_background_loop()
                logger.info(f"{self.model_name} 정리 완료")
            except Exception as e:
                logger.error(f"{self.model_name} 정리 중 오류: {e}")


class LazyTransformersModelWrapper(BaseModelWrapper):
    """Transformers 모델 래퍼 - 지연 로딩"""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self._initialization_attempted = False
        # 지연 로딩 준비 상태로 설정
        self.status = ModelStatus.LAZY_READY

    @property
    def model_type(self) -> ModelType:
        return ModelType.TRANSFORMERS

    def initialize(self) -> bool:
        """Transformers 모델 지연 초기화"""
        if self.status == ModelStatus.READY:
            logger.info(f"{self.model_name} 이미 초기화되어 있습니다.")
            return True

        if self._initialization_attempted and self.status == ModelStatus.ERROR:
            logger.warning(f"{self.model_name} 이전 초기화 실패로 인해 스킵합니다.")
            return False

        self.status = ModelStatus.INITIALIZING
        self._initialization_attempted = True

        try:
            from app.api.quiz_generator.quiz_generator_model import (
                initialize_quiz_model,
            )

            logger.info(f"{self.model_name} (Transformers) 지연 초기화를 시도합니다...")

            # 환경별 최적화 설정 적용
            transformers_config = ModelConfig.get_transformers_config()
            if ModelConfig.is_gcp_environment():
                os.environ["TORCH_DTYPE"] = transformers_config["torch_dtype"]
                os.environ["LOW_CPU_MEM_USAGE"] = str(
                    transformers_config["low_cpu_mem_usage"]
                ).lower()

            model, tokenizer = initialize_quiz_model()
            self._model_data = {
                "model": model,
                "tokenizer": tokenizer,
                "type": "transformers",
            }
            self.status = ModelStatus.READY
            logger.info(f"{self.model_name} 지연 초기화가 성공적으로 완료되었습니다.")
            return True

        except Exception as e:
            self._error_message = str(e)
            self.status = ModelStatus.ERROR
            logger.error(
                f"{self.model_name} 지연 초기화 중 오류 발생: {e}", exc_info=True
            )
            return False

    def get_model_data_with_lazy_init(self):
        """지연 초기화 후 모델 데이터 반환"""
        if self.status == ModelStatus.READY:
            return self._model_data

        logger.info(f"{self.model_name} 요청 시점에 지연 초기화를 수행합니다...")
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
                # torch import를 안전하게 처리
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                except ImportError:
                    logger.warning(
                        "torch 모듈을 찾을 수 없어 GPU 메모리 정리를 건너뜁니다."
                    )

                # 모델 데이터 초기화
                self._model_data = None
                logger.info(f"{self.model_name} 정리 완료")
            except Exception as e:
                logger.error(f"{self.model_name} 정리 중 오류: {e}")


class ModelManager:
    """모델 관리자 클래스"""

    def __init__(self):
        self._models: Dict[str, BaseModelWrapper] = {}
        self._initialize_wrappers()

    def _initialize_wrappers(self):
        """모델 래퍼 초기화"""
        self._models["question_generator"] = VLLMModelWrapper("question_generator")
        self._models["quiz_generator"] = LazyTransformersModelWrapper("quiz_generator")

    def initialize_model(self, model_name: str) -> bool:
        """특정 모델 초기화"""
        if model_name not in self._models:
            logger.error(f"알 수 없는 모델 이름: {model_name}")
            return False

        return self._models[model_name].initialize()

    def initialize_immediate_models(self) -> bool:
        """즉시 로딩 모델들만 초기화 (vLLM)"""
        initialization_success = False

        # vLLM만 즉시 초기화
        if "question_generator" in ENABLED_MODELS:
            logger.info("question_generator (vLLM) 즉시 초기화를 시도합니다...")
            try:
                success = self.initialize_model("question_generator")
                if success:
                    initialization_success = True
                    logger.info("question_generator 즉시 초기화 성공")
                else:
                    logger.warning("question_generator 즉시 초기화 실패")
            except Exception as e:
                logger.error(
                    f"question_generator 즉시 초기화 중 예외 발생: {e}", exc_info=True
                )

        # quiz_generator는 지연 로딩 준비만
        if "quiz_generator" in ENABLED_MODELS:
            quiz_wrapper = self._models.get("quiz_generator")
            if quiz_wrapper and quiz_wrapper.status == ModelStatus.LAZY_READY:
                logger.info("quiz_generator (Transformers) 지연 로딩 모드로 준비됨")
                initialization_success = True

        return initialization_success

    def get_model(self, model_name: str) -> Optional[Any]:
        """모델 데이터 반환 (지연 로딩 지원)"""
        wrapper = self._models.get(model_name)
        if not wrapper:
            return None

        # 지연 로딩 래퍼인 경우
        if hasattr(wrapper, "get_model_data_with_lazy_init"):
            return wrapper.get_model_data_with_lazy_init()

        # 일반 래퍼인 경우
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

    def get_model_wrapper(self, model_name: str) -> Optional[BaseModelWrapper]:
        """모델 래퍼 반환 (고급 사용)"""
        return self._models.get(model_name)

    def get_all_model_status(self) -> Dict[str, Dict[str, Any]]:
        """모든 모델의 상태 정보 반환"""
        return {
            name: wrapper.get_status_info() for name, wrapper in self._models.items()
        }

    def cleanup_all_models(self):
        """모든 모델 정리"""
        logger.info("모든 모델 정리를 시작합니다...")
        cleanup_count = 0

        for name, wrapper in self._models.items():
            if wrapper.status == ModelStatus.READY:
                try:
                    logger.info(f"{name} 모델 정리 중...")
                    wrapper.cleanup()
                    cleanup_count += 1
                except Exception as e:
                    logger.error(f"{name} 모델 정리 중 오류 발생: {e}", exc_info=True)

        logger.info(f"총 {cleanup_count}개 모델이 정리되었습니다.")


# 전역 모델 관리자
model_manager = ModelManager()

# 기존 호환성을 위한 전역 변수들
GLOBAL_MODELS = {
    "question_generator": None,
    "quiz_generator": None,
}

VECTOR_DB_AVAILABLE = False
try:
    from app.vector_db.utils import get_embedding_model, get_keyword_model

    VECTOR_DB_AVAILABLE = True
    logger.info("Vector DB 모듈이 성공적으로 로드되었습니다.")
except ImportError:
    logger.warning("Vector DB 모듈을 찾을 수 없습니다.")


def is_gcp_environment() -> bool:
    """GCP 환경인지 확인"""
    return ModelConfig.is_gcp_environment()


def _sync_global_models():
    """모델 관리자와 기존 GLOBAL_MODELS 동기화 (하위 호환성)"""
    for model_name in GLOBAL_MODELS.keys():
        if model_name == "question_generator":
            # vLLM은 즉시 로드된 모델 반환
            GLOBAL_MODELS[model_name] = model_manager.get_model(model_name)
        elif model_name == "quiz_generator":
            # Transformers는 지연 로딩이므로 None으로 유지 (요청 시점에 로드됨)
            GLOBAL_MODELS[model_name] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"애플리케이션 라이프사이클 시작 (환경: {ENVIRONMENT})")

    # GCP 환경 최적화
    if is_gcp_environment():
        logger.info("GCP 환경에서 최적화된 설정을 적용합니다.")
        import multiprocessing

        cpu_count = multiprocessing.cpu_count()
        logger.info(f"사용 가능한 CPU 코어: {cpu_count}")

    # 즉시 로딩 모델들만 초기화 (vLLM)
    logger.info("즉시 로딩 모델들 초기화 시작...")
    initialization_success = model_manager.initialize_immediate_models()

    # 최소 하나의 모델이라도 준비되어야 함
    if not initialization_success:
        logger.error("어떤 모델도 준비되지 않았습니다. 폴백 모드로 전환합니다.")
        try:
            # 폴백으로 quiz_generator라도 지연 로딩 준비
            quiz_wrapper = model_manager.get_model_wrapper("quiz_generator")
            if quiz_wrapper:
                quiz_wrapper.status = ModelStatus.LAZY_READY
                logger.info("폴백: quiz_generator 지연 로딩 모드로 설정됨")
        except Exception as e:
            logger.critical(f"폴백 모델 설정도 실패했습니다: {e}")

    # 기존 호환성을 위한 동기화
    _sync_global_models()

    # Vector DB 초기화 (vLLM이 로드된 경우에만)
    if VECTOR_DB_AVAILABLE and model_manager.is_model_available("question_generator"):
        logger.info("Vector DB 관련 모델 초기화를 시도합니다...")
        try:
            get_embedding_model()
            get_keyword_model()
            logger.info("Vector DB 관련 모델 초기화가 성공적으로 완료되었습니다.")
        except Exception as e:
            logger.error(f"Vector DB 관련 모델 초기화 중 오류 발생: {e}", exc_info=True)

    # 상태 로깅
    available_models = model_manager.get_available_models()
    logger.info(f"사용 가능한 모델들: {available_models}")
    logger.info("애플리케이션 초기화 완료:")
    logger.info("  - vLLM (question_generator): 즉시 로드됨")
    logger.info("  - Transformers (quiz_generator): 요청 시 지연 로드됨")

    yield

    # 정리 단계
    logger.info("애플리케이션 라이프사이클 종료: 리소스 정리 시작...")
    model_manager.cleanup_all_models()
    logger.info("애플리케이션 리소스 정리가 완료되었습니다.")


app = FastAPI(
    title="Team Matching Quiz AI",
    description=f"Environment: {ENVIRONMENT}, Models: {ENABLED_MODELS}",
    lifespan=lifespan,
)
app.include_router(router)


# 기존 호환성을 위한 유틸리티 함수들
def get_model(model_name: str):
    """특정 모델 반환 (지연 로딩 지원)"""
    return model_manager.get_model(model_name)


def is_model_available(model_name: str) -> bool:
    """모델 사용 가능 여부 확인 (지연 로딩 지원)"""
    return model_manager.is_model_available(model_name)


def get_available_models():
    """사용 가능한 모델 목록 반환"""
    return model_manager.get_available_models()


# 새로운 고급 인터페이스 함수들
def get_model_wrapper(model_name: str) -> Optional[BaseModelWrapper]:
    """모델 래퍼 반환 (타입별 세부 제어용)"""
    return model_manager.get_model_wrapper(model_name)


def get_model_manager() -> ModelManager:
    """모델 관리자 반환"""
    return model_manager


# 개선된 헬스체크 엔드포인트
@app.get("/health")
async def health_check():
    """기본 헬스체크 - 간단한 상태 정보"""
    config_summary = ModelConfig.get_config_summary()
    return {
        "status": "healthy",
        "environment": ENVIRONMENT,
        "enabled_models": ENABLED_MODELS,
        "available_models": get_available_models(),
        "memory_limit": MEMORY_LIMIT,
        "model_details": model_manager.get_all_model_status(),
        "loading_strategy": {"vllm": "immediate", "transformers": "lazy"},
        "config": config_summary,
    }


# 시스템 상세 상태 확인 엔드포인트
@app.get("/health/detailed")
async def detailed_health_check():
    """상세 헬스체크 - 포괄적인 시스템 상태"""
    return check_system_health()


# 모델별 상세 상태 확인 엔드포인트
@app.get("/health/models")
async def models_health_check():
    """모델 전용 헬스체크"""
    return {
        "models": model_manager.get_all_model_status(),
        "summary": {
            "total_models": len(model_manager._models),
            "available_models": len(get_available_models()),
            "enabled_models": ENABLED_MODELS,
            "memory_info": get_memory_info(),
        },
    }


# 시스템 유틸리티 함수들
def get_memory_info() -> Dict[str, Any]:
    """시스템 메모리 정보 조회"""
    memory_info = {"backend": "unknown", "available": False}

    try:
        # torch를 안전하게 import
        try:
            import torch

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                memory_info = {
                    "backend": "cuda",
                    "available": True,
                    "allocated_gb": round(allocated, 2),
                    "reserved_gb": round(reserved, 2),
                    "total_gb": round(total, 2),
                    "free_gb": round(total - reserved, 2),
                    "utilization_percent": round(reserved / total * 100, 1),
                }
            elif torch.backends.mps.is_available():
                memory_info = {
                    "backend": "mps",
                    "available": True,
                    "memory_monitoring": "limited",
                }
            else:
                memory_info = {"backend": "cpu", "available": True, "unlimited": True}
        except ImportError:
            memory_info = {
                "backend": "cpu",
                "available": False,
                "error": "torch not available",
            }
    except Exception as e:
        logger.warning(f"메모리 정보 조회 실패: {e}")
        memory_info["error"] = str(e)

    return memory_info


def check_system_health() -> Dict[str, Any]:
    """시스템 전반적인 상태 체크"""

    # Langfuse 연결 상태 확인
    langfuse_status = check_langfuse_connection()

    health_status = {
        "timestamp": logger.name,  # 현재 시간 대신 로거 이름 사용
        "environment": ENVIRONMENT,
        "models": {
            "total": len(model_manager._models),
            "available": len(model_manager.get_available_models()),
            "status": model_manager.get_all_model_status(),
        },
        "memory": get_memory_info(),
        "vector_db": VECTOR_DB_AVAILABLE,
        "langfuse": langfuse_status,
        "config": ModelConfig.get_config_summary(),
    }

    # 전반적인 건강 상태 판단
    available_models = len(model_manager.get_available_models())
    if available_models == 0:
        health_status["overall_status"] = "critical"
        health_status["message"] = "사용 가능한 모델이 없습니다"
    elif available_models < len(ENABLED_MODELS):
        health_status["overall_status"] = "warning"
        health_status["message"] = "일부 모델이 사용 불가능합니다"
    else:
        health_status["overall_status"] = "healthy"
        health_status["message"] = "모든 시스템이 정상 작동 중입니다"

    return health_status


def check_langfuse_connection() -> Dict[str, Any]:
    """Langfuse 연결 상태 확인"""
    try:
        from langfuse import Langfuse

        # Question Generator Langfuse 확인
        question_langfuse = Langfuse(
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            host=os.getenv("LANGFUSE_HOST"),
        )

        # Quiz Generator Langfuse 확인
        quiz_langfuse = Langfuse(
            secret_key=os.getenv("QUIZ_LANGFUSE_SECRET_KEY"),
            public_key=os.getenv("QUIZ_LANGFUSE_PUBLIC_KEY"),
            host=os.getenv("LANGFUSE_HOST"),
        )

        return {
            "available": True,
            "question_generator": {
                "host": os.getenv("LANGFUSE_HOST"),
                "public_key": (
                    os.getenv("LANGFUSE_PUBLIC_KEY")[:10] + "..."
                    if os.getenv("LANGFUSE_PUBLIC_KEY")
                    else None
                ),
            },
            "quiz_generator": {
                "host": os.getenv("LANGFUSE_HOST"),
                "public_key": (
                    os.getenv("QUIZ_LANGFUSE_PUBLIC_KEY")[:10] + "..."
                    if os.getenv("QUIZ_LANGFUSE_PUBLIC_KEY")
                    else None
                ),
            },
        }
    except ImportError:
        return {"available": False, "error": "Langfuse 모듈이 설치되지 않음"}
    except Exception as e:
        return {"available": False, "error": f"Langfuse 연결 확인 실패: {str(e)}"}
