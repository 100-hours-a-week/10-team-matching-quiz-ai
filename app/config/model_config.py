"""
모델 환경 설정 관리 모듈
"""

import os
from typing import List, Optional
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)


class ModelConfig:
    """모델 환경 설정 관리 클래스"""

    # 기본값 정의
    DEFAULT_ENVIRONMENT = "local"
    DEFAULT_MEMORY_LIMIT = "8GB"

    # 환경별 활성화 모델 매핑
    ENVIRONMENT_MODEL_MAP = {
        "local": ["question_generator", "quiz_generator"],
        "dev": ["question_generator", "quiz_generator"],
        "prod": ["question_generator", "quiz_generator"],
        "gcp-gke": ["question_generator", "quiz_generator"],
        "gcp-vm": ["question_generator", "quiz_generator"],
        "test": ["quiz_generator"],  # 테스트 환경에서는 가벼운 모델만
    }

    # 환경별 메모리 제한
    ENVIRONMENT_MEMORY_MAP = {
        "local": "8GB",
        "dev": "16GB",
        "prod": "32GB",
        "gcp-gke": "24GB",
        "gcp-vm": "16GB",
        "test": "4GB",
    }

    @classmethod
    def get_environment(cls) -> str:
        """현재 실행 환경 반환"""
        environment = os.getenv("ENVIRONMENT", cls.DEFAULT_ENVIRONMENT).lower()

        # 환경 유효성 검사
        if environment not in cls.ENVIRONMENT_MODEL_MAP:
            logger.warning(
                f"알 수 없는 환경 '{environment}', 기본값 '{cls.DEFAULT_ENVIRONMENT}' 사용"
            )
            return cls.DEFAULT_ENVIRONMENT

        return environment

    @classmethod
    def get_enabled_models(cls) -> List[str]:
        """환경에 따른 활성화 모델 목록 반환"""
        environment = cls.get_environment()

        # 환경 변수로 직접 지정된 경우 우선 사용
        env_models = os.getenv("ENABLED_MODELS")
        if env_models:
            models = [model.strip() for model in env_models.split(",")]
            logger.info(f"환경 변수에서 활성화 모델 로드: {models}")
            return models

        # 환경별 기본 모델 사용
        default_models = cls.ENVIRONMENT_MODEL_MAP.get(
            environment, cls.ENVIRONMENT_MODEL_MAP[cls.DEFAULT_ENVIRONMENT]
        )
        logger.info(f"환경 '{environment}'에 대한 기본 모델: {default_models}")
        return default_models

    @classmethod
    def get_model_memory_limit(cls) -> str:
        """환경에 따른 메모리 제한 반환"""
        environment = cls.get_environment()

        # 환경 변수로 직접 지정된 경우 우선 사용
        env_memory = os.getenv("MEMORY_LIMIT")
        if env_memory:
            logger.info(f"환경 변수에서 메모리 제한 로드: {env_memory}")
            return env_memory

        # 환경별 기본 메모리 제한 사용
        default_memory = cls.ENVIRONMENT_MEMORY_MAP.get(
            environment, cls.DEFAULT_MEMORY_LIMIT
        )
        logger.info(f"환경 '{environment}'에 대한 기본 메모리 제한: {default_memory}")
        return default_memory

    @classmethod
    def is_gcp_environment(cls) -> bool:
        """GCP 환경인지 확인"""
        environment = cls.get_environment()
        return environment.startswith("gcp-")

    @classmethod
    def is_production_environment(cls) -> bool:
        """프로덕션 환경인지 확인"""
        environment = cls.get_environment()
        return environment in ["prod", "gcp-gke"]

    @classmethod
    def get_vllm_config(cls) -> dict:
        """환경별 vLLM 설정 반환"""
        environment = cls.get_environment()

        base_config = {
            "tensor_parallel_size": int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")),
            "trust_remote_code": os.getenv("VLLM_TRUST_REMOTE_CODE", "True").lower()
            == "true",
            "download_dir": os.getenv("VLLM_DOWNLOAD_DIR", "./model_cache"),
            "max_model_len": int(os.getenv("VLLM_MAX_MODEL_LEN", "2048")),
            "gpu_memory_utilization": float(
                os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.5")
            ),
        }

        # 환경별 최적화
        if cls.is_gcp_environment():
            base_config.update(
                {
                    "gpu_memory_utilization": 0.7,
                    "max_model_len": 2048,
                    "enforce_eager": True,  # GCP에서 메모리 효율성
                }
            )
        elif environment == "test":
            base_config.update(
                {
                    "gpu_memory_utilization": 0.3,
                    "max_model_len": 1024,
                }
            )

        return base_config

    @classmethod
    def get_transformers_config(cls) -> dict:
        """환경별 Transformers 설정 반환"""
        environment = cls.get_environment()

        base_config = {
            "torch_dtype": "float32",
            "low_cpu_mem_usage": True,
            "device_map": "auto",
        }

        # 환경별 최적화
        if cls.is_gcp_environment():
            base_config.update(
                {
                    "torch_dtype": "float16" if environment == "gcp-gke" else "float32",
                    "low_cpu_mem_usage": True,
                }
            )
        elif environment == "test":
            base_config.update(
                {
                    "torch_dtype": "float32",
                    "low_cpu_mem_usage": True,
                }
            )

        return base_config

    @classmethod
    def get_config_summary(cls) -> dict:
        """현재 설정 요약 반환"""
        return {
            "environment": cls.get_environment(),
            "enabled_models": cls.get_enabled_models(),
            "memory_limit": cls.get_model_memory_limit(),
            "is_gcp": cls.is_gcp_environment(),
            "is_production": cls.is_production_environment(),
            "vllm_config": cls.get_vllm_config(),
            "transformers_config": cls.get_transformers_config(),
        }
