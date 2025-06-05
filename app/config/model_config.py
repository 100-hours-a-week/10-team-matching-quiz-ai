import os
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()


class ModelConfig:
    """모델 설정 관리 클래스"""

    @staticmethod
    def get_environment() -> str:
        """현재 환경 감지"""
        env = os.getenv("ENVIRONMENT", "local")

        # 자동 환경 감지
        if os.getenv("GCP_PROJECT"):
            return "gcp-gke"
        elif os.getenv("KUBERNETES_SERVICE_HOST"):
            return "kubernetes"
        else:
            return env

    @staticmethod
    def get_enabled_models() -> List[str]:
        """활성화된 모델 목록 반환"""
        enabled = os.getenv("ENABLED_MODELS", "question_generator,quiz_generator")
        return [model.strip() for model in enabled.split(",")]

    @staticmethod
    def get_vllm_config() -> Dict[str, Any]:
        """vLLM 설정 반환"""
        return {
            "model_path": os.getenv(
                "LLM_MODEL_PATH", "TommyKong/gemma-3-finetune-4bit"
            ),
            "max_model_len": int(os.getenv("VLLM_MAX_MODEL_LEN", "2048")),
            "gpu_memory_utilization": float(
                os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.5")
            ),
            "trust_remote_code": True,
        }

    @staticmethod
    def get_transformers_config() -> Dict[str, Any]:
        """Transformers 설정 반환"""
        return {
            "model_name": os.getenv(
                "QUIZ_MODEL_NAME", "TommyKong/gemma-3-finetune-4bit"
            ),
            "hf_token": os.getenv("QUIZ_HF_TOKEN"),
            "device_map": "auto",
            "trust_remote_code": True,
        }