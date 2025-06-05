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
        elif os.getenv("COLAB_GPU"):
            return "colab"
        else:
            return env

    @staticmethod
    def get_enabled_models() -> List[str]:
        """활성화된 모델 목록 반환"""
        enabled = os.getenv("ENABLED_MODELS", "question_generator,quiz_generator")
        return [model.strip() for model in enabled.split(",")]

    @staticmethod
    def get_model_memory_limit() -> str:
        """모델 메모리 제한 설정"""
        return os.getenv("MODEL_MEMORY_LIMIT", "auto")

    @staticmethod
    def get_vllm_config() -> Dict[str, Any]:
        """vLLM 설정 반환"""
        return {
            "model_path": os.getenv(
                "LLM_MODEL_PATH", "TommyKong/gemma-3-finetune-4bit"
            ),
            "tensor_parallel_size": int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")),
            "trust_remote_code": os.getenv("VLLM_TRUST_REMOTE_CODE", "True").lower()
            == "true",
            "download_dir": os.getenv("VLLM_DOWNLOAD_DIR", "./model_cache"),
            "max_model_len": int(os.getenv("VLLM_MAX_MODEL_LEN", "2048")),
            "gpu_memory_utilization": float(
                os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.5")
            ),
            "max_num_batched_tokens": int(
                os.getenv("VLLM_MAX_NUM_BATCHED_TOKENS", "4096")
            ),
            "max_num_seqs": int(os.getenv("VLLM_MAX_NUM_SEQS", "16")),
            "enforce_eager": os.getenv("VLLM_ENFORCE_EAGER", "False").lower() == "true",
            "quantization": os.getenv("VLLM_QUANTIZATION", "bitsandbytes"),
            "enable_chunked_prefill": os.getenv(
                "VLLM_ENABLE_CHUNKED_PREFILL", "True"
            ).lower()
            == "true",
            "max_num_seqs_per_batch": int(
                os.getenv("VLLM_MAX_NUM_SEQS_PER_BATCH", "16")
            ),
            "block_size": int(os.getenv("VLLM_BLOCK_SIZE", "16")),
            "swap_space": int(os.getenv("VLLM_SWAP_SPACE", "4")),
            "scheduler_delay_factor": float(
                os.getenv("VLLM_SCHEDULER_DELAY_FACTOR", "0.0")
            ),
            "enable_prefix_caching": os.getenv(
                "VLLM_ENABLE_PREFIX_CACHING", "True"
            ).lower()
            == "true",
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

    @staticmethod
    def is_development() -> bool:
        """개발 환경 여부"""
        return ModelConfig.get_environment() in ["local", "development"]

    @staticmethod
    def is_production() -> bool:
        """프로덕션 환경 여부"""
        return ModelConfig.get_environment() in ["gcp-gke", "kubernetes", "production"]
