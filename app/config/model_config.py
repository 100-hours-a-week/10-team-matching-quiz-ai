import os
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

def _detect_environment() -> str:
    """현재 환경 감지"""
    env = os.getenv("ENVIRONMENT", "local")

    if os.getenv("GCP_PROJECT"):
        return "gcp-vm"
    else:
        return env

def _parse_enabled_models() -> List[str]:
    """활성화된 모델 목록 파싱"""
    enabled = os.getenv("ENABLED_MODELS", "question_generator")
    return [model.strip() for model in enabled.split(",")]

ENVIRONMENT = _detect_environment()

ENABLED_MODELS = _parse_enabled_models()

VLLM_CONFIG = {
    "model_path": os.getenv(
        "LLM_MODEL_PATH", "TommyKong/gemma-3-finetune-4bit"
    ),
    "max_model_len": int(os.getenv("VLLM_MAX_MODEL_LEN", "2048")),
    "gpu_memory_utilization": float(
        os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.5")
    ),
    "trust_remote_code": True,
}

TRANSFORMERS_CONFIG = {
    "model_name" : os.getenv(
        "QUIZ_MODEL_NAME", "unsloth/Qwen3-8B-unsloth-bnb-4bit"),
    "hf_token" : os.getenv("QUIZ_HF_TOKEN"),
            "trust_remote_code": True,    
}


