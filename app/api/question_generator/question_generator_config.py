
import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

# Hugging Face 설정
HF_TOKEN = os.getenv("HF_TOKEN", "")

# 모델 설정
MODEL_PATH = os.getenv("LLM_MODEL_PATH", "TommyKong/gemma-3-finetune-4bit")

# OpenAI API 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Langfuse 설정
LANGFUSE_CONFIG = {
    "secret_key": os.getenv("LANGFUSE_SECRET_KEY"),
    "public_key": os.getenv("LANGFUSE_PUBLIC_KEY"),
    "host": os.getenv("LANGFUSE_HOST"),
}

# vLLM OpenAI 호환 API 서버 설정
VLLM_API_CONFIG = {
    "base_url": os.getenv("VLLM_API_BASE_URL", "http://localhost:8001/v1"),
    "api_key": os.getenv("VLLM_API_KEY", "EMPTY"),  # vLLM 서버에서는 보통 "EMPTY" 사용
    "model_name": os.getenv("VLLM_MODEL_NAME", "TommyKong/gemma-3-finetune-4bit"),
    "timeout": float(os.getenv("VLLM_API_TIMEOUT", "60.0")),
}

# 기존 vLLM 직접 설정은 주석 처리 (더 이상 사용하지 않음)
# VLLM_CONFIG = {
#     "dtype": os.getenv("DTYPE", "auto"),
#     "tensor_parallel_size": int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")),
#     "trust_remote_code": os.getenv("VLLM_TRUST_REMOTE_CODE", "True").lower() == "true",
#     "download_dir": os.getenv("VLLM_DOWNLOAD_DIR", "./model_cache"),
#     "max_model_len": int(os.getenv("VLLM_MAX_MODEL_LEN", "2048")),
#     "gpu_memory_utilization": float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.9")),
#     "max_num_batched_tokens": int(os.getenv("VLLM_MAX_NUM_BATCHED_TOKENS", "4096")),
#     "max_num_seqs": int(os.getenv("VLLM_MAX_NUM_SEQS", "256")),
#     "enforce_eager": os.getenv("VLLM_ENFORCE_EAGER", "False").lower() == "true",
#     "quantization": os.getenv("VLLM_QUANTIZATION"),
#     "kv_cache_dtype": os.getenv("VLLM_KV_CACHE_DTYPE", "auto"),
# }

# 정지 시퀀스 파싱
def _parse_stop_sequences() -> List[str]:
    """정지 시퀀스를 파싱하여 리스트로 반환"""
    stop_sequences_str = os.getenv("VLLM_SAMPLING_STOP_SEQUENCES", "질문 5.,질문 6.")
    if not stop_sequences_str:
        return []
    return [seq.strip() for seq in stop_sequences_str.split(",") if seq.strip()]

# vLLM API 샘플링 설정 (OpenAI 호환 형식)
VLLM_SAMPLING_CONFIG = {
    "temperature": float(os.getenv("VLLM_SAMPLING_TEMPERATURE", "0.7")),
    "top_p": float(os.getenv("VLLM_SAMPLING_TOP_P", "0.9")),
    "max_tokens": int(os.getenv("VLLM_SAMPLING_MAX_TOKENS", "150")),
    "stop": _parse_stop_sequences(),
}

# 기존 샘플링 설정은 주석 처리 (vLLM 직접 사용 시에만 필요)
# SAMPLING_CONFIG = {
#     "temperature": float(os.getenv("VLLM_SAMPLING_TEMPERATURE", "0.7")),
#     "top_p": float(os.getenv("VLLM_SAMPLING_TOP_P", "0.9")),
#     "top_k": int(os.getenv("VLLM_SAMPLING_TOP_K", "50")),
#     "repetition_penalty": float(os.getenv("VLLM_SAMPLING_REPETITION_PENALTY", "1.15")),
#     "max_tokens": int(os.getenv("VLLM_SAMPLING_MAX_TOKENS", "150")),
#     "stop_sequences": _parse_stop_sequences(),
# }

# API 설정
API_CONFIG = {
    "generate_count": int(os.getenv("GENERATE_COUNT", "4")),
    "max_history_questions": int(os.getenv("MAX_HISTORY_QUESTIONS", "20")),
    "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
}

# 파서 설정
PARSER_CONFIG = {
    "max_question_length": int(os.getenv("MAX_QUESTION_LENGTH", "100")),
}
