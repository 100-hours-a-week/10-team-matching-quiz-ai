
import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

# Hugging Face 설정
HF_TOKEN = os.getenv("HF_TOKEN", "")

# 모델 설정
MODEL_PATH = os.getenv("LLM_MODEL_PATH", "TommyKong/gemma-3-finetune-4bit")

# vLLM API 설정
VLLM_API_CONFIG = {
    "base_url": os.getenv("VLLM_API_BASE", "http://127.0.0.1:8000/v1"),
    "api_key": os.getenv("VLLM_API_KEY", "dummy-key"),
    "model_name": os.getenv("VLLM_MODEL_NAME", "TommyKong/gemma-3-finetune-4bit"),  # 수정됨
    "timeout": float(os.getenv("LLM_TIMEOUT", "8.0")),
}

# OpenAI API 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Langfuse 설정
LANGFUSE_CONFIG = {
    "secret_key": os.getenv("LANGFUSE_SECRET_KEY"),
    "public_key": os.getenv("LANGFUSE_PUBLIC_KEY"),
    "host": os.getenv("LANGFUSE_HOST"),
}

# vLLM 서버 설정 (서버 시작용)
VLLM_SERVER_CONFIG = {
    "model_path": MODEL_PATH,
    "host": os.getenv("VLLM_HOST", "127.0.0.1"),
    "port": int(os.getenv("VLLM_PORT", "8000")),
    "dtype": os.getenv("DTYPE", "bfloat16"),
    "tensor_parallel_size": int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")),
    "trust_remote_code": os.getenv("VLLM_TRUST_REMOTE_CODE", "True").lower() == "true",
    "download_dir": os.getenv("VLLM_DOWNLOAD_DIR", "./model_cache"),
    "max_model_len": int(os.getenv("VLLM_MAX_MODEL_LEN", "1024")),
    "gpu_memory_utilization": float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.45")),
    "max_num_batched_tokens": int(os.getenv("VLLM_MAX_NUM_BATCHED_TOKENS", "2048")),
    "max_num_seqs": int(os.getenv("VLLM_MAX_NUM_SEQS", "16")),
    "enforce_eager": os.getenv("VLLM_ENFORCE_EAGER", "False").lower() == "true",
    "quantization": os.getenv("VLLM_QUANTIZATION", "bitsandbytes"),
    "enable_chunked_prefill": os.getenv("VLLM_ENABLE_CHUNKED_PREFILL", "True").lower() == "true",
    "enable_prefix_caching": os.getenv("VLLM_ENABLE_PREFIX_CACHING", "True").lower() == "true",
    "max_num_seqs_per_batch": int(os.getenv("VLLM_MAX_NUM_SEQS_PER_BATCH", "16")),
    "block_size": int(os.getenv("VLLM_BLOCK_SIZE", "16")),
    "swap_space": int(os.getenv("VLLM_SWAP_SPACE", "4")),
    "scheduler_delay_factor": float(os.getenv("VLLM_SCHEDULER_DELAY_FACTOR", "0.0")),
    "worker_multiproc_method": os.getenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn"),
}

def _parse_stop_sequences() -> List[str]:
    """정지 시퀀스를 파싱하여 리스트로 반환"""
    stop_sequences_str = os.getenv("VLLM_SAMPLING_STOP_SEQUENCES", "질문 5.,질문5.,Question 5.,Q5.,Q 5.")
    if not stop_sequences_str:
        return []
    return [seq.strip() for seq in stop_sequences_str.split(",") if seq.strip()]

SAMPLING_CONFIG = {
    "temperature": float(os.getenv("VLLM_SAMPLING_TEMPERATURE", "0.8")),
    "top_p": float(os.getenv("VLLM_SAMPLING_TOP_P", "0.95")),
    "top_k": int(os.getenv("VLLM_SAMPLING_TOP_K", "20")),
    "repetition_penalty": float(os.getenv("VLLM_SAMPLING_REPETITION_PENALTY", "1.15")),
    "max_tokens": int(os.getenv("VLLM_SAMPLING_MAX_TOKENS", "512")),
    "stop_sequences": _parse_stop_sequences(),  # 수정됨
}

# API 설정
API_CONFIG = {
    "generate_count": int(os.getenv("GENERATE_COUNT", "4")),
    "max_history_questions": int(os.getenv("MAX_HISTORY_QUESTIONS", "20")),
    "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
}

# 파서
PARSER_CONFIG = {
    "max_question_length": int(os.getenv('MAX_QUESTION_LENGTH', "100"))
}