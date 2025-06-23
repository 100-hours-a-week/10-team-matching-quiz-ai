import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

# Hugging Face 설정
HF_TOKEN = os.getenv("HF_TOKEN", "")

# vLLM API 설정
VLLM_API_CONFIG = {
    "base_url": os.getenv("VLLM_API_BASE", "http://127.0.0.1:8000/v1"),
    "api_key": os.getenv("VLLM_API_KEY", "dummy-key"),
    "model_name": os.getenv("VLLM_MODEL_NAME", "TommyKong/gemma-3-finetune-4bit"),
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

def _parse_stop_sequences() -> List[str]:
    """정지 시퀀스를 파싱하여 리스트로 반환"""
    stop_sequences_str = os.getenv("VLLM_SAMPLING_STOP_SEQUENCES", "질문 5.,질문5.,Question 5.,Q5.,Q 5.")
    if not stop_sequences_str:
        return []
    return [seq.strip() for seq in stop_sequences_str.split(",") if seq.strip()]

# 샘플링 설정
SAMPLING_CONFIG = {
    "temperature": float(os.getenv("VLLM_SAMPLING_TEMPERATURE", "0.8")),
    "top_p": float(os.getenv("VLLM_SAMPLING_TOP_P", "0.95")),
    "max_tokens": int(os.getenv("VLLM_SAMPLING_MAX_TOKENS", "300")),  # 컨텍스트 길이 문제 해결
    "stop_sequences": _parse_stop_sequences(),
}

# API 설정
API_CONFIG = {
    "generate_count": int(os.getenv("GENERATE_COUNT", "4")),
    "max_history_questions": int(os.getenv("MAX_HISTORY_QUESTIONS", "20")),
    "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
}

# 파서 설정
PARSER_CONFIG = {
    "max_question_length": int(os.getenv('MAX_QUESTION_LENGTH', "100"))
}