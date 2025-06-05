import os
from typing import List
from dotenv import load_dotenv

load_dotenv()


class QuestionGeneratorConfig:
    """질문 생성기 설정 관리 클래스"""

    @staticmethod
    def get_hf_token() -> str:
        """Hugging Face 토큰 반환"""
        return os.getenv("HF_TOKEN", "")

    @staticmethod
    def get_model_path() -> str:
        """모델 경로 반환"""
        return os.getenv("LLM_MODEL_PATH", "TommyKong/gemma-3-finetune-4bit")

    @staticmethod
    def get_openai_api_key() -> str:
        """OpenAI API 키 반환"""
        return os.getenv("OPENAI_API_KEY", "")

    @staticmethod
    def get_langfuse_config() -> dict:
        """Langfuse 설정 반환"""
        return {
            "secret_key": os.getenv("LANGFUSE_SECRET_KEY"),
            "public_key": os.getenv("LANGFUSE_PUBLIC_KEY"),
            "host": os.getenv("LANGFUSE_HOST"),
        }

    @staticmethod
    def get_vllm_config() -> dict:
        """vLLM 설정 반환"""
        return {
            "dtype": os.getenv("DTYPE", "auto"),
            "tensor_parallel_size": int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")),
            "trust_remote_code": os.getenv("VLLM_TRUST_REMOTE_CODE", "True").lower() == "true",
            "download_dir": os.getenv("VLLM_DOWNLOAD_DIR", "./model_cache"),
            "max_model_len": int(os.getenv("VLLM_MAX_MODEL_LEN", "2048")),
            "gpu_memory_utilization": float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.9")),
            "max_num_batched_tokens": int(os.getenv("VLLM_MAX_NUM_BATCHED_TOKENS", "4096")),
            "max_num_seqs": int(os.getenv("VLLM_MAX_NUM_SEQS", "256")),
            "enforce_eager": os.getenv("VLLM_ENFORCE_EAGER", "False").lower() == "true",
            "quantization": os.getenv("VLLM_QUANTIZATION"),
            "kv_cache_dtype": os.getenv("VLLM_KV_CACHE_DTYPE", "auto"),
        }

    @staticmethod
    def get_sampling_config() -> dict:
        """샘플링 설정 반환"""
        return {
            "temperature": float(os.getenv("VLLM_SAMPLING_TEMPERATURE", "0.7")),
            "top_p": float(os.getenv("VLLM_SAMPLING_TOP_P", "0.9")),
            "top_k": int(os.getenv("VLLM_SAMPLING_TOP_K", "50")),
            "repetition_penalty": float(os.getenv("VLLM_SAMPLING_REPETITION_PENALTY", "1.15")),
            "max_tokens": int(os.getenv("VLLM_SAMPLING_MAX_TOKENS", "150")),
            "stop_sequences": QuestionGeneratorConfig._parse_stop_sequences(),
        }

    @staticmethod
    def _parse_stop_sequences() -> List[str]:
        """정지 시퀀스를 파싱하여 리스트로 반환"""
        stop_sequences_str = os.getenv("VLLM_SAMPLING_STOP_SEQUENCES", "질문 5.,질문 6.")
        if not stop_sequences_str:
            return []
        return [seq.strip() for seq in stop_sequences_str.split(",") if seq.strip()]

    @staticmethod
    def get_api_config() -> dict:
        """API 설정 반환"""
        return {
            "generate_count": int(os.getenv("GENERATE_COUNT", "4")),
            "max_history_questions": int(os.getenv("MAX_HISTORY_QUESTIONS", "20")),
            "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        }

    @staticmethod
    def get_parser_config() -> dict:
        """파서 설정 반환"""
        return {
            "max_question_length": int(os.getenv("MAX_QUESTION_LENGTH", "100")),
        }
