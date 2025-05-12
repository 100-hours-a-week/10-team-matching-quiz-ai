import asyncio
import torch
import re
import logging
from vllm import LLM, SamplingParams
from langfuse import Langfuse
import os
from typing import Optional, Union, Dict, Any, List
from huggingface_hub import login
from openai import OpenAI
from dotenv import load_dotenv
# .env 파일 로드
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        logger.info("Successfully logged in to Hugging Face Hub")
    except Exception as e:
        logger.warning(f"Failed to login to Hugging Face Hub: {e}")

MODEL_PATH = os.getenv("LLM_MODEL_PATH")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

langfuse = Langfuse(
    secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
    public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
    host=os.getenv('LANGFUSE_HOST')
)

# 지연 초기화를 위한 전역 변수
llm = None


def str2bool(value: str) -> bool:
    return value.lower() in ("true", "1", "yes")


def initialize_llm():
    """LLM을 초기화하는 함수 - 처음에 실행할때만 호출"""
    global llm
    if llm is not None:
        return llm

    try:
        dtype_env = os.getenv("DTYPE", "auto")
        tensor_parallel_size_env = int(
            os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1"))
        trust_remote_code_env = str2bool(
            os.getenv("VLLM_TRUST_REMOTE_CODE", "True"))
        download_dir_env = os.getenv("VLLM_DOWNLOAD_DIR", "./model_cache")
        max_model_len_env = int(os.getenv("VLLM_MAX_MODEL_LEN", "2048"))
        gpu_memory_utilization_env = float(
            os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.9"))
        max_num_batched_tokens_env = int(
            os.getenv("VLLM_MAX_NUM_BATCHED_TOKENS", "4096"))
        max_num_seqs_env = int(os.getenv("VLLM_MAX_NUM_SEQS", "256"))
        enforce_eager_env = str2bool(os.getenv("VLLM_ENFORCE_EAGER", "False"))
        quantization_env = os.getenv(
            "VLLM_QUANTIZATION", None)  # 예: "awq" 또는 "gptq"
        # load_format_env = os.getenv("VLLM_LOAD_FORMAT", None)

        llm = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=tensor_parallel_size_env,
            trust_remote_code=trust_remote_code_env,
            dtype=dtype_env,
            quantization=quantization_env,  # 양자화 설정이 필요한 경우 주석 해제
            download_dir=download_dir_env,
            max_model_len=max_model_len_env,
            gpu_memory_utilization=gpu_memory_utilization_env,
            max_num_batched_tokens=max_num_batched_tokens_env,
            max_num_seqs=max_num_seqs_env,
            enforce_eager=enforce_eager_env
        )
        logger.info(
            f"Hugging Face 모델 로드 성공: {MODEL_PATH} with params: "
            f"dtype={dtype_env}, tensor_parallel_size={tensor_parallel_size_env}, "
            f"trust_remote_code={trust_remote_code_env}, download_dir='{download_dir_env}', "
            f"max_model_len={max_model_len_env}, gpu_memory_utilization={gpu_memory_utilization_env}, "
            f"max_num_batched_tokens={max_num_batched_tokens_env}, max_num_seqs={max_num_seqs_env}, "
            f"enforce_eager={enforce_eager_env}"
        )
        return llm
    except Exception as e:
        logger.error(f"Hugging Face 모델 로드 실패: {e}")
        raise


async def call_llm(prompt: str, try_fallback: bool = True, trace_id: str = None) -> str:
    """
    LLM에 프롬프트를 안전하게 전송하고 생성된 응답을 반환합니다.

    Parameters:
        prompt (str): LLM에 전달할 텍스트 프롬프트
        try_fallback (bool): 로컬 모델 실패 시 OpenAI 기반 fallback 사용 여부
        trace_id (str): 부모 트레이스 ID (있는 경우 사용)

    Returns:
        str: 모델이 생성한 응답 텍스트

    Raises:
        Exception: LLM 호출 과정에서 발생한 예외 (fallback이 false이거나 fallback도 실패한 경우)
    """
    global llm

    def _generate():
        temperature_env = float(os.getenv("VLLM_SAMPLING_TEMPERATURE", "0.7"))
        top_p_env = float(os.getenv("VLLM_SAMPLING_TOP_P", "0.9"))
        top_k_env = int(os.getenv("VLLM_SAMPLING_TOP_K", "50"))
        repetition_penalty_env = float(
            os.getenv("VLLM_SAMPLING_REPETITION_PENALTY", "1.15"))
        max_tokens_env = int(os.getenv("VLLM_SAMPLING_MAX_TOKENS", "150"))
        stop_sequences_str = os.getenv(
            "VLLM_SAMPLING_STOP_SEQUENCES", "질문 5.,질문 6.")
        stop_sequences_env = [seq.strip() for seq in stop_sequences_str.split(
            ',') if seq.strip()] if stop_sequences_str else []

        params = SamplingParams(
            temperature=temperature_env,
            top_p=top_p_env,
            top_k=top_k_env,
            repetition_penalty=repetition_penalty_env,
            max_tokens=max_tokens_env,
            stop=stop_sequences_env
        )
        try:
            outputs = llm.generate([prompt], params)
            return outputs[0].outputs[0].text
        except Exception as e:
            logger.error(f"LLM 생성 오류: {e}")
            raise

    try:
        generation_kwargs = {
            "name": "local-llm-call",
            "model": MODEL_PATH,
            "input": prompt,
            "metadata": {
                "temperature": float(os.getenv("VLLM_SAMPLING_TEMPERATURE", "0.7")),
                "top_p": float(os.getenv("VLLM_SAMPLING_TOP_P", "0.9")),
                "top_k": int(os.getenv("VLLM_SAMPLING_TOP_K", "50")),
                "repetition_penalty": float(os.getenv("VLLM_SAMPLING_REPETITION_PENALTY", "1.15")),
                "max_tokens": int(os.getenv("VLLM_SAMPLING_MAX_TOKENS", "150")),
            }
        }

        if trace_id:
            generation_kwargs["trace_id"] = trace_id

        generation = langfuse.generation(**generation_kwargs)

        start_time = asyncio.get_event_loop().time()
        timeout = float(os.getenv("LLM_TIMEOUT", "60.0"))
        result = await asyncio.wait_for(
            asyncio.to_thread(_generate),
            timeout=timeout
        )
        execution_time = asyncio.get_event_loop().time() - start_time

        generation.end(
            output=result,
            metadata={"execution_time_seconds": execution_time,
                      **generation_kwargs["metadata"]}
        )

        return result
    except (asyncio.TimeoutError, Exception) as e:
        error_type = "TimeoutError" if isinstance(
            e, asyncio.TimeoutError) else "Error"
        error_message = "LLM 응답 시간 초과" if isinstance(
            e, asyncio.TimeoutError) else f"LLM 호출 실패: {str(e)}"

        if 'generation' in locals() and generation:  # generation 객체가 생성되었는지 확인
            generation.end(
                error=error_message,
                metadata=generation_kwargs.get(
                    "metadata", {})  # 에러 시에도 메타데이터 기록
            )

        logger.error(error_message)

        if try_fallback and OPENAI_API_KEY:
            logger.info("로컬 LLM 실패, OpenAI API로 fallback")
            return await call_openai_api(prompt, trace_id)

        if isinstance(e, asyncio.TimeoutError):
            raise TimeoutError(error_message)
        else:
            raise Exception(error_message)


async def call_openai_api(prompt: str, trace_id: str = None) -> str:
    """
    OpenAI API를 사용하여 프롬프트에 대한 응답을 생성합니다.
    주로 부족한 질문을 채우기 위한 백업 메커니즘으로 사용됩니다.

    Parameters:
        prompt (str): OpenAI 모델에 전달할 텍스트 프롬프트
        trace_id (str): 부모 트레이스 ID (있는 경우 사용)

    Returns:
        str: 모델이 생성한 응답 텍스트

    Raises:
        Exception: API 호출 과정에서 발생한 예외
    """
    model_name = "gpt-4o-mini"

    try:
        generation_kwargs = {
            "name": "openai-api-call",
            "model": model_name,
            "input": [
                {"role": "system", "content": "당신은 IT 기술 면접을 위한 질문을 생성하는 도우미입니다."},
                {"role": "user", "content": prompt}
            ],
        }

        if trace_id:
            generation_kwargs["trace_id"] = trace_id

        generation = langfuse.generation(**generation_kwargs)

        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "당신은 IT 기술 면접을 위한 질문을 생성하는 도우미입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500,
            top_p=0.95
        )

        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost_per_1k_input = 0.00015
        cost_per_1k_output = 0.0006
        cost = (input_tokens / 1000 * cost_per_1k_input) + \
            (output_tokens / 1000 * cost_per_1k_output)

        logger.info(
            f"OpenAI API 호출 성공: 입력 토큰 {input_tokens}, 출력 토큰 {output_tokens}, 비용 ${cost:.5f}")

        generation.end(
            output=response.choices[0].message.content,
            usage_details={
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens
            },
            cost_details={
                "input": input_tokens / 1000 * cost_per_1k_input,
                "output": output_tokens / 1000 * cost_per_1k_output,
                "total": cost
            }
        )

        return response.choices[0].message.content
    except Exception as e:
        if 'generation' in locals() and generation:  # generation 객체가 생성되었는지 확인
            generation.end(error=str(e))
        logger.error(f"OpenAI API 호출 오류: {e}")
        raise
