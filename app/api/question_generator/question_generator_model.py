
import asyncio
import logging
import os
import time
import uuid
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from langfuse import Langfuse
from typing import Optional
from huggingface_hub import login
from openai import OpenAI
from app.api.question_generator.question_generator_config import (
    HF_TOKEN,
    MODEL_PATH,
    OPENAI_API_KEY,
    LANGFUSE_CONFIG,
    VLLM_CONFIG,
    SAMPLING_CONFIG,
    API_CONFIG,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 설정 로드
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        logger.info("Hugging Face Hub에 성공적으로 로그인했습니다.")
    except Exception as e:
        logger.warning(f"Hugging Face Hub 로그인 실패: {e}")

langfuse = Langfuse(**LANGFUSE_CONFIG) if all(LANGFUSE_CONFIG.values()) else None
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

llm: Optional[AsyncLLMEngine] = None


def initialize_llm():
    """LLM을 초기화하는 함수"""
    global llm
    if llm is not None:
        logger.info("AsyncLLMEngine이 이미 초기화되어 있습니다.")
        return llm

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    try:
        engine_args = AsyncEngineArgs(
            model=MODEL_PATH,
            tokenizer=MODEL_PATH,
            tensor_parallel_size=VLLM_CONFIG["tensor_parallel_size"],
            trust_remote_code=VLLM_CONFIG["trust_remote_code"],
            dtype=VLLM_CONFIG["dtype"],
            quantization=VLLM_CONFIG["quantization"],
            download_dir=VLLM_CONFIG["download_dir"],
            max_model_len=VLLM_CONFIG["max_model_len"],
            gpu_memory_utilization=VLLM_CONFIG["gpu_memory_utilization"],
            max_num_batched_tokens=VLLM_CONFIG["max_num_batched_tokens"],
            max_num_seqs=VLLM_CONFIG["max_num_seqs"],
            enforce_eager=VLLM_CONFIG["enforce_eager"],
            enable_chunked_prefill=True,
            disable_log_requests=True,
            kv_cache_dtype=VLLM_CONFIG["kv_cache_dtype"],
        )

        llm = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info(f"AsyncLLMEngine 모델 로드 성공: {MODEL_PATH}")
        return llm
    except Exception as e:
        logger.error(f"AsyncLLMEngine 모델 로드 실패: {e}")
        raise


def get_llm_engine():
    """현재 글로벌 LLM 엔진 반환"""
    global llm
    return llm


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
    # 글로벌 LLM 엔진 직접 사용
    global llm
    llm_engine = llm

    if llm_engine is None:
        logger.error("LLM이 초기화되지 않았습니다.")
        if try_fallback and OPENAI_API_KEY:
            logger.info("LLM이 초기화되지 않아 OpenAI API로 대체합니다.")
            return await call_openai_api(prompt, trace_id)
        raise RuntimeError(
            "LLM (AsyncLLMEngine)이 초기화되지 않았고 fallback이 비활성화되어 있습니다."
        )

    request_id = uuid.uuid4().hex
    generation = None

    async def _generate_with_async_engine():
        params = SamplingParams(
            temperature=SAMPLING_CONFIG["temperature"],
            top_p=SAMPLING_CONFIG["top_p"],
            top_k=SAMPLING_CONFIG["top_k"],
            repetition_penalty=SAMPLING_CONFIG["repetition_penalty"],
            max_tokens=SAMPLING_CONFIG["max_tokens"],
            stop=SAMPLING_CONFIG["stop_sequences"],
        )

        results_generator = llm_engine.generate(prompt, params, request_id)

        final_output_text = ""
        try:
            async for request_output in results_generator:
                if request_output.finished:
                    if (
                        request_output.outputs
                        and request_output.outputs[0].text is not None
                    ):
                        final_output_text = request_output.outputs[0].text
                    else:
                        logger.warning(
                            f"요청 {request_id}이 완료되었지만 텍스트 출력을 찾을 수 없거나 출력이 None입니다."
                        )
                    break
            return final_output_text
        except Exception as e:
            logger.error(f"요청 {request_id}에 대한 AsyncLLMEngine 생성 오류: {e}")
            if llm_engine:
                asyncio.create_task(llm_engine.abort(request_id))
            raise

    generation = None
    try:
        generation_kwargs = {
            "name": "local-llm",
            "model": MODEL_PATH,
            "input": prompt,
            "metadata": SAMPLING_CONFIG,
        }

        if trace_id:
            generation_kwargs["trace_id"] = trace_id

        if langfuse:
            generation = langfuse.generation(**generation_kwargs)

        start_time = asyncio.get_event_loop().time()
        timeout = float(os.getenv("LLM_TIMEOUT", "60.0"))

        result = await asyncio.wait_for(_generate_with_async_engine(), timeout=timeout)
        execution_time = asyncio.get_event_loop().time() - start_time

        if generation:
            generation.end(
                output=result,
                metadata={
                    "execution_time_seconds": execution_time,
                    **generation_kwargs["metadata"],
                },
            )
        return result

    except (asyncio.TimeoutError, Exception) as e:
        error_type = "TimeoutError" if isinstance(e, asyncio.TimeoutError) else "Error"
        error_message = (
            f"LLM ({MODEL_PATH}) 응답 시간 초과 ({timeout}초)"
            if isinstance(e, asyncio.TimeoutError)
            else f"LLM ({MODEL_PATH}) 호출 실패: {str(e)}"
        )

        logger.error(error_message)

        asyncio.create_task(llm_engine.abort(request_id))

        if generation:  # generation 객체가 존재하는지 확인
            generation.end(
                error=error_message, metadata=generation_kwargs.get("metadata", {})
            )

        if try_fallback and OPENAI_API_KEY:
            logger.info(
                f"로컬 LLM ({MODEL_PATH}) 실패/타임아웃, OpenAI API로 fallback합니다."
            )
            return await call_openai_api(prompt, trace_id)

        if isinstance(e, asyncio.TimeoutError):
            # 특정 TimeoutError로 다시 발생
            raise TimeoutError(error_message)
        else:
            raise Exception(error_message)  # 일반 예외 다시 발생


async def call_openai_api(prompt: str, trace_id: str = None) -> str:
    """
    OpenAI API를 사용하여 프롬프트에 대한 응답을 생성합니다.
    주로 부족한 질문을 채우기 위한 백업 메커니즘으로 사용됩니다.
    """
    if not openai_client:
        raise RuntimeError("OpenAI API 키가 설정되지 않았습니다.")        
    model_name = API_CONFIG["openai_model"]
    generation = None  # Langfuse용

    try:
        generation_kwargs = {
            "name": "openai-api-call",
            "model": model_name,
            "input": [
                {
                    "role": "system",
                    "content": "당신은 IT 기술 면접을 위한 질문을 생성하는 도우미입니다.",
                },
                {"role": "user", "content": prompt},
            ],
        }

        if trace_id:
            generation_kwargs["trace_id"] = trace_id

        if langfuse:
            generation = langfuse.generation(**generation_kwargs)

        def _blocking_openai_call():
            return openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 IT 기술 면접을 위한 질문을 생성하는 도우미입니다.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=500,  # 환경 변수를 통해 구성 가능하도록 고려
                top_p=0.95,
            )

        # 시간 추적 시작
        start_time = asyncio.get_event_loop().time()
        response = await asyncio.to_thread(_blocking_openai_call)
        execution_time = asyncio.get_event_loop().time() - start_time

        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        input_cache_tokens = getattr(response.usage, "input_cached_tokens", 0)
        input_cache_read_tokens = getattr(response.usage, "input_cache_read", 0)

        cost_per_1m_input = 0.15
        cost_per_1m_output = 0.60
        cost_per_1m_cache = 0.075

        cost = (
            (input_tokens / 1_000_000) * cost_per_1m_input
            + (output_tokens / 1_000_000) * cost_per_1m_output
            + (input_cache_read_tokens / 1_000_000) * cost_per_1m_cache
            + (input_cache_tokens / 1_000_000) * cost_per_1m_cache
        )

        logger.info(
            f"OpenAI API ({model_name}) 호출 성공: 입력 토큰 {input_tokens}, 출력 토큰 {output_tokens}, 비용 ${cost:.6f}, 실행 시간 {execution_time:.2f}초"
        )

        if generation:
            generation.end(
                output=response.choices[0].message.content,
                usage={
                    "promptTokens": input_tokens,
                    "completionTokens": output_tokens,
                    "totalTokens": input_tokens + output_tokens,
                },
                metadata={
                    "cost_usd": cost,
                    "model_name": model_name,
                    "execution_time_seconds": execution_time,
                },
            )
        return response.choices[0].message.content

    except Exception as e:
        error_message = f"OpenAI API ({model_name}) 호출 오류: {str(e)}"
        logger.error(error_message)
        if generation:
            generation.end(error=error_message)
        raise