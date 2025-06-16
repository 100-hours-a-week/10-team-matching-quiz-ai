import asyncio
import logging
from typing import Optional
from openai import AsyncOpenAI
from app.api.question_generator.question_generator_config import (
    VLLM_API_CONFIG,
    VLLM_SAMPLING_CONFIG,
    OPENAI_API_KEY,
    API_CONFIG,
    LANGFUSE_CONFIG,
)

logger = logging.getLogger(__name__)

# vLLM OpenAI 호환 API 클라이언트
vllm_client = AsyncOpenAI(
    base_url=VLLM_API_CONFIG["base_url"],
    api_key=VLLM_API_CONFIG["api_key"],
    timeout=VLLM_API_CONFIG["timeout"],
)

# OpenAI API 클라이언트 (폴백용)
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Langfuse 초기화
langfuse = None
try:
    if all(LANGFUSE_CONFIG.values()):
        from langfuse import Langfuse
        langfuse = Langfuse(**LANGFUSE_CONFIG)
        logger.info("Langfuse가 성공적으로 초기화되었습니다.")
except ImportError:
    logger.warning("Langfuse 패키지를 찾을 수 없습니다.")
except Exception as e:
    logger.error(f"Langfuse 초기화 실패: {e}")


async def call_vllm_api(prompt: str, trace_id: Optional[str] = None) -> str:
    """
    vLLM OpenAI 호환 API를 사용하여 질문 생성
    
    Args:
        prompt: 생성할 프롬프트
        trace_id: Langfuse 트레이스 ID (선택적)
    
    Returns:
        생성된 텍스트
    
    Raises:
        Exception: API 호출 실패 시
    """
    generation = None
    try:
        generation_kwargs = {
            "name": "vllm-openai-api",
            "model": VLLM_API_CONFIG["model_name"],
            "input": prompt,
            "metadata": {
                "api_base": VLLM_API_CONFIG["base_url"],
                **VLLM_SAMPLING_CONFIG
            },
        }

        if trace_id:
            generation_kwargs["trace_id"] = trace_id

        if langfuse:
            generation = langfuse.generation(**generation_kwargs)

        start_time = asyncio.get_event_loop().time()

        # vLLM API 호출 (OpenAI 호환 형식)
        response = await vllm_client.chat.completions.create(
            model=VLLM_API_CONFIG["model_name"],
            messages=[
                {
                    "role": "system",
                    "content": "당신은 IT 기술 면접을 위한 꼬리질문을 생성하는 전문가입니다."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=VLLM_SAMPLING_CONFIG["temperature"],
            top_p=VLLM_SAMPLING_CONFIG["top_p"],
            max_tokens=VLLM_SAMPLING_CONFIG["max_tokens"],
            stop=VLLM_SAMPLING_CONFIG["stop"],
        )

        execution_time = asyncio.get_event_loop().time() - start_time
        result = response.choices[0].message.content

        # 사용량 및 비용 정보 추출 (가능한 경우)
        usage_info = {}
        if hasattr(response, 'usage') and response.usage:
            usage_info = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        logger.info(
            f"vLLM API 호출 성공: 모델={VLLM_API_CONFIG['model_name']}, "
            f"실행시간={execution_time:.2f}초, 토큰={usage_info.get('total_tokens', 'N/A')}"
        )

        if generation:
            generation.end(
                output=result,
                usage=usage_info,
                metadata={
                    "execution_time_seconds": execution_time,
                    "model_name": VLLM_API_CONFIG["model_name"],
                    "api_base": VLLM_API_CONFIG["base_url"],
                }
            )

        return result

    except Exception as e:
        error_message = f"vLLM API 호출 실패: {str(e)}"
        logger.error(error_message)
        
        if generation:
            generation.end(error=error_message)
        
        raise Exception(error_message)


async def call_openai_api(prompt: str, trace_id: Optional[str] = None) -> str:
    """
    OpenAI API를 사용하여 질문 생성 (폴백용)
    
    Args:
        prompt: 생성할 프롬프트
        trace_id: Langfuse 트레이스 ID (선택적)
    
    Returns:
        생성된 텍스트
    
    Raises:
        Exception: API 호출 실패 시
    """
    if not openai_client:
        raise RuntimeError("OpenAI API 키가 설정되지 않았습니다.")

    model_name = API_CONFIG["openai_model"]
    generation = None

    try:
        generation_kwargs = {
            "name": "openai-api-fallback",
            "model": model_name,
            "input": [
                {
                    "role": "system",
                    "content": "당신은 IT 기술 면접을 위한 꼬리질문을 생성하는 전문가입니다."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        }

        if trace_id:
            generation_kwargs["trace_id"] = trace_id

        if langfuse:
            generation = langfuse.generation(**generation_kwargs)

        start_time = asyncio.get_event_loop().time()

        response = await openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "당신은 IT 기술 면접을 위한 꼬리질문을 생성하는 전문가입니다."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=500,
            top_p=0.95,
        )

        execution_time = asyncio.get_event_loop().time() - start_time
        result = response.choices[0].message.content

        # 사용량 및 비용 계산
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        
        # GPT-4o-mini 기준 비용 (2024년 기준)
        cost_per_1m_input = 0.15
        cost_per_1m_output = 0.60
        
        cost = (
            (input_tokens / 1_000_000) * cost_per_1m_input +
            (output_tokens / 1_000_000) * cost_per_1m_output
        )

        logger.info(
            f"OpenAI API 호출 성공: 모델={model_name}, "
            f"입력토큰={input_tokens}, 출력토큰={output_tokens}, "
            f"비용=${cost:.6f}, 실행시간={execution_time:.2f}초"
        )

        if generation:
            generation.end(
                output=result,
                usage={
                    "promptTokens": input_tokens,
                    "completionTokens": output_tokens,
                    "totalTokens": input_tokens + output_tokens,
                },
                metadata={
                    "cost_usd": cost,
                    "model_name": model_name,
                    "execution_time_seconds": execution_time,
                }
            )

        return result

    except Exception as e:
        error_message = f"OpenAI API 호출 실패: {str(e)}"
        logger.error(error_message)
        
        if generation:
            generation.end(error=error_message)
        
        raise Exception(error_message)


async def call_llm(prompt: str, try_fallback: bool = True, trace_id: Optional[str] = None) -> str:
    """
    LLM 호출 통합 함수
    
    Args:
        prompt: 생성할 프롬프트
        try_fallback: vLLM 실패 시 OpenAI 폴백 사용 여부
        trace_id: Langfuse 트레이스 ID (선택적)
    
    Returns:
        생성된 텍스트
    
    Raises:
        Exception: 모든 API 호출 실패 시
    """
    try:
        # 먼저 vLLM API 시도
        return await call_vllm_api(prompt, trace_id)
    
    except Exception as e:
        logger.warning(f"vLLM API 호출 실패: {e}")
        
        if try_fallback and openai_client:
            logger.info("OpenAI API로 폴백합니다.")
            return await call_openai_api(prompt, trace_id)
        else:
            if not try_fallback:
                logger.error("폴백이 비활성화되어 있어 vLLM 실패 시 중단합니다.")
            else:
                logger.error("OpenAI API가 설정되지 않아 폴백할 수 없습니다.")
            raise


async def check_vllm_api_health() -> bool:
    """
    vLLM API 서버 상태 확인
    
    Returns:
        서버가 정상인지 여부
    """
    try:
        # 간단한 테스트 요청
        response = await vllm_client.chat.completions.create(
            model=VLLM_API_CONFIG["model_name"],
            messages=[{"role": "user", "content": "테스트"}],
            max_tokens=1,
            temperature=0.1,
        )
        
        logger.info(f"vLLM API 서버 상태 확인 성공: {VLLM_API_CONFIG['base_url']}")
        return True
        
    except Exception as e:
        logger.error(f"vLLM API 서버 상태 확인 실패: {e}")
        return False
