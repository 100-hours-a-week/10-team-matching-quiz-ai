import asyncio
import logging
import os
import time
import uuid
from langfuse import Langfuse
from typing import Optional
from huggingface_hub import login
from openai import AsyncOpenAI  # OpenAI 제거
from app.api.question_generator.question_generator_config import (
    HF_TOKEN,
    MODEL_PATH,
    OPENAI_API_KEY,
    LANGFUSE_CONFIG,
    VLLM_API_CONFIG,
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
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

vllm_client: Optional[AsyncOpenAI] = None


def initialize_llm():
    """vLLM API를 초기화하는 함수"""
    global vllm_client
    
    if vllm_client is not None:
        logger.info('vLLM API 클라이언트가 이미 초기화되어 있습니다.')
        return True 
    
    try:
        vllm_client = AsyncOpenAI(
            base_url=VLLM_API_CONFIG['base_url'],
            api_key=VLLM_API_CONFIG['api_key'],
            timeout=VLLM_API_CONFIG['timeout']
        )
        logger.info(f'vLLM API 클라이언트 초기화 성공: {VLLM_API_CONFIG["base_url"]}')
        return True
    except Exception as e:
        logger.error(f'vLLM API 클라이언트 초기화 실패: {e}')
        vllm_client = None
        return False 


def get_llm_engine():
    """현재 vLLM API 클라이언트 반환"""
    global vllm_client
    return vllm_client


async def call_llm(prompt: str, try_fallback: bool = True, trace_id: str = None) -> str:
    global vllm_client

    if vllm_client is None:
        if not initialize_llm():
            if try_fallback:
                logger.warning("vLLM API 초기화 실패, OpenAI API로 폴백")
                return await call_openai_api(prompt, trace_id)
            raise Exception("vLLM API 클라이언트가 초기화되지 않았습니다.")

    try:
        # 기본 파라미터 준비
        request_params = {
            "model": VLLM_API_CONFIG["model_name"],
            "prompt": prompt,
            "max_tokens": SAMPLING_CONFIG["max_tokens"],
            "temperature": SAMPLING_CONFIG["temperature"],
            "top_p": SAMPLING_CONFIG["top_p"],
        }
        
        # stop sequences 안전하게 처리
        stop_seqs = SAMPLING_CONFIG.get("stop_sequences", [])
        if stop_seqs and len(stop_seqs) > 0:
            # 빈 문자열 제거
            clean_stop_seqs = [seq for seq in stop_seqs if seq and seq.strip()]
            if clean_stop_seqs:
                request_params["stop"] = clean_stop_seqs
        
        # 요청 파라미터 로깅
        # logger.info(f"=== vLLM API 요청 시작 ===")
        # logger.info(f"모델: {request_params['model']}")
        # logger.info(f"프롬프트 길이: {len(prompt)}")
        # logger.info(f"Max tokens: {request_params['max_tokens']}")
        # logger.info(f"Temperature: {request_params['temperature']}")
        # logger.info(f"Top P: {request_params['top_p']}")
        # logger.info(f"Stop sequences: {request_params.get('stop', '없음')}")
        
        response = await vllm_client.completions.create(**request_params)
        
        content = response.choices[0].text.strip()
        # logger.info(f"vLLM API 호출 성공: 응답 길이={len(content)}")
        return content

    except Exception as e:
        logger.error(f"=== vLLM API 호출 실패 ===")
        logger.error(f"오류 타입: {type(e).__name__}")
        logger.error(f"오류 메시지: {str(e)}")
        
        # HTTP 오류 상세 정보
        if hasattr(e, 'response') and e.response:
            logger.error(f"HTTP 상태: {e.response.status_code}")
            try:
                logger.error(f"응답 내용: {e.response.text}")
            except:
                logger.error("응답 내용을 읽을 수 없음")
        
        if try_fallback:
            logger.warning("OpenAI API로 폴백합니다.")
            return await call_openai_api(prompt, trace_id)
        
        raise Exception(f"vLLM API 호출 실패: {e}")


async def call_openai_api(prompt: str, trace_id: str = None) -> str:
    """
    OpenAI API (gpt-4o-mini)를 사용하여 프롬프트에 대한 응답을 생성합니다.
    """
    if not openai_client:
        raise RuntimeError("OpenAI API 키가 설정되지 않았습니다.")        
    
    model_name = API_CONFIG["openai_model"]
    generation = None

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

        start_time = time.time()
        
        # AsyncOpenAI를 직접 사용 (asyncio.to_thread 불필요)
        response = await openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "당신은 IT 기술 면접을 위한 질문을 생성하는 도우미입니다.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=512,
        )
        
        execution_time = time.time() - start_time

        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        input_cache_tokens = getattr(response.usage, "input_cached_tokens", 0)
        input_cache_read_tokens = getattr(response.usage, "input_cache_read", 0)

        # gpt-4o-mini 가격 (2024년 기준)
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
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": input_tokens + output_tokens,
                    "cost": cost,
                },
                metadata={"execution_time_seconds": execution_time},
            )

        return response.choices[0].message.content

    except Exception as e:
        error_message = f"OpenAI API ({model_name}) 호출 오류: {str(e)}"
        logger.error(error_message)
        if generation:
            generation.end(error=error_message)
        raise
    

async def check_vllm_health() -> bool:
    """vLLM API 서버 상태 확인"""
    global vllm_client
    
    if vllm_client is None:
        return False
    
    try:
        # 간단한 테스트 요청
        response = await vllm_client.completions.create(
            model=VLLM_API_CONFIG["model_name"],
            prompt="Hello",
            max_tokens=1,
            temperature=0.1
        )
        return True
    except Exception as e:
        logger.error(f"vLLM API 상태 확인 실패: {e}")
        return False