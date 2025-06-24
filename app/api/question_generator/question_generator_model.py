import logging
import time
from typing import Optional
from huggingface_hub import login
from openai import AsyncOpenAI
from langfuse import Langfuse

from app.api.question_generator.question_generator_config import (
    HF_TOKEN,
    OPENAI_API_KEY,
    LANGFUSE_CONFIG,
    VLLM_API_CONFIG,
    SAMPLING_CONFIG,
    API_CONFIG,
)

logger = logging.getLogger(__name__)

# Hugging Face 로그인
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        logger.info("Hugging Face Hub 로그인 성공")
    except Exception as e:
        logger.warning(f"Hugging Face Hub 로그인 실패: {e}")

# 클라이언트 초기화
langfuse = Langfuse(**LANGFUSE_CONFIG) if all(LANGFUSE_CONFIG.values()) else None
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
vllm_client: Optional[AsyncOpenAI] = None


def initialize_llm() -> bool:
    """vLLM API 클라이언트 초기화"""
    global vllm_client
    
    if vllm_client is not None:
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
    return vllm_client


async def call_llm(prompt: str, try_fallback: bool = True, trace_id: str = None) -> str:
    """vLLM API로 텍스트 생성"""
    global vllm_client

    # 클라이언트 초기화 확인
    if vllm_client is None and not initialize_llm():
        if try_fallback:
            logger.warning("vLLM API 초기화 실패, OpenAI API로 폴백")
            return await call_openai_api(prompt, trace_id)
        raise Exception("vLLM API 클라이언트 초기화 실패")

    # Generation 추적 설정
    generation = None
    if langfuse and trace_id:
        generation = langfuse.generation(
            trace_id=trace_id,
            name="vllm_generation",
            model=VLLM_API_CONFIG["model_name"],
            input=prompt,
            metadata={"api_type": "vllm"},
        )

    try:
        # 요청 파라미터 준비
        request_params = {
            "model": VLLM_API_CONFIG["model_name"],
            "prompt": prompt,
            "max_tokens": SAMPLING_CONFIG["max_tokens"],
            "temperature": SAMPLING_CONFIG["temperature"],
            "top_p": SAMPLING_CONFIG["top_p"],
        }
        
        # Stop sequences 처리
        stop_seqs = SAMPLING_CONFIG.get("stop_sequences", [])
        if stop_seqs:
            clean_stop_seqs = [seq.strip() for seq in stop_seqs if seq and seq.strip()]
            if clean_stop_seqs:
                request_params["stop"] = clean_stop_seqs
        
        # API 호출
        response = await vllm_client.completions.create(**request_params)
        content = response.choices[0].text.strip()
        
        # Generation 완료
        if generation:
            generation.end(output=content)
        
        return content

    except Exception as e:
        logger.error(f"vLLM API 호출 실패: {type(e).__name__} - {str(e)}")
        
        # Generation 오류 처리
        if generation:
            generation.end(error=str(e))
        
        # 폴백 처리
        if try_fallback:
            logger.warning("OpenAI API로 폴백")
            return await call_openai_api(prompt, trace_id)
        
        raise Exception(f"vLLM API 호출 실패: {e}")


async def call_openai_api(prompt: str, trace_id: str = None) -> str:
    """OpenAI API로 텍스트 생성"""
    if not openai_client:
        raise RuntimeError("OpenAI API 키가 설정되지 않았습니다.")        
    
    model_name = API_CONFIG["openai_model"]
    generation = None
    start_time = time.time()

    # Generation 추적 설정
    if langfuse and trace_id:
        generation = langfuse.generation(
            trace_id=trace_id,
            name="openai_generation",
            model=model_name,
            input=[
                {"role": "system", "content": "IT 기술 면접 질문 생성 도우미"},
                {"role": "user", "content": prompt},
            ],
        )

    try:
        # API 호출
        response = await openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "IT 기술 면접 질문 생성 도우미"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=512,
        )
        
        execution_time = time.time() - start_time
        content = response.choices[0].message.content

        # 비용 계산 (gpt-4o-mini 기준)
        usage = response.usage
        cost = (
            (usage.prompt_tokens / 1_000_000) * 0.15 +
            (usage.completion_tokens / 1_000_000) * 0.60
        )

        logger.info(
            f"OpenAI API 호출 성공: {usage.prompt_tokens}+{usage.completion_tokens} 토큰, "
            f"${cost:.6f}, {execution_time:.2f}초"
        )

        # Generation 완료
        if generation:
            generation.end(
                output=content,
                usage={
                    "input": usage.prompt_tokens,
                    "output": usage.completion_tokens,
                    "total": usage.total_tokens,
                    "cost": cost,
                },
                metadata={"execution_time_seconds": execution_time},
            )

        return content

    except Exception as e:
        error_msg = f"OpenAI API 호출 실패: {str(e)}"
        logger.error(error_msg)
        
        if generation:
            generation.end(error=error_msg)
        
        raise


async def check_vllm_health() -> bool:
    """vLLM API 서버 상태 확인"""
    if vllm_client is None:
        return False
    
    try:
        await vllm_client.completions.create(
            model=VLLM_API_CONFIG["model_name"],
            prompt="test",
            max_tokens=1,
            temperature=0.1
        )
        return True
    except Exception as e:
        logger.error(f"vLLM API 상태 확인 실패: {e}")
        return False