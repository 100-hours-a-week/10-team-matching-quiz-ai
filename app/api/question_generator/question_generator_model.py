import asyncio
import torch
import re
import logging
from vllm import LLM, SamplingParams
from langfuse import Langfuse
import os
from typing import Optional, Union, Dict, Any
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

MODEL_PATH = os.getenv(
    "LLM_MODEL_PATH")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

langfuse = Langfuse(
    secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
    public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
    host=os.getenv('LANGFUSE_HOST')
)

# mac 구현용 코드 - 4비트 모델을 사용할 경우 dtype 설정이 중요함
# use_mps = torch.backends.mps.is_available()
# dtype = "float16" if use_mps and torch.backends.mps.is_built() else "float32"

dtype = os.getenv("DTYPE", "auto")  # 모델의 dtype, 기본은 auto로 설정
# 문자열을 float로 변환, 타임아웃 설정으로 무한 대기 방지
timeout = float(os.getenv("LLM_TIMEOUT", "60.0"))

# 지연 초기화를 위한 전역 변수
llm = None


def initialize_llm():
    """LLM을 초기화하는 함수 - 처음에 실행할때만 호출"""
    global llm
    if llm is not None:
        return llm

    try:
        llm = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=1,  # gpu에 따라 다르지만 우선 1로 설정
            trust_remote_code=True,
            dtype=dtype,
            # quantization="bitsandbytes",  # 4비트 모델에 맞는 양자화 설정 -> vllm에서는 현재 불가능
            download_dir="./model_cache",  # 캐싱 디렉토리 설정
            max_model_len=4096,  # 모델의 크기와 응답 속도에 따라 조정
        )
        logger.info(f"Hugging Face 모델 로드 성공: {MODEL_PATH}")
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
        params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.15,
            max_tokens=400,
            stop=['질문 5.', '질문 6.']
        )
        try:
            outputs = llm.generate([prompt], params)
            return outputs[0].outputs[0].text
        except Exception as e:
            logger.error(f"LLM 생성 오류: {e}")
            raise

    try:
        # 부모 트레이스 ID 사용 또는 새로운 트레이스 생성
        generation_kwargs = {
            "name": "local-llm-call",
            "model": MODEL_PATH,
            "input": prompt,
        }

        # 부모 트레이스 ID가 있으면 그것을 사용
        if trace_id:
            generation_kwargs["trace_id"] = trace_id

        generation = langfuse.generation(**generation_kwargs)

        start_time = asyncio.get_event_loop().time()
        result = await asyncio.wait_for(
            asyncio.to_thread(_generate),
            timeout=timeout
        )
        execution_time = asyncio.get_event_loop().time() - start_time

        # 성공 결과 기록
        generation.end(
            output=result,
            metadata={"execution_time_seconds": execution_time}
        )

        return result
    except (asyncio.TimeoutError, Exception) as e:
        error_type = "TimeoutError" if isinstance(
            e, asyncio.TimeoutError) else "Error"
        error_message = "LLM 응답 시간 초과" if isinstance(
            e, asyncio.TimeoutError) else f"LLM 호출 실패: {str(e)}"

        # 에러 기록
        if locals().get('generation'):
            generation.end(error=error_message)

        logger.error(error_message)

        # Fallback 로직
        if try_fallback and OPENAI_API_KEY:
            logger.info("로컬 LLM 실패, OpenAI API로 fallback")
            return await call_openai_api(prompt, trace_id)

        # Fallback이 비활성화되었거나 OpenAI 키가 없는 경우 예외 발생
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
        # 부모 트레이스 ID 사용 또는 새로운 트레이스 생성
        generation_kwargs = {
            "name": "openai-api-call",
            "model": model_name,
            "input": [
                {"role": "system", "content": "당신은 IT 기술 면접을 위한 질문을 생성하는 도우미입니다."},
                {"role": "user", "content": prompt}
            ],
        }

        # 부모 트레이스 ID가 있으면 그것을 사용
        if trace_id:
            generation_kwargs["trace_id"] = trace_id

        generation = langfuse.generation(**generation_kwargs)

        # OpenAI API 호출
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

        # 토큰 사용량 및 비용 계산
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        # OpenAI의 gpt-4o-mini 모델 요금
        cost_per_1k_input = 0.00015  # $0.00015 per 1K input tokens
        cost_per_1k_output = 0.0006   # $0.0006 per 1K output tokens
        cost = (input_tokens / 1000 * cost_per_1k_input) + \
            (output_tokens / 1000 * cost_per_1k_output)

        # 상세 로깅
        logger.info(
            f"OpenAI API 호출 성공: 입력 토큰 {input_tokens}, 출력 토큰 {output_tokens}, 비용 ${cost:.5f}")

        # Langfuse에 사용량 및 비용 업데이트
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
        # 에러가 발생한 경우에도 Langfuse에 기록
        if locals().get('generation'):
            generation.end(error=str(e))
        logger.error(f"OpenAI API 호출 오류: {e}")
        raise
