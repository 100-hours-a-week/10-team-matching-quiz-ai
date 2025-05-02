import asyncio
import torch
import re
import logging
from vllm import LLM, SamplingParams
from langfuse import Langfuse
import os
from typing import Optional
from huggingface_hub import login
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hugging Face 토큰 설정 (필요한 경우 환경 변수에서 가져오기)
HF_TOKEN = os.getenv("HF_TOKEN")

# 모델 경로를 Hugging Face 모델 ID로 변경
MODEL_PATH = os.getenv(
    "LLM_MODEL_PATH", "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

langfuse = Langfuse(
    secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
    public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
    host=os.getenv('LANGFUSE_HOST')
)

# mac 구현 코드 - 4비트 모델을 사용할 경우 dtype 설정이 중요함
# use_mps = torch.backends.mps.is_available()
# dtype = "float16" if use_mps and torch.backends.mps.is_built() else "float32"
dtype = "float16"

try:
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        trust_remote_code=True,
        dtype=dtype,
        # quantization="bitsandbytes",  # 4비트 모델에 맞는 양자화 설정
        download_dir="./model_cache",  # 캐싱 디렉토리 설정
        max_model_len=4096,
    )
    logger.info(f"Hugging Face 모델 로드 성공: {MODEL_PATH}")
except Exception as e:
    logger.error(f"Hugging Face 모델 로드 실패: {e}")
    raise


async def call_llm(prompt: str) -> str:
    """
    LLM에 프롬프트를 안전하게 전송하고 생성된 응답을 반환합니다.

    Parameters:
        prompt (str): LLM에 전달할 텍스트 프롬프트

    Returns:
        str: 모델이 생성한 응답 텍스트

    Raises:
        Exception: LLM 호출 과정에서 발생한 예외
    """

    def _generate():
        params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.15,
            max_tokens=200,
            stop=['질문 5.', '질문 6.']
        )
        try:
            outputs = llm.generate([prompt], params)
            return outputs[0].outputs[0].text
        except Exception as e:
            logger.error(f"LLM 생성 오류: {e}")
            raise

    try:
        # 타임아웃 설정으로 무한 대기 방지
        return await asyncio.wait_for(
            asyncio.to_thread(_generate),
            timeout=30.0  # 30초 타임아웃
        )
    except asyncio.TimeoutError:
        logger.error("LLM 응답 시간 초과")
        raise TimeoutError("LLM 응답 시간 초과")
    except Exception as e:
        logger.error(f"LLM 호출 실패: {str(e)}")
        raise Exception(f"LLM 호출 실패: {str(e)}")


async def call_openai_api(prompt: str):
    """
    OpenAI API를 사용하여 프롬프트에 대한 응답을 생성합니다.
    주로 부족한 질문을 채우기 위한 백업 메커니즘으로 사용됩니다.

    Parameters:
        prompt (str): OpenAI 모델에 전달할 텍스트 프롬프트

    Returns:
        Dict[str, Any]: {
            'text': 모델이 생성한 응답 텍스트,
            'input_tokens': 입력 토큰 수,
            'output_tokens': 출력 토큰 수,
            'cost': 실제 비용(USD)
        }

    Raises:
        Exception: API 호출 과정에서 발생한 예외
    """
    # Langfuse client 초기화 (환경 변수에서 키를 가져오거나 직접 설정)
    model_name = "gpt-4o-mini"

    try:
        # Langfuse로 Generation 추적 시작
        generation = langfuse.generation(
            name="openai-api-call",
            model=model_name,
            input=[
                {"role": "system", "content": "당신은 IT 기술 면접을 위한 질문을 생성하는 도우미입니다."},
                {"role": "user", "content": prompt}
            ],
        )

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

        # OpenAI의 gpt-4o-mini 모델 요금 (최신 요금 확인: https://openai.com/pricing)
        cost_per_1k_input = 0.00015  # $0.00015 per 1K input tokens
        cost_per_1k_output = 0.0006   # $0.0006 per 1K output tokens
        cost = (input_tokens / 1000 * cost_per_1k_input) + \
            (output_tokens / 1000 * cost_per_1k_output)

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
