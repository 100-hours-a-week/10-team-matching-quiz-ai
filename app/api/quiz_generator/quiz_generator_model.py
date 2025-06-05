import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from langfuse import Langfuse
from app.api.quiz_generator.quiz_generator_config import (
    QUIZ_MODEL_NAME,
    QUIZ_HF_TOKEN,
    QUIZ_LANGFUSE_SECRET_KEY,
    QUIZ_LANGFUSE_PUBLIC_KEY,
    QUIZ_LANGFUSE_HOST,
)

# Langfuse 클라이언트 초기화
langfuse = Langfuse(
    secret_key=QUIZ_LANGFUSE_SECRET_KEY,
    public_key=QUIZ_LANGFUSE_PUBLIC_KEY,
    host=QUIZ_LANGFUSE_HOST,
)

# 디바이스 및 dtype 설정
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32
print(f"디바이스 설정됨: {device}")

# Hugging Face 토큰 로그인
login(QUIZ_HF_TOKEN)

# 토크나이저 로딩 (로컬 아님, 모델 이름 직접 사용)
tokenizer = AutoTokenizer.from_pretrained(QUIZ_MODEL_NAME, trust_remote_code=True)

# 모델 로딩 (양자화된 모델이면 transformers가 자동 인식)
model = AutoModelForCausalLM.from_pretrained(
    QUIZ_MODEL_NAME, trust_remote_code=True, device_map="auto", torch_dtype=dtype
).to(device)


def initialize_quiz_model():
    """퀴즈 생성 모델 초기화 함수"""
    global model, tokenizer
    try:
        if model is not None and tokenizer is not None:
            print("퀴즈 모델이 이미 초기화되어 있습니다.")
            return model, tokenizer

        print(f"퀴즈 모델 초기화 시작: {QUIZ_MODEL_NAME}")

        # 토크나이저와 모델이 위에서 이미 로드됨
        print("퀴즈 모델 초기화 완료")
        return model, tokenizer
    except Exception as e:
        print(f"퀴즈 모델 초기화 실패: {e}")
        raise Exception(f"퀴즈 모델 초기화 실패: {e}")


# 퀴즈 생성 함수
def generate_quiz(prompt: str, max_tokens: int = 3000, trace_id: str = None) -> str:
    """
    퀴즈 생성 함수 with Langfuse tracking

    Args:
        prompt: 입력 프롬프트
        max_tokens: 최대 토큰 수
        trace_id: 부모 트레이스 ID (선택적)

    Returns:
        생성된 퀴즈 텍스트
    """
    generation = None

    try:
        print("prompt 생성 및 디바이스 전송 중...")

        prompt_tokens = tokenizer(prompt)["input_ids"]
        print(f"[DEBUG] Prompt token 수: {len(prompt_tokens)}")

        # Langfuse generation 객체 생성
        generation_kwargs = {
            "name": "quiz-generation",
            "model": QUIZ_MODEL_NAME,
            "input": prompt,
            "metadata": {
                "max_tokens": max_tokens,
                "temperature": 0.8,
                "top_k": 80,
                "top_p": 0.9,
                "repetition_penalty": 1.05,
                "device": str(device),
                "dtype": str(dtype),
                "prompt_tokens": len(prompt_tokens),
            },
        }

        if trace_id:
            generation_kwargs["trace_id"] = trace_id

        generation = langfuse.generation(**generation_kwargs)

        # context window 제한
        max_context = 16392 - max_tokens
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_context
        ).to(device)

        print("quiz generate 시작")

        # 모델 추론 실행
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.8,
                do_sample=True,
                top_k=80,
                top_p=0.9,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        print("quiz 생성 완료")

        # 출력 토큰 수 계산
        output_tokens = len(tokenizer(decoded)["input_ids"])

        # Langfuse generation 완료
        if generation:
            generation.end(
                output=decoded,
                usage={
                    "promptTokens": len(prompt_tokens),
                    "completionTokens": output_tokens - len(prompt_tokens),
                    "totalTokens": output_tokens,
                },
                metadata={
                    **generation_kwargs["metadata"],
                    "output_tokens": output_tokens - len(prompt_tokens),
                    "total_tokens": output_tokens,
                },
            )

        return decoded

    except Exception as e:
        error_message = f"Quiz 생성 중 오류 발생: {str(e)}"
        print(error_message)

        if generation:
            generation.end(
                error=error_message,
                metadata=(
                    generation_kwargs.get("metadata", {})
                    if "generation_kwargs" in locals()
                    else {}
                ),
            )

        raise Exception(error_message)
