import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from app.api.quiz_generator.quiz_generator_config import QUIZ_MODEL_NAME, QUIZ_HF_TOKEN
import logging
import sys

# 로거 설정 / 핸들러
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)  # stdout에 출력
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# 전역 변수로 모델과 토크나이저 선언 (Worker 전용)
model = None
tokenizer = None
device = None
dtype = None


def initialize_quiz_model():
    """
    퀴즈 생성 모델을 초기화하는 함수 (Worker 전용),
    전역 변수에 저장
    """
    global model, tokenizer, device, dtype

    # 이미 초기화 완료된 경우 중복 로딩 방지
    if model is not None and tokenizer is not None:
        logger.info("Quiz Model이 초기화되어 재로딩 하지 않음")
        return model, tokenizer

    logger.info("Quiz Model 초기화 시작")

    try:
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

        logger.info(f"Device 설정: {device}")

        # Hugging Face 토큰 로그인
        if QUIZ_HF_TOKEN:
            login(QUIZ_HF_TOKEN)
            logger.info("Hugging Face 로그인 완료")

        # 토크나이저 로딩
        logger.info(f"Tokenizer 로딩 중: {QUIZ_MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(
            QUIZ_MODEL_NAME, trust_remote_code=True
        )

        # 모델 로딩
        logger.info(f"Quiz Model 로딩 중: {QUIZ_MODEL_NAME}")
        model = AutoModelForCausalLM.from_pretrained(
            QUIZ_MODEL_NAME,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=dtype,
            load_in_4bit=True
        ).to(device)

        logger.info("Quiz Model 초기화 완료")
        return model, tokenizer

    except Exception as e:
        logger.error(f"Quiz Model 초기화 실패: {str(e)}")
        raise


def generate_quiz(prompt: str, max_tokens: int = 4096, use_chat_template: bool = True) -> str:
    """퀴즈 생성 함수 (Worker 전용 - 직접 모델 접근)"""
    global model, tokenizer
    
    try:
        # 모델이 아직 Load되지 않았으면 초기화 시도
        if model is None or tokenizer is None:
            logger.info("Quiz Model이 Load되지 않음 -> 초기화 시도")
            initialize_quiz_model()

        # 초기화 실패 재확인    
        if model is None or tokenizer is None:
            raise RuntimeError("Quiz Model을 로드할 수 없음")

        # 채팅 템플릿 강제 적용
        if use_chat_template:
            messages = [{"role": "user", "content": prompt}]
            # 모델 Prompt 가공
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False # 추론 생략
            )
        else:
            text = prompt

        logger.info("Quiz 생성 시작")
        logger.debug(f"Prompt 길이: {len(text)}")

        # 토큰 수 확인
        prompt_tokens = tokenizer(text)["input_ids"]
        logger.debug(f"Prompt token 개수: {len(prompt_tokens)}")

        # 디바이스 설정
        device = next(model.parameters()).device

        # Context window 제한 (모델의 최대 길이에 맞춰 조정)
        max_context = 8192 - max_tokens
        
        # Token화 & Tensor 변환
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_context
        ).to(device)

        logger.info("Quiz Model이 Quiz 생성 중")

        # 모델 생성 파라미터 최적화
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.8,
            do_sample=True,
            top_k=80,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        # 디코딩
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.info("Quiz generation 완료")

        return decoded

    except Exception as e:
        logger.error(f"Quiz 생성 중 Error: {str(e)}")
        raise


def cleanup_quiz_model():
    """모델 메모리 정리 (Worker 전용)"""
    global model, tokenizer, device, dtype

    try:
        if model is not None:
            del model
            model = None
            logger.info("Quiz model 메모리 정리")

        if tokenizer is not None:
            del tokenizer
            tokenizer = None
            logger.info("Tokenizer 메모리 정리")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU(CUDA) 캐시 메모리 정리")

    except Exception as e:
        logger.error(f"메모리 정리 중 오류 발생: {str(e)}")


def get_model_status():
    """모델 상태 정보 반환 (Worker 전용)"""
    try:
        return {
            "model_loaded": model is not None,
            "tokenizer_loaded": tokenizer is not None,
            "device": device,
            "dtype": str(dtype) if dtype else None,
            "model_name": QUIZ_MODEL_NAME,
            "mode": "worker"
        }
    except Exception as e:
        logger.error(f"모델 상태 확인 중 오류 발생: {str(e)}")
        return {"error": "상태 확인 실패"}