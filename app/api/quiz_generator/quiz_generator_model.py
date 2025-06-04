from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from app.api.quiz_generator.quiz_generator_config import QUIZ_MODEL_NAME
from app.api.quiz_generator.quiz_generator_config import QUIZ_HF_TOKEN
from huggingface_hub import login
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# 전역 변수로 모델과 토크나이저 저장 (지연 로딩을 위해)
_model: Optional[AutoModelForCausalLM] = None
_tokenizer: Optional[AutoTokenizer] = None
_device: Optional[str] = None


def get_device() -> str:
    """디바이스 설정 반환"""
    global _device
    if _device is None:
        if torch.cuda.is_available():
            _device = "cuda"
        elif torch.backends.mps.is_available():
            _device = "mps"
        else:
            _device = "cpu"
        logger.info(f"디바이스 설정됨: {_device}")
    return _device


def initialize_quiz_model() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Quiz 모델과 토크나이저 초기화 (지연 로딩)"""
    global _model, _tokenizer

    if _model is not None and _tokenizer is not None:
        logger.info("Quiz 모델이 이미 초기화되어 있습니다.")
        return _model, _tokenizer

    try:
        device = get_device()

        # HuggingFace 로그인
        if QUIZ_HF_TOKEN:
            login(QUIZ_HF_TOKEN)

        logger.info(f"Quiz 모델 로딩 시작: {QUIZ_MODEL_NAME}")

        # 토크나이저 로딩
        _tokenizer = AutoTokenizer.from_pretrained(
            QUIZ_MODEL_NAME, trust_remote_code=True
        )

        # 모델 로딩
        _model = AutoModelForCausalLM.from_pretrained(
            QUIZ_MODEL_NAME,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            trust_remote_code=True,
        ).to(device)

        logger.info(f"Quiz 모델 로딩 완료: {QUIZ_MODEL_NAME} on {device}")
        return _model, _tokenizer

    except Exception as e:
        logger.error(f"Quiz 모델 초기화 중 오류 발생: {e}")
        raise


def get_quiz_model() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Quiz 모델과 토크나이저 반환 (필요시 초기화)"""
    if _model is None or _tokenizer is None:
        return initialize_quiz_model()
    return _model, _tokenizer


def generate_quiz(prompt: str, max_tokens: int = 2500) -> str:
    """Quiz 생성 함수 - 지연 로딩된 모델 사용"""
    logger.info("Quiz 생성 시작")

    # 모델과 토크나이저 가져오기 (필요시 초기화)
    model, tokenizer = get_quiz_model()
    device = get_device()

    prompt_tokens = tokenizer(prompt)["input_ids"]
    logger.info(f"Prompt token 수: {len(prompt_tokens)}")

    # prompt 길이 제한 적용
    max_context = 4096 - max_tokens
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_context
    ).to(device)

    logger.info("Quiz 모델 추론 시작")
    output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.8,
        do_sample=True,
        top_k=80,
        top_p=0.9,
        repetition_penalty=1.05,
    )

    result = tokenizer.decode(output[0], skip_special_tokens=True)
    logger.info("Quiz 생성 완료")
    return result


def cleanup_quiz_model():
    """Quiz 모델 메모리 정리"""
    global _model, _tokenizer

    if _model is not None:
        del _model
        _model = None

    if _tokenizer is not None:
        del _tokenizer
        _tokenizer = None

    # 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    logger.info("Quiz 모델 메모리 정리 완료")
