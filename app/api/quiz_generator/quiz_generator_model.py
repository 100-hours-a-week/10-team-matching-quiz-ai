import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from app.api.quiz_generator.quiz_generator_config import QUIZ_MODEL_NAME, QUIZ_HF_TOKEN
import logging

logger = logging.getLogger(__name__)

# 전역 변수로 모델과 토크나이저 선언 (지연 로딩을 위해)
model = None
tokenizer = None
device = None
dtype = None


def initialize_quiz_model():
    """퀴즈 생성 모델을 지연 초기화하는 함수"""
    global model, tokenizer, device, dtype

    if model is not None and tokenizer is not None:
        logger.info("Quiz model already initialized")
        return model, tokenizer

    logger.info("Initializing quiz generation model...")

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

        logger.info(f"Device set to: {device}")

        # Hugging Face 토큰 로그인
        if QUIZ_HF_TOKEN:
            login(QUIZ_HF_TOKEN)
            logger.info("Successfully logged in to Hugging Face")

        # 토크나이저 로딩
        logger.info(f"Loading tokenizer: {QUIZ_MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(
            QUIZ_MODEL_NAME, trust_remote_code=True
        )

        # 모델 로딩
        logger.info(f"Loading model: {QUIZ_MODEL_NAME}")
        model = AutoModelForCausalLM.from_pretrained(
            QUIZ_MODEL_NAME,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=dtype,
            load_in_4bit=True
        ).to(device)

        logger.info("Quiz model initialized successfully")
        return model, tokenizer

    except Exception as e:
        logger.error(f"Failed to initialize quiz model: {str(e)}")
        raise


def generate_quiz(prompt: str, max_tokens: int = 2048, use_chat_template: bool = False) -> str:
    """퀴즈 생성 함수 - 모델 관리자를 통해 접근"""
    try:
        # 모델 관리자를 통해 모델 데이터 가져오기 (지연 로딩 지원)
        from app.main import get_model

        model_data = get_model("quiz_generator")
        if not model_data:
            raise RuntimeError("quiz_generator 모델을 로드할 수 없습니다.")

        model = model_data["model"]
        tokenizer = model_data["tokenizer"]

        use_chat_template = True

        if use_chat_template:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        else:
            text = prompt

        logger.info("Starting quiz generation...")
        logger.debug(f"Input prompt length: {len(prompt)} characters")

        # 토큰 수 확인
        prompt_tokens = tokenizer(prompt)["input_ids"]
        logger.debug(f"Prompt token count: {len(prompt_tokens)}")

        # 디바이스 설정
        device = next(model.parameters()).device

        # Context window 제한 (모델의 최대 길이에 맞춰 조정)
        max_context = 4096 - max_tokens
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_context
        ).to(device)

        logger.info("Generating quiz content...")

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
        )

        # 디코딩
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.info("Quiz generation completed successfully")

        return decoded

    except Exception as e:
        logger.error(f"Error during quiz generation: {str(e)}")
        raise


def cleanup_quiz_model():
    """모델 메모리 정리"""
    global model, tokenizer, device, dtype

    try:
        if model is not None:
            del model
            model = None
            logger.info("Quiz model memory cleaned up")

        if tokenizer is not None:
            del tokenizer
            tokenizer = None
            logger.info("Tokenizer memory cleaned up")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")

    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")


def get_model_status():
    """모델 상태 정보 반환"""
    return {
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "device": device,
        "dtype": str(dtype) if dtype else None,
        "model_name": QUIZ_MODEL_NAME,
    }
