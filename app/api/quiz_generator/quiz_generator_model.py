import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from app.api.quiz_generator.quiz_generator_config import QUIZ_MODEL_NAME, QUIZ_HF_TOKEN

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
tokenizer = AutoTokenizer.from_pretrained(
    QUIZ_MODEL_NAME,
    trust_remote_code=True
)

# 모델 로딩 (양자화된 모델이면 transformers가 자동 인식)
model = AutoModelForCausalLM.from_pretrained(
    QUIZ_MODEL_NAME,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=dtype
).to(device)


def generate_quiz(prompt: str, max_tokens: int = 1000, use_chat_template: bool = True) -> str:
    print("prompt 생성 및 디바이스 전송 중...")

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

    max_context = 2048 - max_tokens
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_context).to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.8,
        do_sample=True,
        top_k=80,
        top_p=0.9,
        repetition_penalty=1.05
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)