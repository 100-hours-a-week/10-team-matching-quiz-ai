import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from app.api.quiz_generator.quiz_generator_config import QUIZ_MODEL_NAME, QUIZ_HF_TOKEN

# 디바이스 설정
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

# Hugging Face 로그인
login(QUIZ_HF_TOKEN)

# 모델 디렉토리 경로 설정
local_model_dir = f"./models/{QUIZ_MODEL_NAME.split('/')[-1]}"

# 토크나이저 로딩
tokenizer = AutoTokenizer.from_pretrained(
    local_model_dir,
    trust_remote_code=True,
    local_files_only=True
)

# 모델 로딩 (양자화된 모델은 quantization_config 필요 없음)
model = AutoModelForCausalLM.from_pretrained(
    local_model_dir,
    trust_remote_code=True,
    local_files_only=True,
    device_map="auto",  # CUDA나 MPS 환경에서 자동으로 분배
    torch_dtype=dtype
).to(device)

# 퀴즈 생성 함수
def generate_quiz(prompt: str, max_tokens: int = 2500) -> str:
    print("prompt 생성 및 디바이스 전송 중...")

    prompt_tokens = tokenizer(prompt)['input_ids']
    print(f"[DEBUG] Prompt token 수: {len(prompt_tokens)}")

    # context window 제한 (예: 4096 토큰 기준)
    max_context = 4096 - max_tokens
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_context).to(device)

    print("quiz generate 시작")
    output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.8,
        do_sample=True,
        top_k=80,
        top_p=0.9,
        repetition_penalty=1.05
    )

    print("[DEBUG] 모델 generate 호출 결과:", output)
    print("quiz 생성 완료")

    return tokenizer.decode(output[0], skip_special_tokens=True)