import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
from app.api.quiz_generator.quiz_generator_config import QUIZ_MODEL_NAME, QUIZ_HF_TOKEN

# 디바이스 설정
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"디바이스 설정됨: {device}")

# Hugging Face 로그인
login(QUIZ_HF_TOKEN)

# 모델 디렉토리
local_model_dir = f"./models/{QUIZ_MODEL_NAME.split('/')[-1]}"

# BitsAndBytes 양자화 설정
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    quant_method="bitsandbytes",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 토크나이저 로딩
tokenizer = AutoTokenizer.from_pretrained(local_model_dir, trust_remote_code=True)

# 모델 로딩
model = AutoModelForCausalLM.from_pretrained(
    local_model_dir,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=quant_config,
    torch_dtype=torch.float16 if device != "cpu" else torch.float32,
).to(device)


def generate_quiz(prompt: str, max_tokens: int = 2500) -> str:
    print("prompt 생성 및 디바이스 전송 중...")

    prompt_tokens = tokenizer(prompt)['input_ids']
    print(f"[DEBUG] Prompt token 수: {len(prompt_tokens)}")

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
