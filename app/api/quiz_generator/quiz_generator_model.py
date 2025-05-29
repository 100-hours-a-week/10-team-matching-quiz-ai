from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from app.api.quiz_generator.quiz_generator_config import QUIZ_MODEL_NAME

# 1. 디바이스 설정: Mac M1/M2는 mps, 그 외는 cpu
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"디바이스 설정됨: {device}")

# 2. 토크나이저 로딩
tokenizer = AutoTokenizer.from_pretrained(QUIZ_MODEL_NAME, trust_remote_code=True)

# 3. 모델 로딩 후 디바이스 지정
model = AutoModelForCausalLM.from_pretrained(
    QUIZ_MODEL_NAME,
    torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    trust_remote_code=True
).to(device)  # 👉 모델을 직접 디바이스로 이동

# 4. generate_quiz 함수 내에서도 입력을 같은 디바이스로 이동
def generate_quiz(prompt: str, max_tokens: int = 300) -> str:
    print("prompt 생성 및 디바이스 전송 중...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)  

    print("모델 generate 시작")
    output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.7,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )

    print("모델 응답 생성 완료")
    return tokenizer.decode(output[0], skip_special_tokens=True)
