from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from app.api.quiz_generator.quiz_generator_config import QUIZ_MODEL_NAME
from app.api.quiz_generator.quiz_generator_config import QUIZ_HF_TOKEN
from huggingface_hub import login


# 디바이스 설정: Mac M1/M2는 mps, 그 외는 cpu
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"디바이스 설정됨: {device}")

# 토크나이저 로딩
login(QUIZ_HF_TOKEN)
tokenizer = AutoTokenizer.from_pretrained(QUIZ_MODEL_NAME, trust_remote_code=True)

# 모델 로딩 후 디바이스 지정
model = AutoModelForCausalLM.from_pretrained(
    QUIZ_MODEL_NAME,
    torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    trust_remote_code=True
).to(device)  


def generate_quiz(prompt: str, max_tokens: int = 4000) -> str:
    print("prompt 생성 및 디바이스 전송 중...")

    # 출력 시작을 유도할 prefix
    prefix = "문제 1  \n난이도: 하  \n문제: "
    
    # prefix를 prompt에 붙여줌
    full_prompt = prompt + "\n" + prefix

    # tokenizer 처리
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

    print("quiz generate 시작")
    output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.8,
        do_sample=True,
        top_k=80,
        top_p=0.9,
        repetition_penalty=1.05,
        stream=False
    )

    print("quiz 생성 완료")
    return tokenizer.decode(output[0], skip_special_tokens=True)
