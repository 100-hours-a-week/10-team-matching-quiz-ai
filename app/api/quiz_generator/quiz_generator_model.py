import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login, snapshot_download
from app.api.quiz_generator.quiz_generator_config import QUIZ_MODEL_NAME, QUIZ_HF_TOKEN

# 디바이스 설정
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"디바이스 설정됨: {device}")

# config.json 자동 생성 함수
def ensure_config_json(model_dir: str, model_name: str):
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        print("[INFO] config.json 이미 존재함")
        return
    
    os.makedirs(model_dir, exist_ok=True)
    config_data = {
        "_name_or_path": model_name,
        "model_type": "qwen3",
        "architectures": ["QWenLMHeadModel"],
        "trust_remote_code": True,
        "quantization_config": {
            "quant_method": "gptq",
            "bits": 8,
            "group_size": 128,
            "desc_act": False
        }
    }
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    print(f"config.json 생성 완료: {config_path}")


# 모델 전체 스냅샷을 저장
local_model_dir = f"./models/{QUIZ_MODEL_NAME.split('/')[-1]}"
if not os.path.exists(os.path.join(local_model_dir, "tokenizer_config.json")):
    print("[INFO] tokenizer 관련 파일 다운로드 시작")
    snapshot_download(
        repo_id=QUIZ_MODEL_NAME,
        local_dir=local_model_dir,
        local_dir_use_symlinks=False,  # symlink 오류 방지
        token=QUIZ_HF_TOKEN,
        resume_download=True
    )
    print("tokenizer 관련 파일 다운로드 완료")


# === 실행 ===
login(QUIZ_HF_TOKEN)

# Hugging Face 모델명에서 로컬 경로 추출
local_model_dir = f"./models/{QUIZ_MODEL_NAME.split('/')[-1]}"
ensure_config_json(local_model_dir, QUIZ_MODEL_NAME)

# 토크나이저 로딩
tokenizer = AutoTokenizer.from_pretrained(local_model_dir, trust_remote_code=True)

# 모델 로딩
model = AutoModelForCausalLM.from_pretrained(
    local_model_dir,
    torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    trust_remote_code=True,
).to(device)


def generate_quiz(prompt: str, max_tokens: int = 2500) -> str:
    print("prompt 생성 및 디바이스 전송 중...")

    prompt_tokens = tokenizer(prompt)['input_ids']
    print(f"[DEBUG] Prompt token 수: {len(prompt_tokens)}")

    # prompt 길이 제한 적용
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
