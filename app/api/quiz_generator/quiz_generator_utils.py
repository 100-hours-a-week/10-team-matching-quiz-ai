import os
import json
from huggingface_hub import snapshot_download

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
    print(f"[INFO] config.json 생성 완료: {config_path}")

def ensure_quantization_config(model_dir: str):
    quant_config_path = os.path.join(model_dir, "quantization_config.json")
    if os.path.exists(quant_config_path):
        print("[INFO] quantization_config.json 이미 존재함")
        return

    quant_config = {
        "quant_method": "bitsandbytes",
        "load_in_8bit": True,
        "bnb_4bit_use_double_quant": False,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "float16"
    }
    with open(quant_config_path, "w") as f:
        json.dump(quant_config, f, indent=2, ensure_ascii=False)
    print("[INFO] quantization_config.json 생성 완료")

def ensure_tokenizer_files(model_name: str, model_dir: str, hf_token: str):
    if not os.path.exists(os.path.join(model_dir, "tokenizer_config.json")):
        print("[INFO] tokenizer 관련 파일 다운로드 시작")
        snapshot_download(
            repo_id=model_name,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            token=hf_token,
            resume_download=True
        )
        print("[INFO] tokenizer 관련 파일 다운로드 완료")
    else:
        print("[INFO] tokenizer 관련 파일 이미 존재함")
