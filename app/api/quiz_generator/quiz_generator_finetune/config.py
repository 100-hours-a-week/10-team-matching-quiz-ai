from peft import LoraConfig
from transformers import BitsAndBytesConfig
import torch

def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

def get_lora_config():
    return LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",           # 문맥 이해 및 응답 논리
            "gate_proj", "up_proj", "down_proj",              # 표현력, 정교한 포맷 반영
            "lm_head"                                          # 최종 출력 형식 제어
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )