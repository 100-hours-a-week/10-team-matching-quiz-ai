import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from transformers import logging as hf_logging
from trl import SFTTrainer
from peft import prepare_model_for_kbit_training, get_peft_model
from datetime import datetime

from config import get_bnb_config, get_lora_config
from dataset import load_dataset_from_jsonl

hf_logging.set_verbosity_info()

# ─── 환경 설정 ─────────────────────────────────────────────
model_name = "Qwen/Qwen3-8B"
data_path = "quiz_finetune_ft.jsonl"
time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"./qwen3_lora_output_{time_stamp}"
os.makedirs(output_dir, exist_ok=True)

# ─── 모델 로딩 ─────────────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=get_bnb_config(),
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
model = prepare_model_for_kbit_training(model)

# ─── 토크나이저 ─────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(
    model_name, trust_remote_code=True
)

# ─── LoRA 적용 ─────────────────────────────────────────────
model = get_peft_model(model, get_lora_config())
model.print_trainable_parameters()

# ─── 데이터 로딩 ───────────────────────────────────────────
dataset = load_dataset_from_jsonl(data_path)

# ─── 학습 파라미터 설정 ──────────────────────────────────────
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    max_grad_norm=0.3,
    num_train_epochs=15,
    learning_rate=2e-4,
    save_total_limit=3,
    save_strategy="steps",                # 스탭마다 저장
    save_steps=100,
    logging_dir=f"{output_dir}/logs",     # 텐서보드 로그 저장 위치
    logging_steps=10,
    output_dir=output_dir,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    max_steps=1000,
    report_to="tensorboard"
)

# ─── 학습 실행 ─────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    peft_config=get_lora_config()
)

trainer.tokenizer = tokenizer


# ─── 모델 저장 ─────────────────────────────────────────────
model.save_pretrained(f"{output_dir}/final_model")
tokenizer.save_pretrained(f"{output_dir}/final_model")

print("훈련 완료 및 모델 저장됨:", output_dir)

def main():
    trainer.train(resume_from_checkpoint="./qwen3_lora_output_20250703_003109/checkpoint-300")

    # ─── 모델 저장 ─────────────────────────────────────────────
    model.save_pretrained(f"{output_dir}/final_model")
    tokenizer.save_pretrained(f"{output_dir}/final_model")

    print("훈련 완료 및 모델 저장됨:", output_dir)


if __name__ == "__main__":
    main()

