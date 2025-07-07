import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from transformers import logging as hf_logging
from trl import SFTTrainer
from peft import prepare_model_for_kbit_training, get_peft_model
from datetime import datetime
from sklearn.model_selection import train_test_split

from config import get_bnb_config, get_lora_config
from dataset import load_dataset_from_jsonl

hf_logging.set_verbosity_info()

# ─── 환경 설정 ─────────────────────────────────────────────
model_name = "Qwen/Qwen3-8B"
data_path = "quiz_finetune.jsonl"
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

# ─── 데이터 로딩 및 분할 ─────────────────────────────────────
dataset = load_dataset_from_jsonl(data_path)
# dataset: list of dict (or Dataset object)
if isinstance(dataset, list):  # 만약 리스트라면 train_test_split
    train_data, val_data = train_test_split(dataset, test_size=0.1, random_state=42)
else:
    # Huggingface Dataset이면 .train_test_split 가능
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_data, val_data = split["train"], split["test"]

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
    save_steps=200,
    logging_dir=f"{output_dir}/logs",     # 텐서보드 로그 저장 위치
    logging_steps=10,
    output_dir=output_dir,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    max_steps=2000,
    report_to="tensorboard",
    evaluation_strategy="steps",          # [추가] 평가 전략
    eval_steps=200,                      # [추가] N 스텝마다 평가
    load_best_model_at_end=True,          # [추가] 베스트 체크포인트 자동 로딩
    metric_for_best_model="eval_loss",    # [추가] 평가 기준
    greater_is_better=False,              # [추가] loss 기준이므로 False
)

# ─── Trainer 생성 ──────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=training_args,
    peft_config=get_lora_config()
)
trainer.tokenizer = tokenizer

def main():
    trainer.train(resume_from_checkpoint="./qwen3_lora_output_20250707_070819/checkpoint-1300")  # resume 쓸 거면 체크포인트 경로, 아니면 None

    # 베스트 모델 저장 (자동으로 best로 로딩됨)
    model.save_pretrained(f"{output_dir}/final_model")
    tokenizer.save_pretrained(f"{output_dir}/final_model")

    print("훈련 완료 및 모델 저장됨:", output_dir)

if __name__ == "__main__":
    main()
