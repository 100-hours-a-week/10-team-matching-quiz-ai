import os
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from transformers import logging as hf_logging
from trl import SFTTrainer
from peft import prepare_model_for_kbit_training, get_peft_model
from sklearn.model_selection import train_test_split

from config import get_bnb_config, get_lora_config
from dataset import load_dataset_from_jsonl  # 함수가 Dataset 반환해야 함

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
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# ─── LoRA 적용 ─────────────────────────────────────────────
model = get_peft_model(model, get_lora_config())
model.print_trainable_parameters()

# ─── 데이터 로딩 및 분할 ─────────────────────────────────────
dataset = load_dataset_from_jsonl(data_path)  # Dataset 객체 반환
# 만약 Dataset 객체가 아니라면 아래처럼 리스트 분할
if not hasattr(dataset, 'train_test_split'):
    # 리스트라면 Dataset.from_list로 변환 필요
    from datasets import Dataset
    dataset = Dataset.from_list(dataset)

split = dataset.train_test_split(test_size=0.1, seed=42)
train_data = split["train"]
val_data = split["test"]

# ─── 학습 파라미터 설정 ──────────────────────────────────────
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    max_grad_norm=0.3,
    num_train_epochs=15,
    learning_rate=2e-4,
    save_total_limit=3,
    save_strategy="steps",
    save_steps=200,
    logging_dir=f"{output_dir}/logs",
    logging_steps=10,
    output_dir=output_dir,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    max_steps=2000,
    report_to="tensorboard",
    evaluation_strategy="steps",     # ⬅️ Best 모델 선정 위한 평가 옵션
    eval_steps=200,
    load_best_model_at_end=True,     # ⬅️ Best checkpoint 자동 불러오기
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# ─── Trainer 생성 ──────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=training_args,
    peft_config=get_lora_config(),
)
trainer.tokenizer = tokenizer

def main():
    trainer.train(resume_from_checkpoint="./qwen3_lora_output_20250707_051148/checkpoint-1000")
    # Best checkpoint 자동 로드됨!
    model.save_pretrained(f"{output_dir}/final_model")
    tokenizer.save_pretrained(f"{output_dir}/final_model")
    print("훈련 완료 및 모델 저장됨:", output_dir)

if __name__ == "__main__":
    main()
