from unsloth import FastLanguageModel
import torch
from datasets import load_dataset

max_seq_length = 2048
dtype = torch.bfloat16  # 서버 환경에 따라 bf16/fp16 결정
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-8B",   # ✅ Unsloth 공식 Qwen3-8B (4bit 지원!)
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# LoRA 어댑터 추가 (필수)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=123,
    use_rslora=False,
    loftq_config=None,
)

# 데이터셋 로딩 (jsonl → Huggingface Dataset)
dataset = load_dataset("json", data_files="quiz_finetune_unsloth_alpaca.jsonl", split="train")

# Alpaca prompt 포맷 함수
EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    return {
        "text": [
            f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{inst}\n\n### Response:\n{out}{EOS_TOKEN}"
            for inst, out in zip(examples["instruction"], examples["output"])
        ]
    }

dataset = dataset.map(formatting_prompts_func, batched=True)
tokenizer.padding_side = "right"

from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        max_steps=1000,
        logging_steps=10,
        output_dir="./unsloth_qwen3_output",
        save_strategy="steps",
        save_steps=200,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        warmup_steps=10,
    ),
)
trainer.train()
trainer.save_model("./unsloth_qwen3_output/final_model")
tokenizer.save_pretrained("./unsloth_qwen3_output/final_model")
