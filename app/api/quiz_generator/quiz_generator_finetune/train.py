from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from huggingface_hub import login
import os

# 1. 허깅페이스 인증 (필수! - 토큰 노출 유의, 실제론 환경변수로 관리 추천)
login(os.getenv("HF_TOKEN"))

# 2. 모델 세팅
max_seq_length = 2048
dtype = torch.bfloat16  # bf16/fp16은 서버 환경에 따라 조절
load_in_4bit = True

model_name = "unsloth/Qwen3-8B"  # ✅ 실제로 허깅페이스에 있는 이름!
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# 3. PEFT(LoRA) 어댑터 적용 (중요!)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=123,
    use_rslora=False,
    loftq_config=None,
)

# 4. 데이터셋 로딩
dataset = load_dataset("json", data_files="quiz_finetune_unsloth.jsonl", split="train")

# 5. Alpaca 스타일 프롬프트 포맷 함수 (instruction/output key에 맞춰서 조정!)
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

# 6. 트레이너 준비
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

# 7. 학습 실행
trainer.train()

# 8. 최종 모델(어댑터 포함) 저장 및 push_to_hub 예시
final_output_dir = "./unsloth_qwen3_output/final_model"
trainer.save_model(final_output_dir)
tokenizer.save_pretrained(final_output_dir)

# 9. 허깅페이스로 push (adapter_config.json, adapter_model.safetensors 포함)
# (본인 허브에 쓸 새 모델명 지정!)
model.push_to_hub("Chan-980730/wingterview-qwen3-8b-lora-quiz", use_temp_dir=False)
tokenizer.push_to_hub("Chan-980730/wingterview-qwen3-8b-lora-quiz")
