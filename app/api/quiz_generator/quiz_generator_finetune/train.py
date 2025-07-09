import torch
from unsloth import FastLanguageModel
from datasets import load_dataset

# 1. Unsloth 모델 로딩
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-8B",   # ✅ 요걸로 하세요!
    max_seq_length = 2048,
    dtype = torch.bfloat16,
    load_in_4bit = True,
)

# 2. 데이터셋 로딩 (Huggingface Dataset)
dataset = load_dataset("json", data_files="quiz_finetune_unsloth.jsonl", split="train")

# 3. input/output을 text로 합치기 (Unsloth 권장 방식)
def format_example(example):
    # instruction이 필요하면 맨 앞에 추가!
    return {"text": f"{example['input']}\n{example['output']}"}
dataset = dataset.map(format_example)

# 4. SFTTrainer 선언 (Unsloth용)
from unsloth import SFTTrainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    max_seq_length = 2048,
    batch_size = 2,
    gradient_accumulation_steps = 2,
    lr_scheduler_type = "cosine",
    warmup_ratio = 0.05,
    num_train_epochs = 3,
    learning_rate = 2e-4,
    save_steps = 200,
    output_dir = "./unsloth_qwen3_output",
    logging_steps = 10,
    report_to = "tensorboard",
)

# 5. 학습 실행
trainer.train()

# 6. 모델 저장
trainer.save_model("./unsloth_qwen3_output/final_model")
tokenizer.save_pretrained("./unsloth_qwen3_output/final_model")
