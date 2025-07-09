from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./unsloth_qwen3_output/final_model",
    max_seq_length=2048,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
instruction = "아래 문장을 바탕으로 객관식 4지선다 퀴즈 한 문제를 생성해줘. ... (생략) ... 문장: 인공지능(AI)은 데이터를 학습하여 패턴을 인식하고, 예측 및 결정을 내리는 기술입니다."
inputs = tokenizer(
    [
        f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
    ],
    return_tensors="pt",
).to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
