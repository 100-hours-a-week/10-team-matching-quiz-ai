import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse

def load_model(model_path, base_model="Qwen/Qwen3-8B"):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base, model_path)
    model.eval()
    return tokenizer, model

def generate_answer(tokenizer, model, instruction, input_text=None, max_new_tokens=512):
    if input_text:
        prompt = f"{instruction}\n\n{input_text}"
    else:
        prompt = instruction

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.7
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="훈련된 모델 경로")
    parser.add_argument("--instruction", type=str, required=True, help="명령/instruction 입력")
    parser.add_argument("--input", type=str, default=None, help="질문/입력 텍스트")
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_dir)
    output = generate_answer(tokenizer, model, args.instruction, args.input)
    print("\n🧠 생성된 응답:\n", output)
