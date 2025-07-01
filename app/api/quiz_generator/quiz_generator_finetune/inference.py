import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
import time

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
        prompt = f"{instruction}\n\n{input_text}" if instruction else input_text
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
    parser.add_argument("--instruction", type=str, default="", help="명령/instruction 입력")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="최대 출력 토큰 수")
    args = parser.parse_args()

    # 예시로 10개 input (여기다 원하는 문제 리스트 넣으면 됨)
    inputs = [
        "REST API의 장점은 무엇인가요?",
        "SQL과 NoSQL의 차이점은?",
        "Python의 GIL(Global Interpreter Lock)은 무엇인가요?",
        "머신러닝과 딥러닝의 차이는?",
        "Langchain을 프로젝트에서 사용해보신 적이 있으신가요?",
        "JWT(Json Web Token)의 주요 목적은 무엇인가요?"
    ]

    tokenizer, model = load_model(args.model_dir)
    
    total_start = time.time()
    results = []

    for idx, input_text in enumerate(inputs):
        print(f"\n[{idx+1}] 문제 생성 중...")
        start = time.time()
        output = generate_answer(tokenizer, model, args.instruction, input_text, max_new_tokens=args.max_new_tokens)
        elapsed = time.time() - start
        print(f"[{idx+1}] 생성 완료 (소요 시간: {elapsed:.2f}초)")
        print("-" * 20)
        print(output)
        print("-" * 20)
        results.append({
            "input": input_text,
            "output": output,
            "elapsed": elapsed
        })
    
    total_elapsed = time.time() - total_start
    print(f"\n전체 10문제 생성 소요 시간: {total_elapsed:.2f}초")
    print("문제별 소요 시간(초):", [round(r['elapsed'], 2) for r in results])
