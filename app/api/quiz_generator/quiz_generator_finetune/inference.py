import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
import time
from math import ceil

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

def sanitize_input(input_text):
    keywords = ["사용해보신 적", "경험이 있으신가요", "해보신 경험", "해보셨나요"]
    for kw in keywords:
        if kw in input_text:
            before = input_text.split("을")[0].split("를")[0].replace("프로젝트에서 ", "")
            concept = before.strip()
            return f"{concept}의 주요 개념은 무엇인가요?"
    return input_text

def batch_generate(tokenizer, model, input_list, instruction="", max_new_tokens=512):
    # instruction 적용 및 sanitize 적용
    prompts = [
        f"{instruction}\n\n{sanitize_input(inp)}" if instruction else sanitize_input(inp)
        for inp in input_list
    ]
    tokens = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.7
        )
    return [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="훈련된 모델 경로")
    parser.add_argument("--instruction", type=str, default="", help="명령/instruction 입력")
    parser.add_argument("--max_new_tokens", type=int, default=320, help="최대 출력 토큰 수")
    parser.add_argument("--batch_size", type=int, default=2, help="배치 크기(몇 개씩 동시에 생성)")
    args = parser.parse_args()

    input_pool = [
        "REST API의 장점은 무엇인가요?",
        "SQL과 NoSQL의 차이점은?",
        "Python의 GIL(Global Interpreter Lock)은 무엇인가요?",
        "머신러닝과 딥러닝의 차이는?",
        "Langchain을 프로젝트에서 사용해보신 적이 있으신가요?",
        "JWT(Json Web Token)의 주요 목적은 무엇인가요?",
        "클라우드에서 오토스케일링이란?",
        "파이썬에서 데코레이터란 무엇인가요?",
        "프로젝트를 하면서 rag를 사용해보신 경험이 있으신가요?"
    ]

    num_quiz = 10
    if len(input_pool) < num_quiz:
        inputs = (input_pool * (num_quiz // len(input_pool) + 1))[:num_quiz]
    else:
        inputs = input_pool[:num_quiz]

    tokenizer, model = load_model(args.model_dir)

    total_start = time.time()
    results = []
    BATCH_SIZE = args.batch_size

    print(f"\n== Batch Inference: {BATCH_SIZE}개씩 동시에 생성 ==")
    for batch_idx in range(ceil(len(inputs) / BATCH_SIZE)):
        batch_inputs = inputs[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
        print(f"\n[{batch_idx+1}]번째 배치 ({len(batch_inputs)}문제) 생성 중...")
        start = time.time()
        batch_outputs = batch_generate(
            tokenizer, model, batch_inputs,
            instruction=args.instruction, max_new_tokens=args.max_new_tokens
        )
        elapsed = time.time() - start
        for i, (inp, out) in enumerate(zip(batch_inputs, batch_outputs)):
            print(f"[{batch_idx*BATCH_SIZE+i+1}] 생성 완료 (소요 시간: {elapsed/len(batch_inputs):.2f}초)")
            print("-" * 20)
            print(out)
            print("-" * 20)
            results.append({
                "input": inp,
                "output": out,
                "elapsed": elapsed / len(batch_inputs)
            })

    total_elapsed = time.time() - total_start
    print(f"\n전체 10문제 생성 소요 시간: {total_elapsed:.2f}초")
    print("문제별 소요 시간(초):", [round(r['elapsed'], 2) for r in results])
