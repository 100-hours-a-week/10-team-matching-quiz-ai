import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
import time

def load_model(model_path, base_model="Qwen/Qwen3-8B"):
    print(f"[INFO] 토크나이저/기반 모델({base_model}) 불러오는 중...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, padding_side="left")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    print(f"[INFO] LoRA weight({model_path}) 합성 및 최종 모델 준비 중...")
    model = PeftModel.from_pretrained(base, model_path)
    model.eval()
    print("[INFO] 모델 및 토크나이저 준비 완료!")
    return tokenizer, model

def batch_generate(tokenizer, model, input_list, instruction="", max_new_tokens=1024):
    print(f"[INFO] 총 {len(input_list)}세트 배치 생성 시작!")
    prompts = [
        f"{instruction}\n\n{inp}" if instruction else inp
        for inp in input_list
    ]
    tokens = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        print("[INFO] LLM 추론(generate) 시작...")
        outputs = model.generate(
            **tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.7
        )
    print("[INFO] 생성 완료!")
    return [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="훈련된 모델 경로 (ex: ...)")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="최대 출력 토큰 수")
    args = parser.parse_args()

    # 👇 여기!!! (변수 선언부)
    instruction = (
        "아래 질문들로 총 10개의 객관식 퀴즈를 만들고 난이도별 하 4문제, 중 3문제, 상 3문제로 순서대로 출력하세요."
    )

    print("========== [QUIZ GENERATOR INFERENCE START] ==========")
    print("[INFO] 모델 로딩 중...")
    tokenizer, model = load_model(args.model_dir)

    while True:
        print("\n" + "="*60)
        print("여러 줄(여러 질문)을 한 번에 입력하세요!")
        print("입력이 끝나면 빈 줄을 한 번 입력하면 됩니다. (최대 10개 권장)")
        print("종료하려면 그냥 Enter(빈 줄)만 입력.")
        print("="*60)
        
        batch_inputs = []
        while True:
            line = input()
            if not line.strip():
                break
            batch_inputs.append(line.strip())
        if not batch_inputs:
            print("[INFO] 입력이 없어 종료합니다.")
            break

        total_start = time.time()
        # 👇 batch_generate에 instruction 인자로 넣는 부분!
        results = batch_generate(
            tokenizer, model, batch_inputs,
            instruction=instruction,  # 여기도!
            max_new_tokens=args.max_new_tokens
        )

        for idx, (inp, out) in enumerate(zip(batch_inputs, results)):
            print(f"\n[세트 {idx+1}] 질문: {inp}")
            print("=" * 60)
            print("생성된 퀴즈 세트(원본 출력):\n")
            print(out)
            print("=" * 60)
        print(f"[INFO] 소요 시간: {time.time() - total_start:.2f}초")

    print("========== [QUIZ GENERATOR INFERENCE END] ==========")
