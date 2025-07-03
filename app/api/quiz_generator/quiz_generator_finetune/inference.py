import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
import time
from math import ceil

def load_model(model_path, base_model="Qwen/Qwen3-8B"):
    # 토크나이저/기반 모델 로딩
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, padding_side="left")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    # LoRA weight 합성
    model = PeftModel.from_pretrained(base, model_path)
    model.eval()
    return tokenizer, model

def batch_generate(tokenizer, model, input_list, instruction="", max_new_tokens=512):
    # instruction은 거의 안써도 되지만, 혹시 쓸 일이 있으면 포함
    prompts = [
        f"{instruction}\n\n{inp}" if instruction else inp
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
    # 문자열 출력 결과
    return [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="훈련된 모델 경로 (ex: ./qwen3_lora_output_20250703_003109/checkpoint-300)")
    parser.add_argument("--instruction", type=str, default="", help="명령/instruction 입력 (거의 사용 X)")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="최대 출력 토큰 수")
    parser.add_argument("--batch_size", type=int, default=1, help="한 번에 생성할 세트 개수")
    args = parser.parse_args()

    # ==== [실전 input 예시 - 여러 세트 한번에!] ====
    input_prompts = [
        """[질문 목록]
REST API란 무엇인가요?
JWT의 목적은?
Docker 이미지와 컨테이너의 차이점은?
[유사 참고 질문 목록]
HTTP와 HTTPS의 차이점은?
API 인증 방식 종류는?
""",
        """[질문 목록]
Kubernetes에서 PV/PVC란?
클라우드에서 오토스케일링이란?
MLOps란?
[유사 참고 질문 목록]
파드와 디플로이먼트의 차이점은?
VM과 컨테이너의 차이?
"""
    ]

    # 필요한 만큼 input_prompts에 세트 추가 가능
    inputs = input_prompts[:args.batch_size]

    tokenizer, model = load_model(args.model_dir)

    total_start = time.time()
    results = batch_generate(
        tokenizer, model, inputs,
        instruction=args.instruction,
        max_new_tokens=args.max_new_tokens
    )

    for idx, (inp, out) in enumerate(zip(inputs, results)):
        print(f"\n[세트 {idx+1}] 생성 결과")
        print("=" * 60)
        print(f"입력 프롬프트:\n{inp.strip()}")
        print("-" * 40)
        print("생성된 퀴즈 세트(원본 출력):\n")
        print(out)
        print("=" * 60)

    print(f"\n전체 소요 시간: {time.time() - total_start:.2f}초")
