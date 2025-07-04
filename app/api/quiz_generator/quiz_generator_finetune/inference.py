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

def batch_generate(tokenizer, model, input_list, instruction="", max_new_tokens=2048):
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
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="최대 출력 토큰 수")
    parser.add_argument("--batch_size", type=int, default=1, help="한 번에 생성할 세트 개수")
    args = parser.parse_args()

    # ==== [실전 input 예시 - 여러 세트 한번에!] ====
    input_prompts = [
        """
        질문 목록을 기반으로 유사 참고 질문 목록을 참고해서 객관식 퀴즈를 난이도 하 4개, 중 3개, 상 3개 순서대로 10개만 생성해줘 모든 문제는 서로 중복되면 안돼.
        [질문 목록]
        REST API란 무엇인가요?
        머신러닝과 딥러닝의 차이점은 무엇인가요?
        Transformer는 언제 사용하나요?
        [유사 참고 질문 목록]
        머신러닝은 데이터를 기반으로 패턴을 학습하는 알고리즘 전체를 의미하며, 딥러닝은 인공신경망(특히 다층 구조)을 활용하는 머신러닝의 한 분야이다.
        Transformer는 어텐션 메커니즘을 활용하여 입력 시퀀스 내 단어들 간의 관계를 효율적으로 학습할 수 있도록 설계된 딥러닝 모델 구조이다.
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
        print("생성된 퀴즈 세트(원본 출력):\n")
        print(out)
        print("=" * 60)

    print(f"\n전체 소요 시간: {time.time() - total_start:.2f}초")
