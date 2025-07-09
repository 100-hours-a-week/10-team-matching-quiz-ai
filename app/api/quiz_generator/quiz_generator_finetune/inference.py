import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
import time

def load_model(model_path, base_model="Qwen/Qwen3-8B"):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, padding_side="left")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, model_path)
    model.eval()
    return tokenizer, model

def generate_single_quiz(tokenizer, model, prompt, instruction, max_new_tokens=512):
    full_prompt = f"{instruction}\n\n{prompt}"
    tokens = tokenizer(full_prompt, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.7
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="훈련된 모델 경로 (ex: ...)")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="최대 출력 토큰 수")
    args = parser.parse_args()

    instruction = "아래 문장에 대해 객관식 4지선다 퀴즈 한 문제를 생성해줘. 정답 인덱스와 해설도 꼭 포함해줘."

    question_list = [
        "파이썬의 대표적인 특징은 무엇인가요?",
        "딥러닝과 머신러닝의 차이점을 설명하세요.",
        "클라우드 컴퓨팅의 장점은 무엇인가요?",
    ]
    rag_sentences = [
        "파이썬은 간결한 문법과 다양한 라이브러리를 제공하며, 초보자도 쉽게 배울 수 있는 프로그래밍 언어입니다.",
        "딥러닝은 인공신경망을 기반으로 하는 머신러닝의 한 분야이며, 데이터로부터 자동으로 특징을 학습합니다.",
        "클라우드 컴퓨팅은 비용 절감과 확장성, 유연한 리소스 할당이 강점입니다.",
    ]

    print("========== [QUIZ GENERATOR INFERENCE START] ==========")
    print("[INFO] 모델 로딩 중...")
    tokenizer, model = load_model(args.model_dir)

    total_start = time.time()
    results = []
    idx_q = 0
    idx_r = 0
    mode = 0  # 0: 질문, 1: rag
    used_prompts = set()
    while len(results) < 10:
        if mode == 0 and idx_q < len(question_list):
            prompt = question_list[idx_q]
            idx_q += 1
        elif mode == 1 and idx_r < len(rag_sentences):
            prompt = rag_sentences[idx_r]
            idx_r += 1
        else:
            if idx_q < len(question_list):
                prompt = question_list[idx_q]
                idx_q += 1
            elif idx_r < len(rag_sentences):
                prompt = rag_sentences[idx_r]
                idx_r += 1
            else:
                break
        mode = 1 - mode

        if prompt in used_prompts:
            continue
        used_prompts.add(prompt)

        start = time.time()
        quiz = generate_single_quiz(
            tokenizer, model, prompt,
            instruction=instruction,
            max_new_tokens=args.max_new_tokens
        )
        elapsed = time.time() - start
        print(f"[{len(results)+1}] {prompt}\n---\n{quiz}\n[소요 시간: {elapsed:.2f}초]\n")
        results.append(quiz)

    print(f"\n[INFO] 전체 소요 시간: {time.time() - total_start:.2f}초")
    print("========== [QUIZ GENERATOR INFERENCE END] ==========")
    print("최종 10개 퀴즈만 파싱 결과:")
    for idx, quiz in enumerate(results):
        print(f"\n--- Quiz {idx+1} ---\n{quiz}")
