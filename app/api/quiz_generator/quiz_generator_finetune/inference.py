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

def generate_single_quiz(tokenizer, model, prompt, max_new_tokens=512):
    tokens = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
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

    # 질문 리스트와 RAG 문장 리스트 (예시)
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
    used_prompts = set()

    q_len = len(question_list)
    r_len = len(rag_sentences)
    i = 0

    # 최대 10개까지 번갈아 생성(질문→rag→질문→...)
    while len(results) < 10:
        if i % 2 == 0:
            prompt = question_list[(i // 2) % q_len]
        else:
            prompt = rag_sentences[(i // 2) % r_len]

        if prompt in used_prompts:
            i += 1
            continue
        used_prompts.add(prompt)

        start = time.time()
        quiz = generate_single_quiz(
            tokenizer, model, prompt,
            max_new_tokens=args.max_new_tokens
        )
        elapsed = time.time() - start
        print(f"[{len(results)+1}] {prompt}\n---\n{quiz}\n[소요 시간: {elapsed:.2f}초]\n")
        results.append({"prompt": prompt, "quiz": quiz, "elapsed": elapsed})
        i += 1

    print(f"\n[INFO] 전체 소요 시간: {time.time() - total_start:.2f}초")
    print("========== [QUIZ GENERATOR INFERENCE END] ==========")

    # 👇 최종 생성된 퀴즈 깔끔하게 출력
    print("\n========== [최종 10개 퀴즈 결과만 정리] ==========")
    for idx, item in enumerate(results, 1):
        print(f"\n--- Quiz {idx} ---")
        print(f"질문/문장: {item['prompt']}")
        print(f"생성 퀴즈:\n{item['quiz']}")
        print(f"생성 시간: {item['elapsed']:.2f}초")

    print("\n==============================================")
