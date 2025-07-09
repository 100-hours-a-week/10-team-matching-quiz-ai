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

def batch_generate_quizzes(tokenizer, model, prompts, instruction="", max_new_tokens=512):
    prompts_with_instruction = [
        f"{instruction}\n\n{p}" if instruction else p for p in prompts
    ]
    tokens = tokenizer(prompts_with_instruction, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        start = time.time()
        outputs = model.generate(
            **tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.7
        )
        elapsed = time.time() - start
    decoded = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
    return decoded, elapsed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="훈련된 모델 경로 (ex: ...)")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="최대 출력 토큰 수")
    args = parser.parse_args()

    # 지시문 (포맷 예시 포함)
    instruction = (
        "아래 문장에 대해 객관식 4지선다 퀴즈 한 문제를 생성해줘. "
        "문제, 선지, 정답 인덱스(1~4), 해설을 꼭 포함해서 아래 포맷 예시를 따르세요.\n"
        "난이도: 하\n"
        "문제: ...\n"
        "선지: [ ... ]\n"
        "정답 인덱스: ...\n"
        "해설: ...\n"
    )

    question_list = [
        "파이썬의 대표적인 특징은 무엇인가요?",
        "딥러닝과 머신러닝의 차이점을 설명하세요.",
        "클라우드 컴퓨팅의 장점은 무엇인가요?",
        "RAG의 개념을 설명해 주세요.",
        "SQL과 NoSQL의 차이점을 말해 주세요."
    ]

    rag_sentences = [
        "파이썬은 간결한 문법과 다양한 라이브러리를 제공하여, 초보자도 쉽게 배울 수 있는 언어입니다.",
        "딥러닝은 대량의 데이터를 처리하고, 자동으로 특징을 추출하는 머신러닝의 하위 분야입니다.",
        "클라우드 컴퓨팅은 유연한 리소스 할당과 비용 절감, 무한한 확장성을 제공합니다.",
        "RAG는 외부 지식 검색 결과를 활용해 정답을 생성하는 생성형 AI 구조입니다.",
        "SQL은 관계형 데이터베이스, NoSQL은 비정형 데이터에 최적화된 데이터베이스입니다."
    ]

    print("========== [QUIZ GENERATOR INFERENCE START] ==========")
    print("[INFO] 모델 로딩 중...")
    tokenizer, model = load_model(args.model_dir)

    total_start = time.time()
    # --- question, rag 번갈아가며 최대 10개 모으기 ---
    prompts = []
    i = 0
    while len(prompts) < 10:
        if i < len(question_list):
            prompts.append(question_list[i])
        if len(prompts) < 10 and i < len(rag_sentences):
            prompts.append(rag_sentences[i])
        i += 1
        if i >= max(len(question_list), len(rag_sentences)):
            break

    # batch inference
    results, batch_time = batch_generate_quizzes(
        tokenizer, model,
        prompts=prompts,
        instruction=instruction,
        max_new_tokens=args.max_new_tokens
    )
    print(f"\n[INFO] 전체 {len(results)}개 퀴즈 생성에 걸린 시간: {batch_time:.2f}초")
    print("========== [QUIZ GENERATOR INFERENCE END] ==========")

    print("\n========== [최종 생성된 퀴즈] ==========")
    for idx, quiz in enumerate(results, 1):
        print(f"\n--- Quiz {idx} ---\n{quiz}")

    print("\n========================================")
