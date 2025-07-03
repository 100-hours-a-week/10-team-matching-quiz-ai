import random
import json

def make_dummy_sample(idx):
    # 1~10개 랜덤 질문, 0~질문개수까지 랜덤 참고질문
    num_questions = random.randint(1, 10)
    num_related = random.randint(0, num_questions)

    question_list = [f"질문 {i+1}" for i in range(num_questions)]
    related_list = [f"참고질문 {i+1}" for i in range(num_related)]

    # input: 지시문 + 질문/참고질문
    input_str = (
        "질문 목록을 기반으로 유사 참고 질문 목록을 참고해서 객관식 퀴즈를 난이도 하 4개, 중 3개, 상 3개 순서대로 10개만 생성해줘 모든 문제는 서로 중복되면 안돼\n"
        + "[질문 목록]\n" + "\n".join(question_list) +
        "\n[유사 참고 질문 목록]\n" + "\n".join(related_list)
    )

    # output: 하4-중3-상3 더미 문제
    output_list = []
    for diff, cnt in zip(["하", "중", "상"], [4, 3, 3]):
        for n in range(cnt):
            output_list.append({
                "difficulty": diff,
                "question": f"{diff} 더미 문제 {n+1} (예시)",
                "options": [f"{diff} 선지 {i+1}" for i in range(4)],
                "answer_index": random.randint(1, 4),
                "explanation": f"{diff} 더미 해설 {n+1}"
            })

    # jsonl 한 줄(문자열)
    return {
        "input": input_str,
        "output": output_list
    }

# 1500개 생성
out_path = "quiz_dummy_ft.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for i in range(1500):
        sample = make_dummy_sample(i)
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

print(f"생성 완료: {out_path} (총 1500개)")
