import json

input_path = "quiz_finetune.jsonl"
output_path = "quiz_finetune_unsloth.jsonl"

instruction = (
    "아래 문장에 대해 객관식 4지선다 퀴즈 한 문제를 생성해줘. "
    "문제, 선지, 정답 인덱스(1~4), 해설을 꼭 포함해서 아래 포맷 예시를 따르세요.\n\n"
    "[포맷 예시]\n"
    "난이도: 하\n"
    "문제: ...\n"
    "선지: [ ... ]\n"
    "정답 인덱스: ...\n"
    "해설: ...\n"
)

with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
    for i, line in enumerate(fin):
        row = json.loads(line)
        data = row["output"]
        # answer_index 없는 경우 건너뜀 + 알림
        if "answer_index" not in data:
            print(f"[SKIP] {i+1}번째 줄에 'answer_index' 없음!")
            continue
        output_text = (
            f"난이도: {data['difficulty']}\n"
            f"문제: {data['question']}\n"
            f"선지: {json.dumps(data['options'], ensure_ascii=False)}\n"
            f"정답 인덱스: {data['answer_index']}\n"
            f"해설: {data['explanation']}"
        )
        prompt = (
            f"{instruction}\n문장: {row['input']}"
        )
        json.dump({"input": prompt, "output": output_text}, fout, ensure_ascii=False)
        fout.write("\n")
