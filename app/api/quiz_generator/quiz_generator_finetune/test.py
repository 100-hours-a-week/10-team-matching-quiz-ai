import json

# 원본 jsonl 경로
input_path = "quiz_finetune.jsonl"         # 기존 파일명
output_path = "quiz_finetune_unsloth.jsonl"  # 새 파일명

# instruction과 포맷 예시
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
    for line in fin:
        row = json.loads(line)
        # 기존 구조: {"input": "...", "output": {...}}
        # output 구조 예시: {"difficulty": "...", "question": "...", "options": [...], "answer_index": 1, "explanation": "..."}
        data = row["output"]
        # 원하는 output 텍스트 포맷으로 변환
        output_text = (
            f"난이도: {data['difficulty']}\n"
            f"문제: {data['question']}\n"
            f"선지: {json.dumps(data['options'], ensure_ascii=False)}\n"
            f"정답 인덱스: {data['output']['answer_index']}\n"
            f"해설: {data['explanation']}"
        )
        # instruction + 포맷예시 + 실제 문장
        prompt = (
            f"{instruction}\n문장: {row['input']}"
        )
        json.dump({"input": prompt, "output": output_text}, fout, ensure_ascii=False)
        fout.write("\n")
