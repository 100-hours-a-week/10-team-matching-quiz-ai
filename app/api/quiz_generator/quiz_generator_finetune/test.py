import json
import re

file_path = "quiz_finetune_ft.jsonl"   # 기존 파일 경로
output_path = file_path.replace(".jsonl", "_input.jsonl")

instruction = "질문 목록을 기반으로 유사 참고 질문 목록을 참고해서 객관식 퀴즈를 난이도 하 4개, 중 3개, 상 3개 순서대로 10개만 생성해줘 모든 문제는 서로 중복되면 안돼"

def split_questions(text):
    """입력 텍스트에서 [질문 목록], [유사 참고 질문 목록] 분리"""
    q_match = re.search(r"\[질문 목록\]\n(.*?)(\[유사 참고 질문 목록\]\n|$)", text, re.DOTALL)
    r_match = re.search(r"\[유사 참고 질문 목록\]\n(.*)", text, re.DOTALL)
    qs = q_match.group(1).strip() if q_match else ""
    rs = r_match.group(1).strip() if r_match else ""
    return qs, rs

lines = []
with open(file_path, "r", encoding="utf-8") as fin:
    for line in fin:
        obj = json.loads(line)
        # 기존 input에서 질문/유사질문 추출 (혹은 output을 바탕으로 자동 추출해도 됨)
        # 여기서는 기존 input을 최대한 살리는 예시 (변환 필요하면 맞게 수정!)
        qs, rs = split_questions(obj["input"])
        if not qs:  # fallback: 개행으로 나눈 첫번째 블록
            parts = obj["input"].split('\n\n')
            qs = parts[0].strip() if parts else ""
        # 최종 input 생성
        new_input = f"{instruction}\n[질문 목록]\n{qs}\n[유사 참고 질문 목록]\n{rs if rs else ''}"
        new_obj = {
            "input": new_input,
            "output": obj["output"]
        }
        lines.append(new_obj)

with open(output_path, "w", encoding="utf-8") as fout:
    for obj in lines:
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"✅ input만 변환 완료! ({len(lines)}개) → {output_path}")
