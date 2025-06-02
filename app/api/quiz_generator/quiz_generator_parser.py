import re
from typing import List, Dict
import random


def parse_choices(raw_options: str) -> List[str]:
    split_by_letter = re.findall(r"[A-D]\.\s*([^A-D]+?)(?=(?:[A-D]\.|$))", raw_options)
    if len(split_by_letter) == 4:
        return [opt.strip() for opt in split_by_letter]

    options = re.split(r"[,\n]", raw_options)
    options = [opt.strip() for opt in options if opt.strip()]
    return options if len(options) == 4 else []

# Quiz 형식 검증 함수
def is_valid_quiz_item(item: Dict) -> bool:
    if not item.get("question"): return False
    if not isinstance(item.get("options"), list): return False
    if len(item["options"]) != 4: return False
    if not isinstance(item.get("answer_index"), int): return False
    if not (1 <= item["answer_index"] <= 4): return False
    if not item.get("explanation"): return False
    return True

# 생성된 Quiz 중 10문제 선별
def filter_and_select_quizzes(quizzes: List[Dict]) -> List[Dict]:
    # 형식 검증
    valid_quizzes = [q for q in quizzes if is_valid_quiz_item(q)]

    # 난이도별 분리
    easy = [q for q in valid_quizzes if q["difficulty"] == "하"]
    medium = [q for q in valid_quizzes if q["difficulty"] == "중"]
    hard = [q for q in valid_quizzes if q["difficulty"] == "상"]

    # 조건에 맞게 개수만큼 추출 (순서 고정)
    selected = (
        easy[:4] +
        medium[:3] +
        hard[:3]
    )

    # 문제 번호 붙이기
    for i, q in enumerate(selected, 1):
        q["number"] = i

    return selected


def parse_response(response_text: str):
    start_index = response_text.find("난이도:")
    if start_index != -1:
        response_text = response_text[start_index:]

    # 프롬프트 설명 제거
    first_quiz_start = response_text.find("난이도:")
    if first_quiz_start != -1:
        response_text = response_text[first_quiz_start:]

    # Quiz 정규식 추출
    QUESTION_PATTERN = re.compile(
        r"#\s*?난이도:\s*(?P<difficulty>하|중|상)\s*"
        r"#\s*?문제:\s*(?P<question>.*?)\s*"
        r"#\s*?선지:\s*\[(?P<choices>.*?)\]\s*"
        r"#\s*?정답\s*인덱스:\s*(?P<answer_index>[1-4])\s*"
        r"#\s*?해설:\s*(?P<explanation>.*?)(?=\n#\s*?난이도:|\Z)",
        re.DOTALL
    )


    quiz_list = []
    matches = QUESTION_PATTERN.findall(response_text) 
    valid_difficulties = {"상", "중", "하"}

    for i, match in enumerate(matches, 1):
        difficulty, question, options, answer_index, explanation = match
        option_list = parse_choices(options)

        if len(option_list) != 4:
            print(f"[{i}] 1. 보기 항목 수가 4개가 아님:", option_list)
            continue
        if difficulty.strip() not in valid_difficulties:
            print(f"[{i}] 2. 난이도 필드가 잘못됨:", difficulty)
            continue
        if not answer_index.isdigit() or not (1 <= int(answer_index) <= 4):
            print(f"[{i}] 3. 정답 인덱스가 유효하지 않음:", answer_index)
            continue

        quiz_list.append({
            "difficulty": difficulty.strip(),
            "question": question.strip(),
            "options": option_list,
            "answer_index": int(answer_index),  # 1부터 시작
            "explanation": explanation.strip()
        })

    return quiz_list
