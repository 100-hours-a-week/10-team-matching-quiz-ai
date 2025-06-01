import re
from typing import List


def parse_choices(raw_options: str) -> List[str]:
    """
    보기 항목을 A. 보기1 B. 보기2 또는 쉼표 구분 형식에서 추출
    """
    # A. 보기 형식일 경우
    split_by_letter = re.findall(r"[A-D]\.\s*([^A-D]+?)(?=(?:[A-D]\.|$))", raw_options)
    if len(split_by_letter) == 4:
        return [opt.strip() for opt in split_by_letter]

    # 쉼표(,) 형식일 경우
    options = [opt.strip() for opt in raw_options.split(",")]
    return options if len(options) == 4 else []


def parse_response(response_text: str):
    start_index = response_text.find("난이도:")
    if start_index != -1:
        response_text = response_text[start_index:]

    QUESTION_PATTERN = re.compile(
        r"#\s*난이도:\s*(?P<difficulty>하|중|상)\s*"
        r"#\s*문제:\s*(?P<question>.*?)\s*"
        r"#\s*선지:\s*\[(?P<choices>.*?)\]\s*"
        r"#\s*정답 인덱스:\s*(?P<answer_index>[1-4])\s*"
        r"#\s*해설:\s*(?P<explanation>.*?)\s*(?=#\s*난이도:|$)",
        re.DOTALL
    )

    quiz_list = []
    matches = QUESTION_PATTERN.findall(response_text)  # <-- 여기 수정!

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
