import re
from typing import List


def parse_choices(raw_options: str) -> List[str]:
    # A. ~ B. ~ C. ~ D. ~ 형식인 경우 파싱
    split_by_letter = re.findall(r"[A-D]\.\s*([^A-D]+?)(?=(?:[A-D]\.|$))", raw_options)
    if len(split_by_letter) == 4:
        return [opt.strip() for opt in split_by_letter]
    # 일반 쉼표 구분 시도
    options = [opt.strip() for opt in raw_options.split(",")]
    return options if len(options) == 4 else []


import re

def parse_response(response_text: str):
    pattern = re.compile(
        r"난이도:\s*(.*?)\s*문제:\s*(.*?)\s*선지:\s*\[(.*?)\]\s*정답 인덱스:\s*(\d+)\s*해설:\s*(.*?)(?=\n난이도:|\Z)",
        re.DOTALL
    )
    
    quiz_list = []
    matches = re.findall(pattern, response_text)
    
    for match in matches:
        difficulty, question, options, answer_index, explanation = match
        option_list = [opt.strip() for opt in options.split(",")]

        if len(option_list) != 4:
            print("보기 항목 수가 4개가 아님:", option_list)
            continue

        quiz_list.append({
            "difficulty": difficulty.strip(),
            "question": question.strip(),
            "options": option_list,
            "answer_index": int(answer_index),
            "explanation": explanation.strip()
        })

    return quiz_list
