import re
from typing import List

def parse_choices(raw_options: str) -> List[str]:
    # A. ~ B. ~ 형식 처리
    split_by_letter = re.findall(r"[A-D]\.\s*([^A-D]+?)(?=(?:[A-D]\.|$))", raw_options)
    if len(split_by_letter) == 4:
        return [opt.strip() for opt in split_by_letter]
    # 쉼표 기준 처리
    options = [opt.strip() for opt in raw_options.split(",")]
    return options if len(options) == 4 else []

def parse_response(response_text: str):
    # 필요 없는 머리말 제거
    start_index = response_text.find("난이도:")
    if start_index != -1:
        response_text = response_text[start_index:]

    pattern = re.compile(
        r"난이도:\s*(.*?)\s*문제:\s*(.*?)\s*선지:\s*\[(.*?)\]\s*정답 인덱스:\s*(\d+)\s*해설:\s*(.*?)(?=\n난이도:|\Z)",
        re.DOTALL
    )
    
    quiz_list = []
    matches = re.findall(pattern, response_text)
    valid_difficulties = {"상", "중", "하"}
    
    for match in matches:
        difficulty, question, options, answer_index, explanation = match
        option_list = parse_choices(options)

        if len(option_list) != 4:
            print("보기 항목 수가 4개가 아님:", option_list)
            continue
        if difficulty.strip() not in valid_difficulties:
            print("난이도 필드가 잘못됨:", difficulty)
            continue

        quiz_list.append({
            "difficulty": difficulty.strip(),
            "question": question.strip(),
            "options": option_list,
            "answer_index": int(answer_index),
            "explanation": explanation.strip()
        })

    return quiz_list
