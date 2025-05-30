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


def parse_response(response_text: str) -> list:
    quiz_list = []
    pattern = re.compile(
        r"난이도: (.*?)\n문제: (.*?)\n선지: \[(.*?)\]\n정답 인덱스: (\d+)\n해설: (.*?)(?=\n난이도:|\Z)",
        re.DOTALL
    )
    for match in re.finditer(pattern, response_text):
        print("⚠️ match.groups():", match.groups())

        if len(match.groups()) != 5:
            print("❌ 예상한 5개 그룹이 아닙니다. 스킵합니다. →", match.groups())
            continue

        difficulty, question, options, answer_index, explanation = match.groups()

        # 🔽 문자열 옵션 안전하게 파싱
        options = parse_choices(options)
        if len(options) != 4:
            print("❌ 보기 4개가 아님:", options)
            continue


        quiz_list.append({
            "difficulty": difficulty.strip(),
            "question": question.strip(),
            "options": options,
            "answer_index": int(answer_index), 
            "explanation": explanation.strip().strip("---").strip()
        })

    return quiz_list
