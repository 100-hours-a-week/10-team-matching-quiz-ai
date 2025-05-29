import re
from typing import List, Dict
import ast

def build_prompt(history: list, user_id: str = "user001") -> str:
    joined = "\n".join([f"- {q}" for q in history])
    return f"""너는 사용자의 면접에 대한 4지선다형 객관식 퀴즈를 생성해주는 선생님이야.
사용자가 이전에 받았던 면접 질문들을 참고하여 4지선다형 객관식 퀴즈를 만들어줘.

{joined}

- 이 면접 질문들을 바탕으로, 사용자가 받았던 면접 질문들을 참고하여 객관식 퀴즈 10문제를 생성해줘.
- 난이도는 상, 중, 하 이렇게 3가지로 구성해주고 모든 면접질문들을 다 참고하여 골고루 10문제를 생성해줘.
- 하 4문제, 중 3문제, 상 3문제 이렇게 순서대로 생성해줘.
- 모든 문제들은 다 내용이 다르게 만들어줘.
- options를 만들 때는 너무 답이 무엇인지 알게 하기 보다는 난이도에 따라 헷갈리는 option을 포함시켜줘.
- answer_index는 1~4로 구성해주고 question과 explanation은 너무 짧지 않게 면접 질문에 대해 심화 학습을 하는 느낌을 받을 수 있게 만들어줘.

**출력은 아래의 형식을 꼭 지켜줘.**

## 요구사항:
- 난이도: 하 4문제 / 중 3문제 / 상 3문제
- 각 문항은 다음 형식을 따르세요:

난이도: (상/중/하 중 하나)
문제: (개념을 묻는 질문)
선지: [보기1, 보기2, 보기3, 보기4]  (헷갈리는 보기 포함, 너무 쉬운 오답은 피할 것, 꼭 ,로 구분할 것)
정답 인덱스: (1~4 중 하나)
해설: (정답 이유 + 오답과의 차이 간단히 설명)

⚠️ 출력은 위의 형식으로 10개 연속으로 출력하고, 다른 설명은 포함하지 마.
"""


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
