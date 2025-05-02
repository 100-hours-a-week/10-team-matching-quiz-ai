import re
from typing import List


def parse_questions(text: str) -> List[str]:
    """
    주어진 텍스트에서 "질문 N. " 형식의 질문 목록을 추출합니다.

    Args:
        text: 질문이 포함된 문자열

    Returns:
        추출된 질문 문자열 리스트
    """
    # "질문 N. " 형식에 더 정확하게 맞는 패턴
    # 각 질문은 다음 질문 시작 전까지 또는 문자열 끝까지 매칭 (DOTALL 플래그로 개행 포함)
    pattern = re.compile(
        r'^질문\s+\d+\.\s*(.+?)(?=\n질문\s+\d+\.|\Z)', re.MULTILINE | re.DOTALL)

    questions = []
    for m in pattern.finditer(text):
        # group(1)은 질문 내용만 캡처
        q = m.group(1).strip()
        # '**' 같은 마크다운 제거 (필요시)
        q = q.replace('**', '').strip()

        # 빈 문자열이거나 플레이스홀더("...")인 경우 제외
        if not q or q == "...":
            continue
        questions.append(q)
    return questions
