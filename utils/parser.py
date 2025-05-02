import re
import logging
from typing import List, Optional, Pattern

logger = logging.getLogger(__name__)


def parse_questions(text: str,
                    patterns: Optional[List[Pattern]] = None,
                    strip_markdown: bool = True) -> List[str]:
    """
    주어진 텍스트에서 질문 목록을 추출합니다.

    기본적으로 "질문 N." 형식을 찾지만, 추가 패턴을 전달하여 다른 형식도 인식할 수 있습니다.

    Args:
        text: 질문이 포함된 문자열
        patterns: 사용할 추가 정규표현식 패턴 목록 (기본 패턴은 항상 사용됨)
        strip_markdown: 마크다운 서식 제거 여부

    Returns:
        추출된 질문 문자열 리스트
    """
    # 기본 패턴: "질문 N. " 형식
    default_pattern = re.compile(
        r'^질문\s+\d+\.\s*(.+?)(?=\n질문\s+\d+\.|\Z)', re.MULTILINE | re.DOTALL)

    # 추가 패턴들 (필요한 경우)
    all_patterns = [default_pattern]
    if patterns:
        all_patterns.extend(patterns)

    questions = []
    text_length = len(text)

    # 안전 조치: 텍스트가 너무 짧으면 처리하지 않음
    if text_length < 5:
        logger.warning(f"Text too short to parse questions: '{text}'")
        return questions

    # 텍스트가 너무 길면 로깅
    if text_length > 10000:
        logger.info(
            f"Processing large text ({text_length} chars) for question extraction")

    # 모든 패턴에 대해 매칭 시도
    for pattern in all_patterns:
        for m in pattern.finditer(text):
            # group(1)은 질문 내용만 캡처
            q = m.group(1).strip()

            # 마크다운 제거 (옵션에 따라)
            if strip_markdown:
                # 기본 마크다운 서식 제거
                q = q.replace('**', '').replace('*',
                                                '').replace('`', '').replace('#', '').strip()
                # 추가 마크다운 패턴 제거 (필요시)

            # 빈 문자열이거나 플레이스홀더인 경우 제외
            if not q or q == "..." or len(q) < 3:
                continue

            # 중복 제거 로직
            if q not in questions:
                questions.append(q)

    if not questions:
        logger.debug(f"No questions found in text: '{text[:100]}...'")

    return questions
