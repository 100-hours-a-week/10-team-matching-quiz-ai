import re
import unicodedata
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def _norm(text: str) -> str:
    """텍스트를 정규화하고, 문장 끝에 마침표를 추가합니다."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    # 이미 구두점으로 끝나는 경우 추가하지 않음 (물음표, 느낌표, 마침표, 말줄임표 등)
    if not re.search(r"[\.\!\?…]$", text):
        text += "."
    return text


def _strip_md(s: str) -> str:
    """간단한 마크다운 구문을 제거합니다."""
    # 링크 제거: [text](url) -> text
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    # 취소선 제거: ~~text~~ -> text
    s = re.sub(r"~~([^~]+)~~", r"\1", s)
    # 볼드, 이탤릭, 코드, 헤더 등 제거
    s = re.sub(r"(\*\*|\*|_|`|#+)", "", s)
    return s.strip()


def korean_char_length(text: str) -> int:
    """텍스트의 글자 수를 반환합니다 (구두점 포함)."""
    return len(text)


def parse_questions(
    text: str,
    strip_md: bool = True,
) -> List[str]:
    """
    "질문 n. ", "문제 n. ", "꼬리질문 n. ", "Question n. " 등 형식의 텍스트에서
    질문 내용만 추출하여 리스트로 반환합니다. 각 질문은 100자 이내로 제한됩니다.
    """
    parsed_questions: List[str] = []
    lines = text.split("\n")

    for line in lines:
        line = line.strip()
        # 다양한 질문 형식 패턴을 찾고, 그 뒤의 내용을 캡처합니다.
        # (?:질문|문제|꼬리질문|꼬리\s*질문|Question) - 다양한 질문 접두어 매칭
        # \s*\d+\s*\. - 숫자와 점, 그리고 그 주변 공백 유연하게 처리
        # \s*(.+) - 질문 내용 캡처
        match = re.match(
            r"^(?:질문|문제|꼬리질문|꼬리\s*질문|Question)\s*\d+\s*\.\s*(.+)", line, re.IGNORECASE
        )
        if match:
            question_text = match.group(1).strip()

            if strip_md:
                question_text = _strip_md(question_text)

            question_text = _norm(question_text)

            if korean_char_length(question_text) <= 100:
                parsed_questions.append(question_text)
            else:
                logger.warning(
                    f"질문이 100자를 초과하여 제외됩니다 (길이: {korean_char_length(question_text)}): {question_text}"
                )
        elif line and not re.match(r"^\s*$", line):
            logger.debug(f"질문 형식에 맞지 않는 라인입니다: {line}")

    return parsed_questions
