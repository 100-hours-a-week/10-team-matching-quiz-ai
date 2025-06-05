import re
import unicodedata
import logging
from typing import List, Optional
from app.api.question_generator.question_generator_config import PARSER_CONFIG

logger = logging.getLogger(__name__)

# 설정에서 최대 질문 길이 가져오기
MAX_QUESTION_LENGTH = PARSER_CONFIG["max_question_length"]

# "질문 1. ..." 또는 "Q1. ..." 형식의 질문 문장을 추출하는 정규식
QUESTION_PATTERN = re.compile(
    r"^(?:질문|문제|꼬리질문|꼬리\s*질문|Question|Q)\s*\d+\s*\.\s*(.+)", re.IGNORECASE
)

# 문장이 마침표, 느낌표, 물음표, … 등으로 끝나는지 확인하는 정규식
ENDING_PUNCT_PATTERN = re.compile(r"[\.\!\?…]$")

# 빈 줄(공백 문자만 있는 줄)을 판별하는 정규식
EMPTY_LINE_PATTERN = re.compile(r"^\s*$")

# 마크다운 형식의 링크를 감지하는 정규식: [텍스트](URL)
MD_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\([^)]+\)")

# 마크다운 취소선 형식(~~내용~~)을 감지하는 정규식
MD_STRIKETHROUGH_PATTERN = re.compile(r"~~([^~]+)~~")

# 마크다운 서식 기호(굵게, 기울임, 코드, 헤더 등)를 감지하는 정규식
MD_FORMATTING_PATTERN = re.compile(r"(\*\*|\*|_|`|#+)")


def _norm(text: str) -> str:
    """텍스트를 정규화"""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not ENDING_PUNCT_PATTERN.search(text):
        text += "."
    return text


def _strip_md(s: str) -> str:
    """마크다운 서식을 제거"""
    s = MD_LINK_PATTERN.sub(r"\1", s)
    s = MD_STRIKETHROUGH_PATTERN.sub(r"\1", s)
    s = MD_FORMATTING_PATTERN.sub("", s)
    return s.strip()


def parse_questions(text: str, strip_md: bool = True) -> List[str]:
    """정제된 텍스트에서 질문을 추출"""
    parsed_questions: List[str] = []
    lines = text.split("\n")

    for line in lines:
        line = line.strip()
        match = QUESTION_PATTERN.match(line)

        if match:
            question_text = match.group(1).strip()

            if strip_md:
                question_text = _strip_md(question_text)

            question_text = _norm(question_text)

            if len(question_text) <= MAX_QUESTION_LENGTH:
                parsed_questions.append(question_text)
            else:
                logger.warning(
                    f"질문이 {MAX_QUESTION_LENGTH}자를 초과하여 제외됩니다 "
                    f"(길이: {len(question_text)}): {question_text}"
                )
        elif line and not EMPTY_LINE_PATTERN.match(line):
            logger.debug(f"질문 형식에 맞지 않는 라인입니다: {line}")

    return parsed_questions
