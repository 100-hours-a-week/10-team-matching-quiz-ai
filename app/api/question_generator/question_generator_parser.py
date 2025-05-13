import re
import unicodedata
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not re.search(r"[\.\!\?…]$", text):
        text += "."
    return text


def _strip_md(s: str) -> str:
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    s = re.sub(r"~~([^~]+)~~", r"\1", s)
    s = re.sub(r"(\*\*|\*|_|`|#+)", "", s)
    return s.strip()


def korean_char_length(text: str) -> int:
    return len(text)


def parse_questions(
    text: str,
    strip_md: bool = True,
) -> List[str]:
    parsed_questions: List[str] = []
    lines = text.split("\n")

    for line in lines:
        line = line.strip()
        match = re.match(
            r"^(?:질문|문제|꼬리질문|꼬리\s*질문|Question|Q)\s*\d+\s*\.\s*(.+)", line, re.IGNORECASE
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
