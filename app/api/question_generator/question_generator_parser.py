import re
import logging
import unicodedata
from typing import List, Optional, Pattern, Tuple, Dict, Set

logger = logging.getLogger(__name__)

# 괄호와 따옴표 관련 상수
_BRACKETS: Dict[str, str] = {'(': ')', '[': ']', '{': '}'}
_QUOTES: Set[str] = {'"', "'", '"', '"', ''', '''}


def fix_balanced_text(text: str) -> str:
    """텍스트의 괄호와 따옴표 균형을 맞춤"""
    # 괄호 균형 맞추기
    stack: List[str] = []
    for ch in text:
        if ch in _BRACKETS:
            stack.append(ch)
        elif ch in _BRACKETS.values() and stack and ch == _BRACKETS[stack[-1]]:
            stack.pop()

    # 닫는 괄호 추가
    if stack:
        insert_at = text.find('\n', len(text) - 1)
        if insert_at == -1:
            insert_at = len(text)
        text = text[:insert_at] + ''.join(_BRACKETS[b]
                                          for b in reversed(stack)) + text[insert_at:]

    # 따옴표 균형 맞추기
    for q in _QUOTES:
        if text.count(q) % 2:
            text += q

    return text


def normalize_text(text: str) -> str:
    """텍스트 정규화: 공백 처리, 유니코드 정규화, 문장 종결 보장"""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'\s+', ' ', text).strip()

    # 문장 끝에 마침표 추가 (이미 마침표, 물음표, 느낌표 등이 없는 경우)
    if text and not re.search(r'[.!?…""''"\')\]\}]$', text):
        text += '.'
    return text


def _strip_markdown(s: str) -> str:
    """마크다운 서식 제거"""
    # 링크 처리: [텍스트](URL) -> 텍스트
    s = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', s)
    # 취소선 처리: ~~텍스트~~ -> 텍스트
    s = re.sub(r'~~([^~]+)~~', r'\1', s)
    # 기본 마크다운 제거 (굵게, 기울임, 코드, 제목)
    s = re.sub(r'(\*\*|\*|`|#+)', '', s)
    return s


# 질문 목록 추출 패턴
_NEXT_MARKER = r'\n\s*(?:-?\s*(?:질문|문제|Question)\s+\d+\.?|Q\s*\d+\.?|\d+\.\s+|\d+\)\s+)'

_DEFAULT_PATTERNS = (
    # 질문/문제 n. 형식
    re.compile(
        rf'^(?:질문|문제)\s+\d+\.?\s*(.+?)(?={_NEXT_MARKER}|\Z)', re.M | re.S),
    # Q n. 형식
    re.compile(rf'^Q\s*\d+\.?\s*(.+?)(?={_NEXT_MARKER}|\Z)', re.M | re.S),
    # Question n. 형식
    re.compile(
        rf'^Question\s+\d+\.?\s*(.+?)(?={_NEXT_MARKER}|\Z)', re.M | re.S),
    # n. 형식
    re.compile(rf'^\d+\.\s+(.+?)(?={_NEXT_MARKER}|\Z)', re.M | re.S),
    # n) 형식
    re.compile(rf'^\d+\)\s+(.+?)(?={_NEXT_MARKER}|\Z)', re.M | re.S),
    # - 질문/문제 n. 형식
    re.compile(
        rf'^\s*-\s*(?:질문|문제|Question)?\s*\d*\.?\s*(.+?)(?={_NEXT_MARKER}|\Z)', re.M | re.S),
    # 불릿 리스트 형식
    re.compile(r'^\s*[-*•]\s+(.+?)(?=\n\s*[-*•]|\Z)', re.M | re.S),
)


def parse_questions(
    text: str,
    patterns: Optional[List[Pattern[str]]] = None,
    strip_markdown: bool = True,
) -> List[str]:
    """텍스트에서 질문 목록 추출

    한국어/영어 다양한 리스트 형식을 지원합니다.
    """
    # 텍스트 정리
    cleaned = text.replace("\n...\n[출력 형식 예시]",
                           "").strip().removesuffix("\nassistant")

    # 너무 짧은 텍스트 검증
    if len(cleaned) < 5:
        raise ValueError("질문 추출하기에 텍스트가 너무 짧습니다")

    # 패턴 목록 구성
    pat_list = list(_DEFAULT_PATTERNS)
    if patterns:
        pat_list.extend(patterns)

    # 모든 패턴으로 매칭 시도
    spans: List[Tuple[int, int, str]] = []
    for pat in pat_list:
        spans.extend((m.start(1), m.end(1), m.group(1).strip())
                     for m in pat.finditer(cleaned))

    # 패턴 매칭 실패 시 폴백 처리
    if not spans:
        logger.info("패턴 매칭 실패, 휴리스틱 방식 사용")
        maybe = [
            normalize_text(fix_balanced_text(line.strip()))
            for line in cleaned.split('\n')
            if len(line) > 10 and ('?' in line or re.search(r'질문|query|ask', line, re.I))
        ]
        return maybe if maybe else [normalize_text(fix_balanced_text(cleaned))]

    # 위치순 정렬 (같은 시작점이면 더 긴 것 우선)
    spans.sort(key=lambda t: (t[0], -(t[1] - t[0])))

    # 질문 목록 구성
    questions: List[str] = []
    processed: set[str] = set()
    last_end = -1

    for start, end, raw in spans:
        # 중복 영역 건너뛰기
        if start < last_end:
            continue

        # 마크다운 제거 및 검증
        q = _strip_markdown(raw) if strip_markdown else raw
        if len(q) < 10 or q in processed or q in {"...", ""}:
            continue

        # 정규화 및 저장
        q = normalize_text(fix_balanced_text(q))
        questions.append(q)
        processed.add(q)
        last_end = end

    return questions
