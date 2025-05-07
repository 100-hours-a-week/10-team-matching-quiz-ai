import re
import unicodedata
import logging
from typing import List, Pattern, Tuple, Dict, Set, Optional

logger = logging.getLogger(__name__)

# ── 상수 ─────────────────────────────────────────────────────────
_BRACKETS: Dict[str, str] = {'(': ')', '[': ']', '{': '}'}
_QUOTES: Set[str] = {'"', "'", '“', '”', '‘', '’'}

# ── 텍스트 유틸 ──────────────────────────────────────────────────


def _fix(text: str) -> str:
    """열린 괄호·따옴표가 있으면 자동으로 닫아 준다"""
    stk: List[str] = []
    for ch in text:
        if ch in _BRACKETS:
            stk.append(ch)
        elif stk and ch == _BRACKETS.get(stk[-1], ''):
            stk.pop()
    if stk:
        pos = text.rfind('\n')
        pos = pos + 1 if pos != -1 else len(text)
        text = text[:pos] + ''.join(_BRACKETS[b]
                                    for b in reversed(stk)) + text[pos:]
    for q in _QUOTES:
        if text.count(q) % 2 != 0:
            text += q
    return text


def _norm(text: str) -> str:
    """NFC 정규화 + 공백 정리 + 끝맺음 부호 보강"""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'\s+', ' ', text).strip()
    if not re.search(r'[\.\!\?…"\'\)\]\}]$', text):
        text += '.'
    return text


def _strip_md(s: str) -> str:
    """마크다운 링크/강조 제거"""
    s = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', s)
    s = re.sub(r'~~([^~]+)~~', r'\1', s)
    return re.sub(r'(\*\*|\*|`|#+)', '', s)


# ── 질문 패턴 ────────────────────────────────────────────────────
_NEXT = (
    r"\n\s*(?:"
    r"-?\s*(?:질문|문제|Question)\s+\d+\.?\:?|"  # Added :? to match optional colon
    r"Q\s*\d+\.?\:?|"  # Added :? to match optional colon
    r"\d+\.\s+|"
    r"\d+\)\s+|"
    r"###\s*Question\s+\d+\:?|"  # Added specific pattern for "### Question N:"
    r"###)"
)

_PATS: Tuple[Pattern[str], ...] = tuple(
    re.compile(p, re.M | re.S) for p in (
        rf"^\s*(?:질문|문제)\s+\d+\.?\s*(.+?)(?={_NEXT}|\Z)",
        rf"^\s*###\s*질문\s+\d+\.?\s*(.+?)(?={_NEXT}|\Z)",
        rf"^\s*Q\s*\d+\.?\s*(.+?)(?={_NEXT}|\Z)",
        rf"^\s*Question\s+\d+\.?\s*(.+?)(?={_NEXT}|\Z)",
        rf"^\s*\d+\.\s+(.+?)(?={_NEXT}|\Z)",
        rf"^\s*\d+\)\s+(.+?)(?={_NEXT}|\Z)",
        # New pattern for "### Question N:"
        rf"^\s*###\s*Question\s+\d+\:?\s*(.+?)(?={_NEXT}|\Z)",
    )
)

# ── 메타데이터 패턴 (필터링용) ────────────────────────────
_META_PATTERNS = [
    r'^\s*\[.+?\]:\s*.+$',
    r'^\s*\*\*\[.+?\]\*\*:.*$',
    r'^\s*\[.+?\]$',
    r'^\s*\*\*\[.+?\]\*\*$',
    r'^\s*#+ .+$',
    r'^\s*\*\*.+?\*\*$',
    r'^\s*-\s+.*',
    r'^.*사용자.*정보.*$',
    r'^.*지원자.*답변.*$',
    r'^.*메인.*질문.*$',
    r'^.*사용자.*키워드.*$',
    r'^.*이전.*질문.*목록.*$',
    r'^.*참고.*질문.*목록.*$',
]

# ── 질문 여부 휴리스틱 ─────────────────────────────────────────
_QUESTION_HINTS = (
    r'\?'
    r'|무엇|어떻게|어디서|언제|누구|왜|어떤'
    r'|차이점|비교|장단점'
    r'|설명(?:해|하)[가-힣]*'
    r'|말씀(?:해|해줄)[가-힣]*'
    r'|주세요'
)


def is_likely_question(text: str) -> bool:
    """텍스트가 실제 질문인지 간단히 판별"""
    if any(re.search(p, text, re.I) for p in _META_PATTERNS):
        return False
    if len(text.strip()) < 10:
        return False
    return bool(re.search(_QUESTION_HINTS, text))

# ── 메인 파서 ──────────────────────────────────────────────────


def parse_questions(
    text: str,
    extra_pats: Optional[List[Pattern[str]]] = None,
    strip_md: bool = True,
) -> List[str]:
    """주어진 텍스트에서 면접 질문 리스트 추출"""
    txt = text.replace("\n...\n[출력 형식 예시]", "").strip()
    if txt.endswith("\nassistant"):
        txt = txt[: -len("\nassistant")]

    msec = re.search(r'##\s*\S*생성\s*결과[^\n]*', txt)
    if msec:
        txt = txt[msec.start():]

    m = re.search(r'(?:질문|문제|Question|Q)\s*\d+\.?|###\s*질문', txt)
    if m:
        txt = txt[m.start():]

    if len(txt) < 5:
        raise ValueError("텍스트가 너무 짧습니다")

    pats = list(_PATS) + (extra_pats or [])
    spans: List[Tuple[int, int, str]] = []
    for p in pats:
        for m in p.finditer(txt):
            spans.append((m.start(1), m.end(1), m.group(1).strip()))

    def count_korean(s: str) -> int:
        return len(re.findall(r'[가-힣]', s))

    if not spans:
        logger.info("패턴 매칭 실패, 휴리스틱 사용")
        questions: List[str] = []
        for line in txt.split('\n'):
            line = line.strip()
            if len(line) > 15 and is_likely_question(line) and count_korean(line) <= 100:
                questions.append(_norm(_fix(line)))
        return questions

    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    qs: List[str] = []
    seen: Set[str] = set()
    end = -1
    for s, e, raw in spans:
        if s < end:
            continue
        end = e
        q = raw
        if strip_md:
            q = _strip_md(q)
        q = q.strip().rstrip(',. ')
        if len(q) < 10 or q in seen:
            continue
        q = _norm(_fix(q))
        if count_korean(q) <= 100 and is_likely_question(q):
            qs.append(q)
            seen.add(q)
    return qs
