import re
import unicodedata
import logging
from typing import List, Pattern, Tuple, Dict, Set, Optional

logger = logging.getLogger(__name__)

_BRACKETS: Dict[str, str] = {'(': ')', '[': ']', '{': '}'}
_QUOTES: Set[str] = {'"', "'", '"', '"', ''', '''}

# ── 텍스트 유틸 ──────────────────────────────────────────────────


def _fix(text: str) -> str:
    stk: List[str] = []
    for ch in text:
        if ch in _BRACKETS:
            stk.append(ch)
        elif ch in _BRACKETS.values() and stk and ch == _BRACKETS[stk[-1]]:
            stk.pop()
    if stk:
        i = text.find('\n', len(text) - 1)
        i = i if i != -1 else len(text)
        text = text[:i] + ''.join(_BRACKETS[b]
                                  for b in reversed(stk)) + text[i:]
    for q in _QUOTES:
        if text.count(q) % 2:
            text += q
    return text


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text if re.search(r'[.!?…"\')\]}]$', text) else text + '.'


def _strip_md(s: str) -> str:
    s = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', s)
    s = re.sub(r'~~([^~]+)~~', r'\1', s)
    return re.sub(r'(\*\*|\*|`|#+)', '', s)


# ── 패턴 ────────────────────────────────────────────────────────
_NEXT = r'\n\s*(?:-?\s*(?:질문|문제|Question)\s+\d+\.?|Q\s*\d+\.?|\d+\.\s+|\d+\)\s+|###)'
_PATS: Tuple[Pattern[str], ...] = tuple(
    re.compile(p, re.M | re.S) for p in (
        rf'^\s*(?:질문|문제)\s+\d+\.?\s*(.+?)(?={_NEXT}|\Z)',  # 질문 1. 형식
        rf'^\s*###\s*질문\s+\d+\.?\s*(.+?)(?={_NEXT}|\Z)',     # ### 질문 1. 형식
        rf'^\s*Q\s*\d+\.?\s*(.+?)(?={_NEXT}|\Z)',             # Q1. 형식
        rf'^\s*Question\s+\d+\.?\s*(.+?)(?={_NEXT}|\Z)',      # Question 1. 형식
        rf'^\s*\d+\.\s+(.+?)(?={_NEXT}|\Z)',                  # 1. 형식
        rf'^\s*\d+\)\s+(.+?)(?={_NEXT}|\Z)',                  # 1) 형식
    )
)

# ── 메타데이터 패턴 (필터링용) ────────────────────────────
_META_PATTERNS = [
    r'^\s*\[.+?\]:\s*.+$',                  # [키워드]: 값 형식
    r'^\s*\*\*\[.+?\]\*\*:\s*.+$',          # **[키워드]**: 값 형식
    r'^\s*\[.+?\]$',                        # [레이블] 형식
    r'^\s*\*\*\[.+?\]\*\*$',                # **[레이블]** 형식
    r'^\s*\#+ .+$',                         # 마크다운 헤더
    r'^\s*\*\*.+?\*\*$',                    # **강조** 형식
    r'^\s*-\s+.*',                          # 리스트 항목
    r'^.*사용자.*정보.*$',                    # '사용자 정보' 포함
    r'^.*지원자.*답변.*$',                    # '지원자 답변' 포함
    r'^.*메인.*질문.*$',                      # '메인 질문' 포함
    r'^.*사용자.*키워드.*$',                   # '사용자 키워드' 포함
    r'^.*이전.*질문.*목록.*$',                 # '이전 질문 목록' 포함
    r'^.*참고.*질문.*목록.*$',                 # '참고 질문 목록' 포함
]

# ── 메인 함수 ───────────────────────────────────────────────────


def is_likely_question(text: str) -> bool:
    """텍스트가 실제 질문인지 여부를 판단"""
    # 메타데이터 패턴에 해당하면 질문이 아님
    for pattern in _META_PATTERNS:
        if re.search(pattern, text, re.I):
            return False

    # 명확한 질문 형식인지 확인
    is_explicit_question = bool(re.match(
        r'^(?:질문|문제|Question|Q)?\s*\d+\.?\s+', text))

    # 물음표가 있거나, 특정 질문 구조를 가지면 질문으로 간주
    has_question_mark = '?' in text
    has_question_words = bool(re.search(
        r'\b(무엇|어떻게|어디서|언제|누구|왜|어떤|설명해|말씀해|주세요)\b', text))

    # 접두어 제거 후 텍스트가 너무 짧으면 질문이 아님
    clean_text = re.sub(r'^(?:질문|문제|Question|Q)?\s*\d+\.?\s*', '', text)
    if len(clean_text.strip()) < 10:
        return False

    # [키워드] 형식의 템플릿 질문은 제외
    if re.search(r'\[\s*사용자\s*키워드\s*\]', text):
        return False

    return is_explicit_question or has_question_mark or has_question_words


def parse_questions(
    text: str,
    extra_pats: Optional[List[Pattern[str]]] = None,
    strip_md: bool = True,
) -> List[str]:
    txt = text.replace("\n...\n[출력 형식 예시]",
                       "").strip().removesuffix("\nassistant")

    # 'Generate Results' 섹션 찾기 시도
    results_section = re.search(
        r'##\s*\S*\s*생성\s*결과\s*\S*\s*$.*?(?=---|\Z)', txt, re.DOTALL | re.M)
    if results_section:
        txt = results_section.group(0)

    # 첫번째 질문 패턴 찾기
    m = re.search(r'(?:질문|문제|Question|Q)\s+\d+\.?|###\s*질문', txt)
    if m:
        txt = txt[m.start():]

    if len(txt) < 5:
        raise ValueError("텍스트가 너무 짧습니다")

    pats = list(_PATS) + (extra_pats or [])
    spans: List[Tuple[int, int, str]] = [(m.start(1), m.end(1), m.group(1).strip())
                                         for p in pats for m in p.finditer(txt)]

    def korean_char_count(s: str) -> int:
        return len(re.findall(r'[가-힣]', s))

    # 패턴 매칭이 없으면 줄 단위 질문 찾기
    if not spans:
        logger.info("패턴 매칭 실패, 휴리스틱 사용")
        lines = [l.strip() for l in txt.split('\n') if len(l.strip()) > 15]
        questions = []

        for line in lines:
            if '?' in line and korean_char_count(line) <= 100 and is_likely_question(line):
                questions.append(_norm(_fix(line.strip())))

        return questions

    # 패턴 매칭 결과 처리
    spans.sort(key=lambda t: (t[0], -(t[1]-t[0])))
    qs, seen, end = [], set(), -1

    for s, e, raw in spans:
        if s < end:
            continue

        # 각 매치를 정제하고 질문 형식인지 확인
        clean_q = _strip_md(raw) if strip_md else raw
        clean_q = clean_q.strip()

        # 너무 짧은 텍스트나 중복 제외
        if len(clean_q) < 10 or clean_q in seen:
            continue

        # 정규화 및 고정
        clean_q = _norm(_fix(clean_q.rstrip(' ,.')))

        # 최종 필터링: 한글 길이 및 질문 형식 확인
        if korean_char_count(clean_q) <= 100 and is_likely_question(clean_q):
            qs.append(clean_q)
            seen.add(clean_q)

        end = e

    return qs
