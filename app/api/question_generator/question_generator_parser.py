import re
import unicodedata
import logging
from typing import List, Pattern, Tuple, Dict, Set, Optional

logger = logging.getLogger(__name__)

_BRACKETS: Dict[str, str] = {'(': ')', '[': ']', '{': '}'}
_QUOTES: Set[str] = {'"', "'", '“', '”', '‘', '’'}

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
_NEXT = r'\n\s*(?:-?\s*(?:질문|문제|Question)\s+\d+\.?|Q\s*\d+\.?|\d+\.\s+|\d+\)\s+)'
_PATS: Tuple[Pattern[str], ...] = tuple(
    re.compile(p, re.M | re.S) for p in (
        rf'^\s*(?:질문|문제)\s+\d+\.?\s*(.+?)(?={_NEXT}|\Z)',
        rf'^\s*Q\s*\d+\.?\s*(.+?)(?={_NEXT}|\Z)',
        rf'^\s*Question\s+\d+\.?\s*(.+?)(?={_NEXT}|\Z)',
        rf'^\s*\d+\.\s+(.+?)(?={_NEXT}|\Z)',
        rf'^\s*\d+\)\s+(.+?)(?={_NEXT}|\Z)',
        rf'^\s*-\s*(?:질문|문제|Question)?\s*\d*\.?\s*(.+?)(?={_NEXT}|\Z)',
        r'^\s*[-*•]\s+(.+?)(?=\n\s*[-*•]|\Z)',
    )
)

# ── 메인 함수 ───────────────────────────────────────────────────


def parse_questions(
    text: str,
    extra_pats: Optional[List[Pattern[str]]] = None,
    strip_md: bool = True,
) -> List[str]:
    txt = text.replace("\n...\n[출력 형식 예시]",
                       "").strip().removesuffix("\nassistant")
    m = re.search(r'(?:질문|문제|Question|Q)\s+\d+\.?', txt)
    if m:
        txt = txt[m.start():]
    if len(txt) < 5:
        raise ValueError("텍스트가 너무 짧습니다")

    pats = list(_PATS) + (extra_pats or [])
    spans: List[Tuple[int, int, str]] = [(m.start(1), m.end(1), m.group(1).strip())
                                         for p in pats for m in p.finditer(txt)]

    if not spans:  # 휴리스틱
        logger.info("패턴 매칭 실패, 휴리스틱 사용")
        lines = [l for l in txt.split('\n') if len(l) > 10 and '?' in l]
        return [_norm(_fix(l.strip())) for l in lines] or [_norm(_fix(txt))]

    spans.sort(key=lambda t: (t[0], -(t[1]-t[0])))
    qs, seen, end = [], set(), -1
    for s, e, raw in spans:
        if s < end:
            continue
        q = _strip_md(raw) if strip_md else raw
        if len(q) < 10 or q in seen:
            continue
        q = _norm(_fix(q.rstrip(' ,.')))
        qs.append(q)
        seen.add(q)
        end = e
    return qs
