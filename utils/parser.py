import re
import logging
import unicodedata
from typing import List, Optional, Pattern, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Balance helpers
# ---------------------------------------------------------------------------

_BRACKETS = {'(': ')', '[': ']', '{': '}'}
_QUOTES = ['"', "'", '“', '”', '‘', '’']


def is_balanced_text(text: str) -> bool:
    """Return *True* if parentheses/brackets/quotes in *text* are balanced."""
    stack: List[str] = []
    for ch in text:
        if ch in _BRACKETS:
            stack.append(ch)
        elif ch in _BRACKETS.values():
            if not stack or ch != _BRACKETS[stack.pop()]:
                return False

    if stack:
        return False

    # Count every quote mark we track; any odd count → unbalanced
    return all(text.count(q) % 2 == 0 for q in _QUOTES)


def _insert_missing_closers(text: str, closers: List[str]) -> str:
    """Insert *closers* just before the newline closest to EOF (or at EOF)."""
    if not closers:
        return text
    idx = text.find('\n', len(text) - 1)
    insert_at = idx if idx != -1 else len(text)
    return text[:insert_at] + ''.join(closers) + text[insert_at:]


def fix_unbalanced_text(text: str) -> str:
    """Add minimal closing tokens to *text* so that it becomes balanced."""
    if is_balanced_text(text):
        return text

    stack: List[str] = []
    for ch in text:
        if ch in _BRACKETS:
            stack.append(ch)
        elif ch in _BRACKETS.values() and stack and ch == _BRACKETS[stack[-1]]:
            stack.pop()

    fixed = _insert_missing_closers(text, [_BRACKETS[b] for b in reversed(stack)])

    # fix quotes
    for q in _QUOTES:
        if fixed.count(q) % 2:
            fixed += q
    return fixed

# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

_PUNCT_TAIL = re.compile(r'[.!?…“”‘’"\')\]\}]$')


def normalize_text(text: str) -> str:
    """Collapse whitespace, NFC‑normalise and ensure sentence termination."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'\s+', ' ', text).strip()

    if text and not _PUNCT_TAIL.search(text):
        text += '.'
    return text

# ---------------------------------------------------------------------------
# Markdown stripping helpers
# ---------------------------------------------------------------------------

_MD_BASICS = re.compile(r'(\*\*|\*|`|#+)')
_MD_LINK = re.compile(r'\[([^\]]+)\]\([^)]+\)')
_MD_STRIKE = re.compile(r'~~([^~]+)~~')


def _strip_markdown(s: str) -> str:
    """Remove basic Markdown markup from *s*."""
    s = _MD_LINK.sub(r'\1', s)
    s = _MD_STRIKE.sub(r'\1', s)
    s = _MD_BASICS.sub('', s)
    return s

# ---------------------------------------------------------------------------
# Question parsing
# ---------------------------------------------------------------------------

_NEXT_MARKER = r'\n\s*(?:-?\s*(?:질문|문제|Question)\s+\d+\.?|Q\s*\d+\.?|\d+\.\s+|\d+\)\s+)'

_DEFAULT_PATTERNS: Tuple[Pattern[str], ...] = (
    re.compile(rf'^(?:질문|문제)\s+\d+\.?\s*(.+?)(?={_NEXT_MARKER}|\Z)', re.M | re.S),
    re.compile(rf'^Q\s*\d+\.?\s*(.+?)(?={_NEXT_MARKER}|\Z)', re.M | re.S),
    re.compile(rf'^Question\s+\d+\.?\s*(.+?)(?={_NEXT_MARKER}|\Z)', re.M | re.S),
    re.compile(rf'^\d+\.\s+(.+?)(?={_NEXT_MARKER}|\Z)', re.M | re.S),
    re.compile(rf'^\d+\)\s+(.+?)(?={_NEXT_MARKER}|\Z)', re.M | re.S),
    re.compile(rf'^\s*-\s*(?:질문|문제|Question)?\s*\d*\.?\s*(.+?)(?={_NEXT_MARKER}|\Z)', re.M | re.S),
    re.compile(r'^\s*[-*•]\s+(.+?)(?=\n\s*[-*•]|\Z)', re.M | re.S),
)


def parse_questions(
    text: str,
    patterns: Optional[List[Pattern[str]]] = None,
    strip_markdown: bool = True,
) -> List[str]:
    """Extract a list of questions from *text*.

    Supports multiple Korean/English list styles. Raises *ValueError* if text is
    obviously invalid (shorter than 5 chars).
    """
    cleaned = (
        text.replace("\n...\n[출력 형식 예시]", "")
        .strip()
        .removesuffix("\nassistant")
    )

    if len(cleaned) < 5:
        raise ValueError("Input text too short to parse questions")

    pat_list = list(_DEFAULT_PATTERNS)
    if patterns:
        pat_list.extend(patterns)

    spans: List[Tuple[int, int, str]] = []
    for pat in pat_list:
        spans.extend(
            (m.start(1), m.end(1), m.group(1).strip())
            for m in pat.finditer(cleaned)
        )

    if not spans:
        logger.info("No pattern matched; fallback heuristics engaged.")
        maybe = [
            normalize_text(fix_unbalanced_text(line.strip()))
            for line in cleaned.split('\n')
            if len(line) > 10 and ('?' in line or re.search(r'질문|query|ask', line, re.I))
        ]
        return maybe if maybe else [normalize_text(fix_unbalanced_text(cleaned))]

    # sort by *start* asc, and prefer longer span if start equals
    spans.sort(key=lambda t: (t[0], -(t[1] - t[0])))

    questions: List[str] = []
    processed: set[str] = set()
    last_end = -1

    for start, end, raw in spans:
        if start < last_end:
            continue
        q = _strip_markdown(raw) if strip_markdown else raw
        if len(q) < 10 or q in processed or q in {"...", ""}:
            continue
        q = normalize_text(fix_unbalanced_text(q))
        questions.append(q)
        processed.add(q)
        last_end = end

    return questions
