import re
import unicodedata
import logging
from typing import List, Pattern, Tuple, Dict, Set, Optional

logger = logging.getLogger(__name__)

_BRACKETS: Dict[str, str] = {'(': ')', '[': ']', '{': '}'}
_QUOTES: Set[str] = {'"', "'", '"', '"', ''', '''}


def _fix(text: str) -> str:
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
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'\s+', ' ', text).strip()
    if not re.search(r'[\.\!\?…\"\'\)\]\}]$', text):
        text += '.'
    return text


def _strip_md(s: str) -> str:
    s = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', s)
    s = re.sub(r'~~([^~]+)~~', r'\1', s)
    return re.sub(r'(\*\*|\*|`|#+)', '', s)


# 개선된 _NEXT 패턴: 다양한 질문 형식 인식 개선
_NEXT = (
    r"\n\s*(?:"
    r"-?\s*(?:질문|문제|Question|꼬리질문)\s+\d+\.?\:?|"
    r"Q\s*\d+\.?\:?|"
    r"\d+\.\s+|"
    r"\d+\)\s+|"
    r"###\s*Question\s+\d+\:?|"
    r"###)"
)

# 개선된 패턴: 다양한 질문 형식 인식 및 각 질문 간 더 정확한 구분
_PATS: Tuple[Pattern[str], ...] = tuple(
    re.compile(p, re.M | re.S) for p in (
        # 기존 패턴
        rf"^\s*(?:질문|문제|꼬리질문)\s+\d+\.?\s*(.+?)(?={_NEXT}|\Z)",
        rf"^\s*###\s*(?:질문|꼬리질문)\s+\d+\.?\s*(.+?)(?={_NEXT}|\Z)",
        rf"^\s*Q\s*\d+\.?\s*(.+?)(?={_NEXT}|\Z)",
        rf"^\s*Question\s+\d+\.?\s*(.+?)(?={_NEXT}|\Z)",
        rf"^\s*\d+\.\s+(.+?)(?={_NEXT}|\Z)",
        rf"^\s*\d+\)\s+(.+?)(?={_NEXT}|\Z)",
        rf"^\s*###\s*Question\s+\d+\:?\s*(.+?)(?={_NEXT}|\Z)",
        # 추가된 패턴: 단순 줄바꿈으로 구분된 질문들까지 인식
        r"질문\s+\d+\.?\s*(.+?)(?=\n\s*질문\s+\d+|\Z)",
        r"\d+\.\s+(.+?)(?=\n\s*\d+\.|\Z)",
    )
)

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

# 확장된 질문 힌트: 더 많은 질문 패턴을 인식하도록 개선
_QUESTION_HINTS = (
    r'\?'
    r'|무엇|어떻게|어디서|언제|누구|왜|어떤'
    r'|차이점|비교|장단점|방법|방식|기법'
    r'|설명(?:해|하)[가-힣]*'
    r'|말씀(?:해|해줄)[가-힣]*'
    r'|이야기[가-힣]*'
    r'|설명|논의|소개|안내|제시'
    r'|활용|구현|실행|적용|해결'
    r'|주세요|알려주세요|말해주세요'
    r'|[나요]$'  # 한국어 질문의 종결어미 패턴
    r'|트렌드|방향|연구|발전'  # 연구 트렌드 관련 키워드
)


# 개선된 질문 판단 함수
def is_likely_question(text: str) -> bool:
    # 메타 패턴 체크는 그대로 유지
    if any(re.search(p, text, re.I) for p in _META_PATTERNS):
        return False

    # 최소 길이 조건 완화
    if len(text.strip()) < 8:
        return False

    # 질문 번호가 포함된 경우 (예: "질문 2.")는 질문으로 간주
    if re.search(r'^(?:질문|문제)\s+\d+', text):
        return True

    # 일반적인 질문 식별
    if re.search(_QUESTION_HINTS, text):
        return True

    # 마침표나 물음표로 끝나는 문장
    if re.search(r'[\.\?]\s*$', text):
        # 문장이 충분히 길고, 명사나 동사로 끝나지 않는 경우
        if len(text.strip()) > 15:
            return True

    return False


def korean_char_length(text: str) -> int:
    return sum(1 for _ in text)


def parse_questions(
    text: str,
    extra_pats: Optional[List[Pattern[str]]] = None,
    strip_md: bool = True,
) -> List[str]:
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

    # 패턴 매칭 로직
    pats = list(_PATS) + (extra_pats or [])
    spans: List[Tuple[int, int, str]] = []
    for p in pats:
        for m in p.finditer(txt):
            spans.append((m.start(1), m.end(1), m.group(1).strip()))

    # 패턴 매칭 실패 시 휴리스틱 사용
    if not spans:
        logger.info("패턴 매칭 실패, 휴리스틱 사용")
        questions: List[str] = []

        # 줄 단위로 분리하여 각 줄이 질문인지 확인
        lines = txt.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # 질문 번호 패턴 (예: "질문 1.", "2.")이 있는지 확인
            if re.match(r'(?:질문|문제)?\s*\d+\.?', line):
                question_text = line

                # 질문이 여러 줄에 걸쳐 있을 수 있으므로 다음 질문이 나올 때까지 합침
                j = i + 1
                while j < len(lines) and not re.match(r'(?:질문|문제)?\s*\d+\.?', lines[j].strip()):
                    if lines[j].strip():
                        question_text += " " + lines[j].strip()
                    j += 1

                if len(question_text) > 10:
                    questions.append(_norm(_fix(question_text)))

            # 일반 질문 확인
            elif len(line) > 15 and is_likely_question(line) and korean_char_length(line) <= 100:
                questions.append(_norm(_fix(line)))

        return questions

    # 패턴 매칭 결과 정렬 및 중복 제거
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

        # 너무 짧은 질문이나 이미 본 질문 건너뛰기 (최소 길이 조건 완화)
        if len(q) < 8 or q in seen:
            continue

        q = _norm(_fix(q))

        # 최대 길이 제한
        if korean_char_length(q) > 150:  # 길이 제한 증가
            continue

        # 은/는/이/가 등의 조사로 끝나는 문장도 포함 (주어만 있는 불완전한 문장 제외)
        if len(q) > 12 or is_likely_question(q):
            qs.append(q)
            seen.add(q)

    return qs
