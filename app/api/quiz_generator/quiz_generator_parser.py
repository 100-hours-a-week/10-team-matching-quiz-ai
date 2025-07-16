import re
from typing import List, Dict
import ast
import logging
import sys
import random

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 콘솔 핸들러 설정 추가
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def parse_choices(raw_options: str) -> List[str]:
    """
    문자열 형태의 선지(raw_options)를 4지선다 리스트 형태로 파싱
    """
    try:
        # 1. ["보기1", "보기2"] 형태 대응
        parsed = ast.literal_eval(raw_options)
        return [opt.strip().strip('"') for opt in parsed if isinstance(opt, str)]
    except (ValueError, SyntaxError):
        pass

    # 2. A. ~ 혹은 줄바꿈으로 구분된 선지일 경우
    split_by_letter = re.findall(r"[A-D]\.\s*([^A-D]+?)(?=(?:[A-D]\.|$))", raw_options)
    if len(split_by_letter) == 4:
        return [opt.strip().strip('"') for opt in split_by_letter]

    # 3. 쉼표나 줄바꿈을 기준으로 자르고 공백/따옴표 정리
    options = re.split(r"[,\n]", raw_options)
    options = [opt.strip().strip('"') for opt in options if opt.strip()]
    if len(options) == 4:
        return options

    # 모든 파싱 전략 실패
    logger.warning("선지 파싱 실패")
    return[]


def is_valid_quiz_item(item: Dict) -> bool:
    """
    개별 퀴즈 항목에 대한 유효한 형식인지 확인
    """
    if not item.get("question"):
        logger.debug("생성 실패: question 없음")
        return False

    if not isinstance(item.get("options"), list):
        logger.debug("생성 실패: options 타입이 list 아님")
        return False

    if len(item["options"]) != 4:
        logger.debug("생성 실패: options 개수가 4개가 아님")
        return False

    if not isinstance(item.get("answer_index"), int):
        logger.debug("생성 실패: answer_index가 int 아님")
        return False

    if not (1 <= item["answer_index"] <= 4):
        logger.debug("생성 실패: answer_index 범위 오류")
        return False

    if not item.get("explanation"):
        logger.debug("생성 실패: explanation 없음")
        return False

    if len(item["question"]) > 200:
        logger.debug("생성 실패: question이 너무 김 (%d자)", len(item["question"]))
        return False

    if len(item["explanation"]) > 300:
        logger.debug("생성 실패: explanation이 너무 김 (%d자)", len(item["explanation"]))
        return False

    return True


def score_quiz(q: Dict) -> int:
    """퀴즈 하나를 품질 점수로 환산"""
    score = 0

    # 질문 길이
    # score += len(q.get("question", ""))

    # 보기 개수(4지선다 선호)
    num_opts = len(q.get("options", []))
    if num_opts == 4:
        score += 20        # 이상적
    else:
        score += max(0, 16 - abs(4 - num_opts) * 4)

    # 설명(해설) 길이
    score += len(q.get("explanation", "")) // 2   # 길수록 가점

    # 중복 방지용 난수(동점 시 랜덤 섞임)
    score += random.randint(0, 3)

    return score


def filter_and_select_quizzes(quizzes: List[Dict]) -> List[Dict]:
    """
    생성된 전체 퀴즈 리스트에서 유효한 형식의 퀴즈만 선별하여,
    난이도별 조건에 따라 총 10문제 최종 추출 /
    개수가 부족한 난이도가 있다면 다른 난이도에서 품질 순으로 추가 선별
    """
    logger.info(f"총 파싱된 퀴즈 수: {len(quizzes)}")
    # 난이도별 버킷(리스트 구성))
    buckets = {"하": [], "중": [], "상": []}
    for q in quizzes:
        diff = q.get("difficulty")
        if diff in buckets:
            buckets[diff].append(q)

    # 목표 파싱 개수 설정
    target = {"하": 4, "중": 3, "상": 3}
    selected: List[Dict] = []
    deficits = {}
    
    # 초기 선택 / 부족한 개수 계산
    for diff, tgt in target.items():
        take_cnt = min(len(buckets[diff]), tgt)
        sorted_by_quality = sorted(buckets[diff], key=score_quiz, reverse=True)
        selected.extend(sorted_by_quality[:take_cnt])
        logger.info(f"난이도 [{diff}] 목표: {tgt}, 실제 확보: {take_cnt}")

        if take_cnt < tgt:
            deficits[diff] = tgt - take_cnt

    # 목표 초과분(남는 문제) 구성 / 정렬
    surplus_pool = []
    for diff, tgt in target.items():
        sorted_by_quality = sorted(buckets[diff], key=score_quiz, reverse=True)
        surplus_pool.extend(sorted_by_quality[tgt:])

    surplus_pool.sort(key=score_quiz, reverse=True)

    # 부족한 개수 채우기
    for diff, need in deficits.items():
        logger.info(f"난이도 [{diff}] 부족분: {need} → 다른 난이도에서 보충 시도")
        if surplus_pool:
            for q in surplus_pool[:need]:
                logger.info(f"보충 문제 선택 (점수={score_quiz(q)}): {q['question'][:30]}...")
            selected.extend(surplus_pool[:need])
            surplus_pool = surplus_pool[need:]

    # 전체 파싱 10개 미만일 때 예외처리
    if len(selected) < 10:
        logger.warning(f"총 문제 수가 10개 미만입니다. 부족한 {10 - len(selected)}개를 남은 문제 중에서 보충합니다.")
        remaining_pool = [q for q in quizzes if q not in selected]
        remaining_pool.sort(key=score_quiz, reverse=True)
        selected.extend(remaining_pool[: 10 - len(selected)])

    # 난이도별 최종 선택 개수 집계
    final_counts = {"하": 0, "중": 0, "상": 0}
    for q in selected:
        diff = q.get("difficulty")
        if diff in final_counts:
            final_counts[diff] += 1

    logger.info(
        f"총 {len(selected)}개 퀴즈 선정 완료 "
        f"(하: {final_counts['하']}, 중: {final_counts['중']}, 상: {final_counts['상']})"
    )

    return selected[:10]


def parse_response(response_text: str):
    """
    LLM의 출력 문자열에서 퀴즈 정규식 기반으로 파싱한 후,
    List[Dict] 형식으로 반환
    """
    # 프롬프트 제거
    response_text = remove_prompt_content(response_text)

    # Quiz 정규표현식으로 추출
    QUESTION_PATTERN = re.compile(
        r"난이도:\s*(?P<difficulty>하|중|상)\s*"
        r"문제:\s*(?P<question>.*?)\s*"
        r"선지:\s*\[(?P<choices>.*?)\]\s*"
        r"정답\s*인덱스:\s*(?P<answer_index>[1-4])\s*"
        r"해설:\s*(?P<explanation>.*?)(?=\n난이도:|\Z)",
        re.DOTALL,
    )

    quiz_list = []
    matches = QUESTION_PATTERN.findall(response_text)
    valid_difficulties = {"상", "중", "하"}

    # 퀴즈 하나씩 처리
    for i, match in enumerate(matches, 1):
        difficulty, question, options, answer_index, explanation = match
        
        # option을 파싱해서 4지선다 리스트로 변환
        option_list = parse_choices(options)

        if len(option_list) != 4:
            logger.warning(f"[{i}] 보기 항목 수가 4개가 아님")
            continue
        if difficulty.strip() not in valid_difficulties:
            logger.warning(f"[{i}] 난이도 필드가 잘못됨")
            continue
        if not answer_index.isdigit() or not (1 <= int(answer_index) <= 4):
            logger.warning(f"[{i}] 정답 인덱스가 유효하지 않음")
            continue

        # 퀴즈 Dict 변환 후 추가
        quiz_list.append(
            {
                "difficulty": difficulty.strip(),
                "question": question.strip(),
                "options": option_list,
                "answer_index": int(answer_index),  # 1부터 시작
                "explanation": explanation.strip(),
            }
        )

    return quiz_list


def remove_prompt_content(output: str) -> str:
    """
    프롬프트 안내문을 제거하고 문제 본문만 추출
    """
    end_token = "--- END OF INSTRUCTION ---"
    if end_token in output:
        return output.split(end_token, 1)[-1].strip()
    return output.strip()  # fallback