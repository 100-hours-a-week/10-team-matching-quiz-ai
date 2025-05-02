import os
import random
import json
import time
from tqdm import tqdm
from openai import OpenAI, OpenAIError
from langfuse.decorators import observe
from langfuse.openai import openai as lf_openai

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
lf_openai.api_key = os.environ["OPENAI_API_KEY"]

# ── 데이터 매핑 ─────────────────────────────────────────────────────────────────
MAPPING = {
    "풀스택": {"jobs": ["풀스택 개발자", "백엔드 개발자", "프론트엔드 개발자"]},
    "클라우드": {"jobs": ["클라우드 엔지니어", "솔루션즈 아키텍트", "DevOps 엔지니어"]},
    "인공지능": {"jobs": ["머신러닝 엔지니어", "AI 백엔드 개발자", "데이터 사이언티스트"]},
}

# ── 프롬프트 생성 ────────────────────────────────────────────────────────────────


def build_prompt_for_main_question_only(course: str, job: str) -> str:
    return f"""
당신은 카부캠 교육생들을 대상으로 모의 면접을 진행하는 AI 면접관입니다.

아래 정보를 참고하여,
- 주어진 과정/희망 직무를 바탕으로
- 현실적인 면접 상황에 맞는 대표 질문 하나를 생성하고,
- 해당 대표 질문에 이어질 수 있는 꼬리 질문 4개를 생성하세요.

# 조건
- 과정에 맞는 깊이와 범위를 고려하세요.
- 기술 스택을 활용한 구체적인 질문이어야 합니다.
- 희망 직무에 필요한 역량을 반영하세요.
- 꼬리 질문들은 서로 다른 관점(개념/구현/문제 해결/최신 트렌드 등)을 다뤄야 합니다.
- 결과만 출력하세요.

# 입력 정보
수강 과정: {course}
희망 직무: {job}

# 출력 포맷
메인 질문:
{{대표 질문 한 개}}

꼬리 질문:
A. {{꼬리 질문 1}}
B. {{꼬리 질문 2}}
C. {{꼬리 질문 3}}
D. {{꼬리 질문 4}}
""".strip()

# ── 샘플링 ───────────────────────────────────────────────────────────────────────


def sample_course_job_only():
    course = random.choice(list(MAPPING.keys()))
    job = random.choice(MAPPING[course]["jobs"])
    return course, job

# ── 응답 파싱 ────────────────────────────────────────────────────────────────────


def parse_response(raw: str) -> tuple[str, str]:
    """
    raw: API에서 온 원문 (메인 질문과 꼬리 질문이 섞여 있음)
    returns: (main_question, followups_str)
    """
    if "꼬리 질문" not in raw:
        raise ValueError("꼬리 질문 섹션을 찾을 수 없습니다.")

    main_part, followup_part = raw.split("꼬리 질문", 1)

    # 1) 메인 질문 추출
    lines_main = [l.strip() for l in main_part.splitlines() if l.strip()]
    if lines_main[0].startswith("메인 질문:"):
        if lines_main[0] != "메인 질문:":
            main_q = lines_main[0].split(":", 1)[1].strip()
        else:
            main_q = lines_main[1]
    else:
        main_q = lines_main[0]

    # 2) 꼬리 질문 추출 및 번호 매기기
    mapping = {"A.": 1, "B.": 2, "C.": 3, "D.": 4}
    followups = []
    for line in followup_part.splitlines():
        ls = line.strip()
        for prefix, idx in mapping.items():
            if ls.startswith(prefix):
                text = ls[len(prefix):].strip()
                followups.append(f"질문 {idx}. {text}")
                break

    if not followups:
        raise ValueError("꼬리 질문을 찾을 수 없습니다.")

    return main_q, "\n".join(followups)

# ── Langfuse 모니터링된 API 호출 ─────────────────────────────────────────────────


@observe(name="generate_main_question")
def generate_question(prompt: str, retries: int = 3) -> str:
    for attempt in range(1, retries + 1):
        try:
            resp = lf_openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",  "content": "당신은 전문 IT 면접관입니다."},
                    {"role": "user",    "content": prompt},
                ],
                temperature=0.8,
                max_tokens=200,
            )
            return resp.choices[0].message.content.strip()
        except OpenAIError as e:
            if "quota" in str(e).lower():
                raise
            time.sleep(2**attempt)
    return ""

# ── 메인 함수: 기존 followup_dataset_2000.jsonl에 이어붙이기 ────────────────────────


def main(total: int = 500, output_file: str = "fine_tuning_dataset_2.jsonl") -> None:
    seen = set()
    successful = 0
    attempts = 0
    max_attempts = total * 3
    error_counts = {"parsing": 0, "api": 0, "other": 0}

    with open(output_file, "a", encoding="utf-8") as f_out:
        pbar = tqdm(total=total, desc="메인 질문 생성 중")
        while successful < total and attempts < max_attempts:
            attempts += 1
            try:
                course, job = sample_course_job_only()
                prompt = build_prompt_for_main_question_only(course, job)
                raw = generate_question(prompt)
                if not raw:
                    error_counts["api"] += 1
                    continue

                main_q, followups = parse_response(raw)
                if main_q in seen:
                    continue
                seen.add(main_q)

                record = {
                    "input":  f"메인 질문: {main_q}",
                    "output": followups
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()

                successful += 1
                pbar.update(1)

            except ValueError:
                error_counts["parsing"] += 1
            except OpenAIError:
                error_counts["api"] += 1
                time.sleep(5)
            except Exception:
                error_counts["other"] += 1
                time.sleep(1)

        pbar.close()

    print(f"\n🗄️ 총 {successful}개 메인 질문을 '{output_file}'에 이어붙였습니다.")
    print(f"📊 오류 통계: {error_counts}")
    print(f"⚙️ 성공률: {successful/attempts:.2%}")


if __name__ == "__main__":
    main()
