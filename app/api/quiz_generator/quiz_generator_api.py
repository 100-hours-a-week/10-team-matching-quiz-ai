from fastapi import APIRouter
from app.api.quiz_generator.quiz_generator_schema import FollowupRequest, FollowupResponse, QuizItem
from app.api.quiz_generator.quiz_generator_parser import parse_response, filter_and_select_quizzes
from app.api.quiz_generator.quiz_generator_model import generate_quiz
from app.api.quiz_generator.quiz_generator_config import (
    QUIZ_LANGFUSE_SECRET_KEY,
    QUIZ_LANGFUSE_PUBLIC_KEY,
    QUIZ_LANGFUSE_HOST,
)
from langfuse import Langfuse

router = APIRouter()

# Langfuse 클라이언트 초기화
langfuse = Langfuse(
    secret_key=QUIZ_LANGFUSE_SECRET_KEY,
    public_key=QUIZ_LANGFUSE_PUBLIC_KEY,
    host=QUIZ_LANGFUSE_HOST,
)

@router.post("/generate_quiz", response_model=FollowupResponse)
def generate_quiz_api(req: FollowupRequest):
    print(" 요청 수신: /generate_quiz")

    trace = langfuse.trace(
        name="quiz_generation",
        user_id=req.interview_id,
        tags=["quiz", "generate"],
        metadata={"endpoint": "/generate_quiz"}
    )

    prompt_template = langfuse.get_prompt("quiz_generation")
    joined_questions = "\n".join(req.question_history_list)

    if hasattr(prompt_template, "compile"):
        prompt = prompt_template.compile(joined=joined_questions)
    else:
        prompt = prompt_template.replace("{{joined}}", joined_questions)
    prompt += (
        "\n\n- 출력은 반드시 다음 형식을 따를 것:\n"
        "난이도: 하|중|상\n"
        "문제: (문장)\n"
        "선지: [1. ..., 2. ..., 3. ..., 4. ...]\n"
        "정답 인덱스: (1~4)\n"
        "해설: (문장)\n\n"
        "- 위 형식 그대로 15~20문제를 연속 출력하시오. 설명은 포함하지 말고 문제만 출력하시오."
    )

    # prompt 생성 span
    if trace:
        span_prompt = trace.span(name="build_prompt", input=prompt)
        span_prompt.update(status="success")

    # LLM 호출
    print("quiz generate 시작")
    raw_output = generate_quiz(prompt)
    print("quiz 생성 완료")

    # LLM 응답 처리 span
    if trace:
        span_llm = trace.span(name="llm_response", input=prompt, output=raw_output)
        span_llm.update(status="success")

    # 파싱 시도
    parsed_list = parse_response(raw_output, verbose=True)

    # 먼저 전체 형식이 맞는 퀴즈 수만 체크 (여기선 에러 발생 안 함)
    print(f"[DEBUG] 총 형식이 맞는 문제 수: {len(parsed_list)}개")

    # 난이도별 필터링 (이 함수가 하4/중3/상3으로 엄격하게 뽑음)
    final_quizzes = filter_and_select_quizzes(parsed_list)

    # 난이도별 충족 안 될 때만 에러
    if len(final_quizzes) != 10:
        # 각 난이도 개수 확인
        easy = len([q for q in parsed_list if q["difficulty"] == "하"])
        medium = len([q for q in parsed_list if q["difficulty"] == "중"])
        hard = len([q for q in parsed_list if q["difficulty"] == "상"])

        if trace:
            trace.span(name="filtering_error", input=str(parsed_list)).update(
                status="error",
                metadata={
                    "reason": f"난이도별 문제 부족",
                    "하": easy,
                    "중": medium,
                    "상": hard
                }
            )
        raise ValueError(f"난이도별 문제 부족 - 하:{easy}, 중:{medium}, 상:{hard}")

    quiz_items = [QuizItem(**item) for item in final_quizzes]

    print(f"\n 전체 quiz 응답 수: 총 {len(quiz_items)}문항")

    trace.update(status="success")

    return FollowupResponse(
        message="quiz_generated",
        data={
            "user_id": req.interview_id,
            "questions": quiz_items
        }
    )
