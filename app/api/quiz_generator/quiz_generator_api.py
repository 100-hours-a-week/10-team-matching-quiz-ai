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
        metadata={"endpoint": "/quiz/generate_quiz"}
    )

    prompt_template = langfuse.get_prompt("quiz_generation")
    joined_questions = "\n".join(req.question_history_list)

    if hasattr(prompt_template, "compile"):
        prompt = prompt_template.compile(joined=joined_questions)
    else:
        prompt = prompt_template.replace("{{joined}}", joined_questions)
        prompt += "\n\n-출력은 위의 형식을 정확히 따르고, 반드시 최소 15문제를 연속 출력하시오. 다른 설명은 절대 포함하지 마세요."

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
    parsed_list = parse_response(raw_output)
    final_quizzes = filter_and_select_quizzes(parsed_list)

    if len(final_quizzes) != 10:
        if trace:
            span_error = trace.span(name="validation_error", input=raw_output)
            span_error.update(status="error", metadata={"reason": f"{len(final_quizzes)}개 생성됨"})
        raise ValueError(f"형식이 맞는 퀴즈가 부족합니다: {len(final_quizzes)}개 추출됨")

    quiz_items = [QuizItem(**item) for item in final_quizzes]

    print(f"\n 전체 quiz 응답 수: 총 {len(quiz_items)}문항")

    return FollowupResponse(
        message="quiz_generated",
        data={
            "user_id": req.interview_id,
            "questions": quiz_items
        }
    )
