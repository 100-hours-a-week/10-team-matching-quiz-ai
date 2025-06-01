from fastapi import APIRouter
from app.api.quiz_generator.quiz_generator_schema import FollowupRequest, FollowupResponse, QuizItem
from app.api.quiz_generator.quiz_generator_parser import parse_response
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
        prompt += "\n\n-출력은 위의 형식을 정확히 따르고, 반드시 10문제를 연속 출력하시오. 다른 설명은 절대 포함하지 마세요."

    if trace:
        trace.span(name="build_prompt", input=prompt)

    # LLM 호출
    print("quiz generate 시작")
    raw_output = generate_quiz(prompt)
    print("quiz 생성 완료")

    if trace:
        trace.span(name="llm_response", input=prompt, output=raw_output)

    # 파싱 시도
    parsed_list = parse_response(raw_output)

    # 문제 수가 10개인지 확인
    if len(parsed_list) != 10:
        raise ValueError(f"quiz 문제 수가 부족합니다: {len(parsed_list)}개 생성됨")

    quiz_items = [QuizItem(**item) for item in parsed_list]

    print(f"\n 전체 quiz 응답 수: 총 {len(quiz_items)}문항")

    return FollowupResponse(
        message="quiz_generated",
        data={
            "user_id": req.interview_id,
            "questions": quiz_items
        }
    )
