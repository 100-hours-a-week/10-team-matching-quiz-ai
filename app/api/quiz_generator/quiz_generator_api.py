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

    # Langfuse Prompt 불러오기
    prompt_template = langfuse.get_prompt("quiz_generation")

    joined_questions = "\n".join(req.question_history_list)

    # Langfuse 템플릿 객체가 compile 메서드를 지원하는지 확인
    if hasattr(prompt_template, "compile"):
        prompt = prompt_template.compile(joined=joined_questions)
    else:
        # fallback: 직접 문자열 치환
        prompt = prompt_template.replace("{{joined}}", joined_questions)

    # 프롬프트 빌드 트레이스 기록
    if trace:
        trace.span(name="build_prompt", input=prompt)

    # 퀴즈 생성
    raw_output = generate_quiz(prompt)

    if trace:
        trace.span(
            name="llm_response",
            input=prompt,
            output=raw_output,
        )

    # 퀴즈 파싱
    parsed_list = parse_response(raw_output)
    quiz_items = [QuizItem(**item) for item in parsed_list]

    print(f"\n 전체 퀴즈 응답 반환 완료: 총 {len(quiz_items)}문항")

    return FollowupResponse(
        message="quiz_generated",
        data={
            "user_id": req.interview_id,
            "questions": quiz_items
        }
    )
