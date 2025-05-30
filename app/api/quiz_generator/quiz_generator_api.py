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
        prompt += "\n\n-출력은 위의 형식을 정확히 따르고, 반드시 JSON 형식으로 10문제를 연속 출력하시오. 다른 설명은 절대 포함하지 마세요."


    print("\n=== [DEBUG] 최종 프롬프트 내용 ===\n")
    print(prompt)
    print("\n==============================\n")

    if trace:
        trace.span(name="build_prompt", input=prompt)

    # LLM 호출
    print("모델 generate 시작")
    raw_output = generate_quiz(prompt)
    print("모델 응답 생성 완료")

    print("\n=== [DEBUG] LLM 응답 내용 ===\n")
    print(raw_output)
    print("\n==============================\n")

    if trace:
        trace.span(name="llm_response", input=prompt, output=raw_output)

    # 파싱 시도
    parsed_list = parse_response(raw_output)

    print("\n=== [DEBUG] 파싱된 리스트 ===")
    print(parsed_list)
    print("============================\n")

    quiz_items = [QuizItem(**item) for item in parsed_list]

    print(f"\n 전체 퀴즈 응답 반환 완료: 총 {len(quiz_items)}문항")

    return FollowupResponse(
        message="quiz_generated",
        data={
            "user_id": req.interview_id,
            "questions": quiz_items
        }
    )
