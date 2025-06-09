from fastapi import APIRouter, HTTPException
from app.api.quiz_generator.quiz_generator_schema import (
    FollowupRequest,
    FollowupResponse,
    QuizItem,
)
from app.api.quiz_generator.quiz_generator_parser import (
    parse_response,
    filter_and_select_quizzes,
    remove_prompt_content,
)
from app.api.quiz_generator.quiz_generator_model import generate_quiz
from app.api.quiz_generator.quiz_generator_config import (
    QUIZ_LANGFUSE_SECRET_KEY,
    QUIZ_LANGFUSE_PUBLIC_KEY,
    QUIZ_LANGFUSE_HOST,
)
from langfuse import Langfuse
from app.vector_db.retriever import quiz_rag_retriever
import logging
import time

router = APIRouter()
logger = logging.getLogger(__name__)

# Langfuse 클라이언트 초기화
langfuse = Langfuse(
    secret_key=QUIZ_LANGFUSE_SECRET_KEY,
    public_key=QUIZ_LANGFUSE_PUBLIC_KEY,
    host=QUIZ_LANGFUSE_HOST,
)


@router.post("/generate-quiz", response_model=FollowupResponse)
def generate_quiz_api(req: FollowupRequest):
    logger.info("퀴즈 생성 요청 수신: /generate-quiz")
    request_start_time = time.time()

    # 모델 사용 가능 여부 체크 (import inside function to avoid circular import)
    from app.main import is_model_available

    if not is_model_available("quiz_generator"):
        logger.error("quiz_generator 모델이 사용 불가능합니다.")
        raise HTTPException(
            status_code=503,
            detail="퀴즈 생성 모델이 현재 사용할 수 없습니다. 잠시 후 다시 시도해주세요.",
        )

    try:
        trace = langfuse.trace(
            name="quiz_generation",
            tags=["quiz", "generate"],
            input={"question_list": req.question_history_list},
            metadata={"endpoint": "/generate-quiz"},
        )
    except Exception as e:
        print(f"[WARN] Langfuse trace 시작 실패: {e}")
        trace = None

    prompt_template = langfuse.get_prompt("quiz_generation")

    # RAG 검색 시간 추적
    rag_span = trace.span(name="rag_retrieval") if trace else None
    rag_start_time = time.time()

    quiz_rag = quiz_rag_retriever(req.question_history_list)

    # RAG 결과에서 관련 질문들 추출
    related_questions = []
    for rag_result in quiz_rag:
        if rag_result.get("result"):
            for doc in rag_result["result"]:
                if "content" in doc:
                    related_questions.append(doc["content"])

    rag_execution_time = time.time() - rag_start_time

    if rag_span:
        rag_span.update(
            input={"rag_input": req.question_history_list},
            output={"rag_output": quiz_rag},
            metadata={"execution_time_seconds": rag_execution_time},
        )
        rag_span.end()

    joined_questions = "\n".join(req.question_history_list)
    print(joined_questions)
    related_questions_text = (
        "\n".join(related_questions) if related_questions else "관련 문서 없음"
    )

    context_api = {
        "joined": joined_questions,
        "related_questions": related_questions_text,
    }

    if hasattr(prompt_template, "compile"):
        prompt = prompt_template.compile(**context_api)
    else:
        prompt_text = str(prompt_template)
        prompt = prompt_text.replace("{{joined_questions}}", joined_questions)
        prompt = prompt.replace("{{related_questions}}", related_questions_text)
    prompt += (
        "\n\n출력 지침:\n"
        "- 출력은 반드시 15~20개의 4지선다형 문제로 구성되어야 합니다.\n"
        "- 각 문제는 아래와 같은 형식을 따릅니다:\n"
        "  난이도: 하 | 중 | 상\n"
        "  문제: (자연어 문장)\n"
        "  선지: [보기1, 보기2, 보기3, 보기4]  # JSON 배열 형태\n"
        "  정답 인덱스: 1~4 사이의 숫자\n"
        "  해설: 정답의 이유나 부가 설명\n"
        "- 출력에는 위 문제 형식만 포함하고, 지시사항이나 설명 문구는 포함하지 마세요.\n"
        "- JSON 포맷은 사용하지 않고, 자유 포맷의 텍스트로 출력하세요.\n"
        "- 각 문제는 줄바꿈을 포함해 구분되도록 출력할 것.\n"
        "--- END OF INSTRUCTION ---"
    )

    # prompt 생성 span
    prompt_span = trace.span(name="build_prompt") if trace else None
    prompt_start_time = time.time()

    if prompt_span:
        prompt_execution_time = time.time() - prompt_start_time
        prompt_span.update(
            input={"prompt": prompt},
            metadata={"execution_time_seconds": prompt_execution_time},
        )
        prompt_span.end()

    # LLM 호출 시간 추적
    llm_span = trace.span(name="llm_generation") if trace else None
    llm_start_time = time.time()

    print("quiz generate 시작")
    raw_output = generate_quiz(prompt, use_chat_template=True)
    print("quiz 생성 완료")

    llm_execution_time = time.time() - llm_start_time

    if llm_span:
        llm_span.update(
            input={"prompt": prompt},
            output={"raw_output": raw_output},
            metadata={"execution_time_seconds": llm_execution_time},
        )
        llm_span.end()

    # 프롬프트 내용 제거
    cleaned_output = remove_prompt_content(raw_output)

    # 디버깅 출력 추가
    print("[CLEANED OUTPUT PREVIEW]")
    print(cleaned_output[:200])

    # 파싱 시간 추적
    parsing_span = trace.span(name="response_parsing") if trace else None
    parsing_start_time = time.time()

    parsed_list = parse_response(cleaned_output)

    parsing_execution_time = time.time() - parsing_start_time

    if not parsed_list:
        if parsing_span:
            parsing_span.end(
                error="형식에 맞는 퀴즈를 하나도 파싱하지 못했습니다.",
                metadata={"execution_time_seconds": parsing_execution_time},
            )
        if trace:
            trace.span(name="parsing_error", input=cleaned_output).update(
                status="error"
            )
        raise ValueError("형식에 맞는 퀴즈를 하나도 파싱하지 못했습니다.")

    if parsing_span:
        parsing_span.update(
            input={"raw_output": cleaned_output},
            output={"parsed_questions": parsed_list},
            metadata={"execution_time_seconds": parsing_execution_time},
        )
        parsing_span.end()

    # 먼저 전체 형식이 맞는 퀴즈 수만 체크
    print(f"[DEBUG] 총 형식이 맞는 문제 수: {len(parsed_list)}개")

    # 필터링 시간 추적
    filtering_span = trace.span(name="difficulty_filtering") if trace else None
    filtering_start_time = time.time()

    # 난이도별 필터링 
    final_quizzes = filter_and_select_quizzes(parsed_list)

    filtering_execution_time = time.time() - filtering_start_time

    # 난이도별 충족 안 될 때만 에러
    if len(final_quizzes) != 10:
        # 각 난이도 개수 확인
        easy = len([q for q in parsed_list if q["difficulty"] == "하"])
        medium = len([q for q in parsed_list if q["difficulty"] == "중"])
        hard = len([q for q in parsed_list if q["difficulty"] == "상"])

        if filtering_span:
            filtering_span.end(
                error=f"난이도별 문제 부족 - 하:{easy}, 중:{medium}, 상:{hard}",
                metadata={
                    "execution_time_seconds": filtering_execution_time,
                    "reason": f"난이도별 문제 부족",
                    "하": easy,
                    "중": medium,
                    "상": hard,
                },
            )
        if trace:
            trace.span(name="filtering_error", input=str(parsed_list)).update(
                status="error",
                metadata={
                    "reason": f"난이도별 문제 부족",
                    "하": easy,
                    "중": medium,
                    "상": hard,
                },
            )
        raise ValueError(f"난이도별 문제 부족 - 하:{easy}, 중:{medium}, 상:{hard}")

    if filtering_span:
        filtering_span.update(
            input={"parsed_questions": parsed_list},
            output={"filtered_questions": final_quizzes},
            metadata={"execution_time_seconds": filtering_execution_time},
        )
        filtering_span.end()

    quiz_items = [QuizItem(**item) for item in final_quizzes]

    print(f"\n 전체 quiz 응답 수: 총 {len(quiz_items)}문항")

    request_execution_time = time.time() - request_start_time

    if trace:
        trace.update(
            output={"parsed_output": parsed_list, "final_questions": final_quizzes},
            metadata={"total_execution_time_seconds": request_execution_time},
        )

    logger.info(
        f"퀴즈 생성 완료: questions_count={len(quiz_items)}, execution_time={request_execution_time:.2f}초"
    )

    return FollowupResponse(
        message="quiz_generated",
        data={"user_id": req.interview_id, "questions": quiz_items},
    )