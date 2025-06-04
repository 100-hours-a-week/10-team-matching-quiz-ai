from fastapi import APIRouter
from app.api.quiz_generator.quiz_generator_schema import (
    FollowupRequest,
    FollowupResponse,
    QuizItem
)
from app.api.quiz_generator.quiz_generator_parser import (
    parse_response,
    filter_and_select_quizzes,
    remove_prompt_content
)
from app.api.quiz_generator.quiz_generator_model import generate_quiz
from app.api.quiz_generator.quiz_generator_config import (
    QUIZ_LANGFUSE_SECRET_KEY,
    QUIZ_LANGFUSE_PUBLIC_KEY,
    QUIZ_LANGFUSE_HOST
)
from langfuse import Langfuse
from app.vector_db.retriever import quiz_rag_retriever

router = APIRouter()

# Langfuse 클라이언트 초기화
langfuse = Langfuse(
    secret_key=QUIZ_LANGFUSE_SECRET_KEY,
    public_key=QUIZ_LANGFUSE_PUBLIC_KEY,
    host=QUIZ_LANGFUSE_HOST
)


@router.post("/generate_quiz", response_model=FollowupResponse)
def generate_quiz_api(req: FollowupRequest):
    print(" 요청 수신: /generate_quiz")

    try:
        trace = langfuse.trace(
            name="quiz_generation",
            tags=["quiz", "generate"],
            input={"question_list": req.question_history_list},
            metadata={"endpoint": "/generate_quiz"},
        )
    except Exception as e:
        print(f"[WARN] Langfuse trace 시작 실패: {e}")
        trace = None

    prompt_template = langfuse.get_prompt("quiz_generation")
    quiz_rag = quiz_rag_retriever(req.question_history_list)

    # RAG 결과에서 관련 질문들 추출
    related_questions = []
    for rag_result in quiz_rag:
        if rag_result.get("result"):
            for doc in rag_result["result"]:
                if "content" in doc:
                    related_questions.append(doc["content"])

    joined_questions = "\n".join(req.question_history_list)
    print(joined_questions)
    related_questions_text = (
        "\n".join(related_questions) if related_questions else "관련 문서 없음"
    )

    if trace:
        rag_trace = trace.span(name="rag_trace")
        rag_trace.update(
            input={"rag_input": req.question_history_list},
            output={"rag_output": quiz_rag},
        )
        rag_trace.end()

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
        "  난이도 : 하 | 중 | 상\n"
        "  문제 : (자연어 문장)\n"
        "  선지 : [보기1, 보기2, 보기3, 보기4]  # JSON 배열 형태\n"
        "  정답 인덱스 : 1~4 사이의 숫자\n"
        "  해설 : 정답의 이유나 부가 설명\n"
        "- 출력에는 위 문제 형식만 포함하고, 지시사항이나 설명 문구는 포함하지 마세요.\n"
        "- JSON 포맷은 사용하지 않고, 자유 포맷의 텍스트로 출력하세요.\n"
        "- 각 문제는 줄바꿈을 포함해 구분되도록 출력할 것.\n"
        "--- END OF INSTRUCTION ---"
    )


    # prompt 생성 span
    if trace:
        span_prompt = trace.span(name="build_prompt")
        span_prompt.update(input={"prompt": prompt})
        span_prompt.end()

    # LLM 호출
    print("quiz generate 시작")
    raw_output = generate_quiz(prompt)
    print("quiz 생성 완료")

    # 프롬프트 내용 제거
    cleaned_output = remove_prompt_content(raw_output)
    
    # 디버깅 출력 추가
    print("[CLEANED OUTPUT PREVIEW]")
    print(cleaned_output[:200]) 

    # LLM 응답 처리 span
    if trace:
        span_llm = trace.span(name="llm_response")
        span_llm.update(input={"prompt": prompt}, output={"raw_output": raw_output})
        span_llm.end()

    # 파싱 시도
    if trace:
        parsed_llm = trace.span(name="parsed_question")
    else:
        parsed_llm = None

    parsed_list = parse_response(cleaned_output)
    if not parsed_list:
        if trace:
            trace.span(name="parsing_error", input=cleaned_output).update(status="error")
        raise ValueError("형식에 맞는 퀴즈를 하나도 파싱하지 못했습니다.")

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
                    "상": hard,
                },
            )
        raise ValueError(f"난이도별 문제 부족 - 하:{easy}, 중:{medium}, 상:{hard}")

    quiz_items = [QuizItem(**item) for item in final_quizzes]

    print(f"\n 전체 quiz 응답 수: 총 {len(quiz_items)}문항")

    if parsed_llm:
        parsed_llm.update(
            input={"raw_output": raw_output}, output={"parsed_output": parsed_list}
        )
        parsed_llm.end()

    if trace:
        trace.update(output={"parsed_output": parsed_list})

    return FollowupResponse(
        message="quiz_generated",
        data={"user_id": req.interview_id, "questions": quiz_items},
    )
