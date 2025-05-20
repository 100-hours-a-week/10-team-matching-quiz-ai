from fastapi import APIRouter, HTTPException, status
from app.api.question_generator.question_generator_schema import FollowupRequest, FollowupResponse
from app.api.question_generator.question_generator_model import call_llm, call_openai_api
from app.api.question_generator.question_generator_parser import parse_questions
import asyncio
from langfuse import Langfuse
import os
import logging
import uuid
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import sys

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
# vector_db 모듈이 존재하는지 확인
try:
    from app.vector_db.retriever import rag_retriever

    VECTOR_DB_AVAILABLE = True
    logging.info("Vector DB 모듈이 로드되었습니다.")
except ImportError:
    VECTOR_DB_AVAILABLE = False
    rag_retriever = None
    logging.warning("Vector DB 모듈을 찾을 수 없습니다. RAG 기능이 비활성화됩니다.")


# .env 파일 로드
load_dotenv()

router = APIRouter()
logger = logging.getLogger(__name__)

# Langfuse 클라이언트 초기화
langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

# 상수 정의
GENERATE_COUNT = 4  # 생성할 질문의 수
MAX_HISTORY_QUESTIONS = int(os.getenv("MAX_HISTORY_QUESTIONS", 20))


def validate_request(req: FollowupRequest) -> None:
    """입력 요청 유효성 검사"""
    if not req.selected_question or not req.selected_question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="메인 질문은 비워둘 수 없습니다.",
        )
    if not req.interview_id or not req.interview_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="유효한 interview_id가 필요합니다.",
        )


def prepare_context(req: FollowupRequest, trace) -> Dict[str, Any]:
    """요청 데이터로부터 프롬프트 컨텍스트 준비"""
    # 사용된 질문 목록 최대 20개로 제한
    all_used_questions = (req.passed_questions or [])[-MAX_HISTORY_QUESTIONS:]

    # 이전 질문 목록 형식화
    passed_section = ""
    if all_used_questions:
        joined = "\n".join(f"- {q}" for q in all_used_questions)
        passed_section = f"\n\n[이전 질문 목록]\n{joined}"

    retrieved_section = ""
    # RAG 스팬 생성 -> RAG을 추적하기 위해
    rag_span = trace.span(name="rag_retrieval")

    if VECTOR_DB_AVAILABLE and rag_retriever:
        try:
            rag_results = rag_retriever(
                req.selected_question, req.keyword or ""
            )
            rag_span.update(
                input={"query": req.selected_question, "keyword": req.keyword or ""},
                output={"results": rag_results},
            )
            retrieved_questions = [r["question"] for r in rag_results['results']]
            if retrieved_questions:
                joined_rag = "\n".join(f"- {q}" for q in retrieved_questions)
                retrieved_section = f"\n\n[유사한 기존 질문]\n{joined_rag}"
            rag_span.end()
        except Exception as e:
            logging.warning(f"RAG 검색 실패: {e}")
            rag_span.end(error=str(e))
            retrieved_section = ""  # fallback
    else:
        logging.info("Vector DB 모듈이 없어 RAG 검색을 건너뜁니다.")
        rag_span.update(
            input={"status": "skipped"}, output={"reason": "Vector DB 모듈이 없음"}
        )
        rag_span.end()

    return {
        "selected_question": req.selected_question,
        "keyword": req.keyword or "",
        "passed_questions": passed_section,
        "retrieved_questions": retrieved_section,
        "num_questions": GENERATE_COUNT,
    }


async def generate_primary_questions(prompt: str, trace) -> List[str]:
    """주 질문 생성 로직"""
    llm_span = trace.span(name="llm_call")

    try:
        raw_response = await call_llm(prompt, trace_id=trace.id)
        llm_span.update(input={"prompt": prompt}, output={"raw_response": raw_response})
        llm_span.end()

        parsing_span = trace.span(name="response_parsing")
        questions = parse_questions(raw_response)[:GENERATE_COUNT]
        parsing_span.update(
            input={"raw_response": raw_response}, output={"parsed_questions": questions}
        )
        parsing_span.end()

        return questions
    except Exception as e:
        if not llm_span.ended:
            llm_span.end(error=str(e))
        raise


async def generate_additional_questions(
    req: FollowupRequest,
    passed_section: str,
    remaining_count: int,
    trace,
    existing_questions: List[str],
) -> List[str]:
    """질문 개수가 부족할 시 추가질문을 생성"""
    context_api = {
        "selected_question": req.selected_question,
        "keyword": req.keyword or "",
        "passed_questions": passed_section,
        "ungenerated_questions_num": remaining_count,
    }

    prompt_template_api = langfuse.get_prompt("followup_questions_generator_api")
    prompt_api = prompt_template_api.compile(**context_api)

    llm_span_api = trace.span(name="open-api-call")

    try:
        raw_response_api = await call_openai_api(prompt_api, trace_id=trace.id)
        llm_span_api.update(
            input={"prompt_api": prompt_api},
            output={"raw_response_api": raw_response_api},
        )
        llm_span_api.end()

        parsing_span_api = trace.span(name="response_parsing_additional")
        additional_questions = parse_questions(raw_response_api)
        parsing_span_api.update(
            input={"raw_response_api": raw_response_api},
            output={"parsed_questions": additional_questions},
        )
        parsing_span_api.end()

        # 중복 제거 및 최대 개수까지만 포함
        unique_questions = [
            q for q in additional_questions if q not in existing_questions
        ]
        result = existing_questions.copy()
        result.extend(unique_questions)

        return result[:GENERATE_COUNT]
    except Exception as e:
        if not llm_span_api.ended:
            llm_span_api.end(error=str(e))
        raise


@router.post("/followup-questions", response_model=FollowupResponse)
async def generate_followup(req: FollowupRequest) -> FollowupResponse:
    logger.info(f"요청 받음: interview_id={req.interview_id}, req_data={req.dict()}")

    # 입력값 검증
    validate_request(req)

    # 트레이스 ID 생성 및 컨텍스트 준비
    # 고유의 질문을 추적하기 위해 uuid 사용
    trace_id = f"followup_{req.interview_id}_{uuid.uuid4().hex}"

    trace = langfuse.trace(
        id=trace_id,
        name="followup_generation_llm",
        input={
            "interview_id": req.interview_id,
            "selected_question": req.selected_question,
            "keyword": req.keyword,
            "passed_questions": req.passed_questions or [],
        },
    )

    context = prepare_context(req, trace)

    # 메인 프롬프트 컴파일
    prompt_template = langfuse.get_prompt(
        "followup_questions_generator"
    )  # prompt는 langfuse에서 따로 관리
    prompt = prompt_template.compile(**context)

    try:
        # 주 질문 생성
        generated_questions = await generate_primary_questions(prompt, trace)

        # 필요한 경우 추가 질문 생성
        if len(generated_questions) < GENERATE_COUNT:
            remaining_count = GENERATE_COUNT - len(generated_questions)
            generated_questions = await generate_additional_questions(
                req,
                context["passed_questions"],
                remaining_count,
                trace,
                generated_questions,
            )

        # 결과 기록
        trace.update(output={"followup_questions": generated_questions})
        # trace.end()
        logger.info(
            f"응답 반환: interview_id={req.interview_id}, questions_count={len(generated_questions)}"
        )

        return FollowupResponse(
            message="followup_questions_generated",
            interview_id=req.interview_id,
            followup_questions=generated_questions,
        )

    except Exception as e:
        trace.update(error={"message": str(e)})
        # trace.end()
        logger.error(f"꼬리 질문 생성 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"꼬리 질문 생성 실패: {str(e)}",
        )
