from fastapi import APIRouter, HTTPException, status
from schemas import FollowupRequest, FollowupResponse
from services.llm_service import call_llm, call_openai_api
from services.backend_client import push_to_backend
from utils.parser import parse_questions
import asyncio
from langfuse import Langfuse
import os
import logging
import uuid
from typing import List, Dict, Any, Optional

router = APIRouter()
logger = logging.getLogger(__name__)

# Langfuse 클라이언트 초기화
langfuse = Langfuse(
    secret_key=os.getenv('LANGFUSE_SECRET_KEY'),
    public_key=os.getenv('LANGFUSE_PUBLIC_KEY'),
    host=os.getenv('LANGFUSE_HOST')
)

# 상수 정의
GENERATE_COUNT = 4
MAX_HISTORY_QUESTIONS = 20


async def safe_push(interview_id: int, questions: list[str]) -> None:
    """
    백엔드 전송 중 예외를 잡아서 로깅하는 안전 래퍼
    """
    try:
        await push_to_backend(interview_id, questions)
    except Exception as e:
        logger.error(f"백엔드 전송 실패 (interview_id={interview_id}): {e}")


def validate_request(req: FollowupRequest) -> None:
    """입력 요청 유효성 검사"""
    if not req.selected_question or not req.selected_question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="메인 질문은 비워둘 수 없습니다."
        )
    if req.interview_id <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="유효한 interview_id가 필요합니다."
        )


def prepare_context(req: FollowupRequest) -> Dict[str, Any]:
    """요청 데이터로부터 프롬프트 컨텍스트 준비"""
    # 사용된 질문 목록 최대 20개로 제한
    all_used_questions = (req.passed_questions or [])[-MAX_HISTORY_QUESTIONS:]

    # 이전 질문 목록 형식화
    passed_section = ""
    if all_used_questions:
        joined = "\n".join(f"- {q}" for q in all_used_questions)
        passed_section = f"\n\n[이전 질문 목록]\n{joined}"

    return {
        "selected_question": req.selected_question,
        "keyword": req.keyword or "",
        "passed_questions": passed_section,
        "num_questions": GENERATE_COUNT,
    }


async def generate_primary_questions(prompt: str, trace) -> List[str]:
    """주 질문 생성 로직"""
    llm_span = trace.span(name="llm_call")

    try:
        raw_response = await call_llm(prompt)
        llm_span.update(input={"prompt": prompt}, output={
                        "raw_response": raw_response})
        llm_span.end()

        parsing_span = trace.span(name="response_parsing")
        questions = parse_questions(raw_response)[:GENERATE_COUNT]
        parsing_span.update(
            input={"raw_response": raw_response},
            output={"parsed_questions": questions}
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
    existing_questions: List[str]
) -> List[str]:
    """추가 질문 생성 로직"""
    context_api = {
        "selected_question": req.selected_question,
        "keyword": req.keyword or "",
        "passed_questions": passed_section,
        "ungenerated_questions_num": remaining_count
    }

    prompt_template_api = langfuse.get_prompt(
        "followup_questions_generator_api")
    prompt_api = prompt_template_api.compile(**context_api)

    llm_span_api = trace.span(name="llm_call_additional")

    try:
        raw_response_api = await call_openai_api(prompt_api)
        llm_span_api.update(
            input={"prompt_api": prompt_api},
            output={"raw_response_api": raw_response_api}
        )
        llm_span_api.end()

        parsing_span_api = trace.span(name="response_parsing_additional")
        additional_questions = parse_questions(raw_response_api)
        parsing_span_api.update(
            input={"raw_response_api": raw_response_api},
            output={"parsed_questions": additional_questions}
        )
        parsing_span_api.end()

        # 중복 제거 및 최대 개수까지만 포함
        unique_questions = [
            q for q in additional_questions if q not in existing_questions]
        result = existing_questions.copy()
        result.extend(unique_questions)

        return result[:GENERATE_COUNT]
    except Exception as e:
        if not llm_span_api.ended:
            llm_span_api.end(error=str(e))
        raise


@router.post("/interview/followup-questions", response_model=FollowupResponse)
async def generate_followup(req: FollowupRequest) -> FollowupResponse:
    """
    IT 면접을 위한 새로운 꼬리 질문을 생성하는 엔드포인트 핸들러입니다.
    """
    # 입력값 검증
    validate_request(req)

    # 트레이스 ID 생성 및 컨텍스트 준비
    trace_id = f"followup_{req.interview_id}_{uuid.uuid4().hex}"
    context = prepare_context(req)

    # 메인 프롬프트 컴파일
    prompt_template = langfuse.get_prompt("followup_questions_generator")
    prompt = prompt_template.compile(**context)

    trace = langfuse.trace(
        id=trace_id,
        name="followup_generation_llm",
        input={
            "interview_id": req.interview_id,
            "selected_question": req.selected_question,
            "keyword": req.keyword,
            "passed_questions": context["passed_questions"],
        }
    )

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
                generated_questions
            )

        # 결과 기록 및 백엔드 전송 태스크 생성
        trace.update(output={"followup_questions": generated_questions})
        trace.end()

        asyncio.create_task(safe_push(req.interview_id, generated_questions))

        return FollowupResponse(
            message="followup_questions_generated",
            interview_id=req.interview_id,
            followup_questions=generated_questions
        )

    except Exception as e:
        trace.update(error={"message": str(e)})
        trace.end()
        logger.error(f"꼬리 질문 생성 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"처리 실패: {str(e)}"
        )
