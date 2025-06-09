
from fastapi import APIRouter, HTTPException, status
import asyncio
import logging
import time
from app.api.question_generator.question_generator_schema import (
    FollowupRequest,
    FollowupResponse,
)
from app.api.question_generator.question_generator_model import (
    call_llm,
    call_openai_api,
)
from app.api.question_generator.question_generator_parser import parse_questions
from app.api.question_generator.question_generator_config import (
    LANGFUSE_CONFIG,
    API_CONFIG,
)
from langfuse import Langfuse
import os
import logging
import uuid
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

try:
    from app.vector_db.retriever import rag_retriever

    VECTOR_DB_AVAILABLE = True
    logging.info("Vector DB 모듈이 로드되었습니다.")
except ImportError:
    VECTOR_DB_AVAILABLE = False
    rag_retriever = None
    logging.warning("Vector DB 모듈을 찾을 수 없습니다. RAG 기능이 비활성화됩니다.")

router = APIRouter()
logger = logging.getLogger(__name__)

# 설정에서 Langfuse 초기화
langfuse = Langfuse(**LANGFUSE_CONFIG) if all(LANGFUSE_CONFIG.values()) else None

# 설정에서 API 구성 가져오기
GENERATE_COUNT = API_CONFIG["generate_count"]
MAX_HISTORY_QUESTIONS = API_CONFIG["max_history_questions"]

_prompt_cache = {}


def get_cached_prompt(prompt_name: str):
    """프롬프트를 캐시에서 가져오거나, 없으면 Langfuse에서 로드하여 캐시에 저장"""
    if prompt_name not in _prompt_cache:
        logger.info(f"프롬프트 캐시 미스: {prompt_name} - Langfuse에서 로드 중...")
        if langfuse:
            _prompt_cache[prompt_name] = langfuse.get_prompt(prompt_name)
        else:
            logger.warning("Langfuse가 설정되지 않아 프롬프트를 캐시할 수 없습니다.")
            return None
        logger.info(f"프롬프트 캐시 저장 완료: {prompt_name}")
    else:
        logger.debug(f"프롬프트 캐시 히트: {prompt_name}")

    return _prompt_cache[prompt_name]


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
    all_used_questions = (req.passed_questions or [])[-MAX_HISTORY_QUESTIONS:]

    passed_section = ""
    if all_used_questions:
        joined = "\n".join(f"- {q}" for q in all_used_questions)
        passed_section = f"\n\n[이전 질문 목록]\n{joined}"

    retrieved_section = ""
    rag_span = trace.span(name="rag_retrieval")
    rag_start_time = time.time()

    if VECTOR_DB_AVAILABLE and rag_retriever:
        try:
            rag_results = rag_retriever(req.selected_question, req.keyword or "")
            rag_execution_time = time.time() - rag_start_time
            rag_span.update(
                input={"query": req.selected_question, "keyword": req.keyword or ""},
                output={"results": rag_results},
                metadata={"execution_time_seconds": rag_execution_time},
            )
            retrieved_questions = [r["question"] for r in rag_results["results"]]
            if retrieved_questions:
                joined_rag = "\n".join(f"- {q}" for q in retrieved_questions)
                retrieved_section = f"\n\n[유사한 기존 질문]\n{joined_rag}"
            rag_span.end()
        except Exception as e:
            rag_execution_time = time.time() - rag_start_time
            logging.warning(f"RAG 검색 실패: {e}")
            rag_span.end(
                error=str(e), metadata={"execution_time_seconds": rag_execution_time}
            )
            retrieved_section = ""  # fallback
    else:
        rag_execution_time = time.time() - rag_start_time
        logging.info("Vector DB 모듈이 없어 RAG 검색을 건너뜁니다.")
        rag_span.update(
            input={"status": "skipped"},
            output={"reason": "Vector DB 모듈이 없음"},
            metadata={"execution_time_seconds": rag_execution_time},
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
    llm_start_time = time.time()

    try:
        raw_response = await call_llm(prompt, trace_id=trace.id)
        llm_execution_time = time.time() - llm_start_time
        llm_span.update(
            input={"prompt": prompt},
            output={"raw_response": raw_response},
            metadata={"execution_time_seconds": llm_execution_time},
        )
        llm_span.end()

        parsing_span = trace.span(name="response_parsing")
        parsing_start_time = time.time()
        questions = parse_questions(raw_response)[:GENERATE_COUNT]
        parsing_execution_time = time.time() - parsing_start_time
        parsing_span.update(
            input={"raw_response": raw_response},
            output={"parsed_questions": questions},
            metadata={"execution_time_seconds": parsing_execution_time},
        )
        parsing_span.end()

        return questions
    except Exception as e:
        llm_execution_time = time.time() - llm_start_time
        if not llm_span.ended:
            llm_span.end(
                error=str(e), metadata={"execution_time_seconds": llm_execution_time}
            )
        raise


async def generate_additional_questions(
    req: FollowupRequest,
    passed_section: str,
    remaining_count: int,
    trace,
    existing_questions: List[str],
) -> List[str]:
    """질문 개수가 부족할 시 추가질문을 생성"""
    logger.info(
        f"OpenAI API로 추가 질문 생성 시작: interview_id={req.interview_id}, remaining_count={remaining_count}"
    )

    context_api = {
        "selected_question": req.selected_question,
        "keyword": req.keyword or "",
        "passed_questions": passed_section,
        "ungenerated_questions_num": remaining_count,
    }

    prompt_template_api = get_cached_prompt("followup_questions_generator_api")
    prompt_api = prompt_template_api.compile(**context_api)

    llm_span_api = trace.span(name="open-api-call")
    llm_api_start_time = time.time()

    try:
        raw_response_api = await call_openai_api(prompt_api, trace_id=trace.id)
        llm_api_execution_time = time.time() - llm_api_start_time
        llm_span_api.update(
            input={"prompt_api": prompt_api},
            output={"raw_response_api": raw_response_api},
            metadata={"execution_time_seconds": llm_api_execution_time},
        )
        llm_span_api.end()

        parsing_span_api = trace.span(name="response_parsing_additional")
        parsing_api_start_time = time.time()
        additional_questions = parse_questions(raw_response_api)
        parsing_api_execution_time = time.time() - parsing_api_start_time
        parsing_span_api.update(
            input={"raw_response_api": raw_response_api},
            output={"parsed_questions": additional_questions},
            metadata={"execution_time_seconds": parsing_api_execution_time},
        )
        parsing_span_api.end()

        # 중복 제거 및 최대 개수까지만 포함
        unique_questions = [
            q for q in additional_questions if q not in existing_questions
        ]
        result = existing_questions.copy()
        result.extend(unique_questions)

        logger.info(
            f"OpenAI API로 추가 질문 생성 완료: interview_id={req.interview_id}, generated_count={len(additional_questions)}, unique_count={len(unique_questions)}"
        )

        return result[:GENERATE_COUNT]
    except Exception as e:
        llm_api_execution_time = time.time() - llm_api_start_time
        logger.error(
            f"OpenAI API 추가 질문 생성 실패: interview_id={req.interview_id}, error={str(e)}"
        )
        if not llm_span_api.ended:
            llm_span_api.end(
                error=str(e),
                metadata={"execution_time_seconds": llm_api_execution_time},
            )
        raise


@router.post("/followup-questions", response_model=FollowupResponse)
async def generate_followup(req: FollowupRequest) -> FollowupResponse:
    # Check model availability (import inside function to avoid circular import)
    from app.main import is_model_available

    if not is_model_available("question_generator"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="질문 생성 모델이 현재 사용할 수 없습니다. 모델이 초기화되지 않았거나 오류가 발생했습니다.",
        )

    logger.info(f"요청 받음: interview_id={req.interview_id}, req_data={req.dict()}")

    validate_request(req)

    trace_id = f"followup_{req.interview_id}_{uuid.uuid4().hex}"
    request_start_time = time.time()

    trace = None
    if langfuse:
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

    prompt_template = get_cached_prompt("followup_questions_generator")
    if prompt_template:
        prompt = prompt_template.compile(**context)
    else:
        # 기본 프롬프트 사용
        prompt = create_default_followup_prompt(context)

    try:
        generated_questions = await generate_primary_questions(prompt, trace)
        if len(generated_questions) < GENERATE_COUNT:
            remaining_count = GENERATE_COUNT - len(generated_questions)
            generated_questions = await generate_additional_questions(
                req,
                context["passed_questions"],
                remaining_count,
                trace,
                generated_questions,
            )

        request_execution_time = time.time() - request_start_time
        trace.update(
            output={"followup_questions": generated_questions},
            metadata={"total_execution_time_seconds": request_execution_time},
        )
        logger.info(
            f"응답 반환: interview_id={req.interview_id}, questions_count={len(generated_questions)}, execution_time={request_execution_time:.2f}초"
        )

        return FollowupResponse(
            message="followup_questions_generated",
            interview_id=req.interview_id,
            followup_questions=generated_questions,
        )

    except Exception as e:
        request_execution_time = time.time() - request_start_time
        trace.update(
            error={"message": str(e)},
            metadata={"total_execution_time_seconds": request_execution_time},
        )
        logger.error(f"꼬리 질문 생성 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"꼬리 질문 생성 실패: {str(e)}",
        )


@router.post("/open-api-server", response_model=FollowupResponse)
async def generate_followup_openai(req: FollowupRequest) -> FollowupResponse:
    # Check OpenAI API availability (this endpoint doesn't require local models)
    logger.info(f"요청 받음: interview_id={req.interview_id}, req_data={req.dict()}")

    validate_request(req)

    trace_id = f"followup_{req.interview_id}{uuid.uuid4().hex}"
    request_start_time = time.time()

    trace = None
    if langfuse:
        trace = langfuse.trace(
            id=trace_id,
            name="followup_generation_api",  # 이름 변경으로 구분
            input={
                "interview_id": req.interview_id,
                "selected_question": req.selected_question,
                "keyword": req.keyword,
                "passed_questions": req.passed_questions or [],
            },
        )

    try:
        logger.info(
            f"OpenAI API를 통한 질문 생성 시작: interview_id={req.interview_id}"
        )

        context = prepare_context(req, trace)

        context_api = {
            "selected_question": req.selected_question,
            "keyword": req.keyword or "",
            "passed_questions": context["passed_questions"],
            "num_questions": GENERATE_COUNT,
        }

        prompt_template_api = get_cached_prompt("followup_questions_generator_api")
        prompt_api = prompt_template_api.compile(**context_api)

        llm_span_api = trace.span(name="openai_api_call")
        llm_api_start_time = time.time()

        try:
            raw_response_api = await call_openai_api(prompt_api, trace_id=trace.id)
            llm_api_execution_time = time.time() - llm_api_start_time
            llm_span_api.update(
                input={"prompt_api": prompt_api},
                output={"raw_response_api": raw_response_api},
                metadata={"execution_time_seconds": llm_api_execution_time},
            )
            llm_span_api.end()

            parsing_span_api = trace.span(name="response_parsing_api")
            parsing_api_start_time = time.time()
            generated_questions = parse_questions(raw_response_api)[:GENERATE_COUNT]
            parsing_api_execution_time = time.time() - parsing_api_start_time
            parsing_span_api.update(
                input={"raw_response_api": raw_response_api},
                output={"parsed_questions": generated_questions},
                metadata={"execution_time_seconds": parsing_api_execution_time},
            )
            parsing_span_api.end()

            logger.info(
                f"OpenAI API 질문 생성 완료: interview_id={req.interview_id}, generated_count={len(generated_questions)}"
            )

        except Exception as e:
            llm_api_execution_time = time.time() - llm_api_start_time
            logger.error(
                f"OpenAI API 질문 생성 실패: interview_id={req.interview_id}, error={str(e)}"
            )
            if not llm_span_api.ended:
                llm_span_api.end(
                    error=str(e),
                    metadata={"execution_time_seconds": llm_api_execution_time},
                )
            raise

        request_execution_time = time.time() - request_start_time
        trace.update(
            output={"followup_questions": generated_questions},
            metadata={"total_execution_time_seconds": request_execution_time},
        )
        logger.info(
            f"응답 반환: interview_id={req.interview_id}, questions_count={len(generated_questions)}, execution_time={request_execution_time:.2f}초"
        )

        return FollowupResponse(
            message="followup_questions_generated",
            interview_id=req.interview_id,
            followup_questions=generated_questions,
        )

    except Exception as e:
        request_execution_time = time.time() - request_start_time
        trace.update(
            error={"message": str(e)},
            metadata={"total_execution_time_seconds": request_execution_time},
        )
        logger.error(f"꼬리 질문 생성 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"꼬리 질문 생성 실패: {str(e)}",
        )


def create_default_followup_prompt(context: Dict[str, Any]) -> str:
    """Langfuse가 없을 때 사용할 기본 꼬리질문 프롬프트"""
    return f"""당신은 IT 기술 면접관입니다. 다음 맥락을 바탕으로 꼬리질문 4개를 생성해주세요.

선택된 질문: {context.get('selected_question', '')}
키워드: {context.get('keyword', '')}
이전 질문들: {', '.join(context.get('passed_questions', []))}

요구사항:
1. 키워드 "{context.get('keyword', '')}"와 관련된 꼬리질문 4개를 생성하세요
2. 각 질문은 "질문 1.", "질문 2." 형식으로 시작하세요
3. 이전 질문과 중복되지 않도록 하세요
4. 기술적 깊이를 가진 구체적인 질문을 만드세요

질문 1. 
질문 2. 
질문 3. 
질문 4. """
