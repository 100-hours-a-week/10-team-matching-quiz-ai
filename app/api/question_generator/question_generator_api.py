from fastapi import APIRouter, HTTPException, status
from app.api.question_generator.question_generator_schema import (
    FollowupRequest,
    FollowupResponse,
)
from app.api.question_generator.question_generator_model import (
    call_llm,
    call_openai_api,
)
from app.api.question_generator.question_generator_parser import parse_questions
import asyncio
from langfuse import Langfuse
import os
import logging
import uuid
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

langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

GENERATE_COUNT = 4
MAX_HISTORY_QUESTIONS = int(os.getenv("MAX_HISTORY_QUESTIONS", 20))

_prompt_cache = {}

def get_cached_prompt(prompt_name: str):
    """프롬프트를 캐시에서 가져오거나, 없으면 Langfuse에서 로드하여 캐시에 저장"""
    if prompt_name not in _prompt_cache:
        logger.info(f"프롬프트 캐시 미스: {prompt_name} - Langfuse에서 로드 중...")
        _prompt_cache[prompt_name] = langfuse.get_prompt(prompt_name)
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

    if VECTOR_DB_AVAILABLE and rag_retriever:
        try:
            rag_results = rag_retriever(req.selected_question, req.keyword or "")
            rag_span.update(
                input={"query": req.selected_question, "keyword": req.keyword or ""},
                output={"results": rag_results},
            )
            retrieved_questions = [r["question"] for r in rag_results["results"]]
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
    logger.info(f"OpenAI API로 추가 질문 생성 시작: interview_id={req.interview_id}, remaining_count={remaining_count}")
    
    context_api = {
        "selected_question": req.selected_question,
        "keyword": req.keyword or "",
        "passed_questions": passed_section,
        "ungenerated_questions_num": remaining_count,
    }

    prompt_template_api = get_cached_prompt("followup_questions_generator_api")
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

        logger.info(f"OpenAI API로 추가 질문 생성 완료: interview_id={req.interview_id}, generated_count={len(additional_questions)}, unique_count={len(unique_questions)}")

        return result[:GENERATE_COUNT]
    except Exception as e:
        logger.error(f"OpenAI API 추가 질문 생성 실패: interview_id={req.interview_id}, error={str(e)}")
        if not llm_span_api.ended:
            llm_span_api.end(error=str(e))
        raise


@router.post("/followup-questions", response_model=FollowupResponse)
async def generate_followup(req: FollowupRequest) -> FollowupResponse:
    logger.info(f"요청 받음: interview_id={req.interview_id}, req_data={req.dict()}")

    validate_request(req)

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

    prompt_template = get_cached_prompt("followup_questions_generator")
    prompt = prompt_template.compile(**context)

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

        trace.update(output={"followup_questions": generated_questions})
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
        logger.error(f"꼬리 질문 생성 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"꼬리 질문 생성 실패: {str(e)}",
        )


@router.post("/open-api-server", response_model=FollowupResponse)
async def generate_followup(req: FollowupRequest) -> FollowupResponse:
    logger.info(f"요청 받음: interview_id={req.interview_id}, req_data={req.dict()}")

    validate_request(req)

    trace_id = f"followup_{req.interview_id}{uuid.uuid4().hex}"

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
        logger.info(f"OpenAI API를 통한 질문 생성 시작: interview_id={req.interview_id}")
        
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

        try:
            raw_response_api = await call_openai_api(prompt_api, trace_id=trace.id)
            llm_span_api.update(
                input={"prompt_api": prompt_api},
                output={"raw_response_api": raw_response_api},
            )
            llm_span_api.end()

            parsing_span_api = trace.span(name="response_parsing_api")
            generated_questions = parse_questions(raw_response_api)[:GENERATE_COUNT]
            parsing_span_api.update(
                input={"raw_response_api": raw_response_api},
                output={"parsed_questions": generated_questions},
            )
            parsing_span_api.end()

            logger.info(
                f"OpenAI API 질문 생성 완료: interview_id={req.interview_id}, generated_count={len(generated_questions)}"
            )

        except Exception as e:
            logger.error(f"OpenAI API 질문 생성 실패: interview_id={req.interview_id}, error={str(e)}")
            if not llm_span_api.ended:
                llm_span_api.end(error=str(e))
            raise

        trace.update(output={"followup_questions": generated_questions})
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
        logger.error(f"꼬리 질문 생성 실패: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"꼬리 질문 생성 실패: {str(e)}",
        )