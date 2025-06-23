from fastapi import APIRouter, HTTPException, status
import asyncio
import logging
import time
import os
import uuid
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

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
    VLLM_API_CONFIG,
)
from langfuse import Langfuse

load_dotenv()

try:
    from app.vector_db.retriever import question_rag_retriever
    VECTOR_DB_AVAILABLE = True
    logging.info("Vector DB 모듈이 로드되었습니다.")
except ImportError:
    VECTOR_DB_AVAILABLE = False
    question_rag_retriever = None
    logging.warning("Vector DB 모듈을 찾을 수 없습니다. RAG 기능이 비활성화됩니다.")

router = APIRouter()
logger = logging.getLogger(__name__)

# 설정에서 Langfuse 초기화
langfuse = Langfuse(**LANGFUSE_CONFIG) 

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


def create_default_followup_prompt(context: Dict[str, Any]) -> str:
    """기본 프롬프트 생성 (Langfuse가 없을 때 사용)"""
    return f"""다음 조건에 맞는 IT 기술 면접 꼬리 질문을 {context['num_questions']}개 생성해주세요.

[메인 질문]
{context['selected_question']}

[키워드]
{context['keyword']}

{context['passed_questions']}

{context['retrieved_questions']}

**요구사항:**
1. 메인 질문과 연관된 심화 질문을 생성하세요
2. 각 질문은 "질문 1.", "질문 2." 형식으로 시작하세요
3. 실무 경험을 확인할 수 있는 질문을 포함하세요
4. 이전 질문과 중복되지 않도록 하세요

질문을 생성해주세요:"""


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


def prepare_context(req: FollowupRequest, trace=None) -> Dict[str, Any]:
    """요청 데이터로부터 프롬프트 컨텍스트 준비"""
    all_used_questions = (req.passed_questions or [])[-MAX_HISTORY_QUESTIONS:]

    passed_section = ""
    if all_used_questions:
        joined = "\n".join(f"- {q}" for q in all_used_questions)
        passed_section = f"\n\n[이전 질문 목록]\n{joined}"

    retrieved_section = ""
    rag_span = None
    
    if trace:
        rag_span = trace.span(name="rag_retrieval")
    
    rag_start_time = time.time()

    if VECTOR_DB_AVAILABLE and question_rag_retriever:
        try:
            rag_results = question_rag_retriever(req.selected_question, req.keyword or "")
            rag_execution_time = time.time() - rag_start_time
            
            if rag_span:
                rag_span.update(
                    input={"query": req.selected_question, "keyword": req.keyword or ""},
                    output={"results": rag_results},
                    metadata={"execution_time_seconds": rag_execution_time},
                )
                rag_span.end()
                
            retrieved_questions = [r["question"] for r in rag_results["results"]]
            if retrieved_questions:
                joined_rag = "\n".join(f"- {q}" for q in retrieved_questions)
                retrieved_section = f"\n\n[유사한 기존 질문]\n{joined_rag}"
        except Exception as e:
            rag_execution_time = time.time() - rag_start_time
            logging.warning(f"RAG 검색 실패: {e}")
            
            if rag_span:
                rag_span.end(
                    error=str(e), 
                    metadata={"execution_time_seconds": rag_execution_time}
                )
            retrieved_section = ""
    else:
        rag_execution_time = time.time() - rag_start_time
        logging.info("Vector DB 모듈이 없어 RAG 검색을 건너뜁니다.")
        
        if rag_span:
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


async def generate_primary_questions(prompt: str, trace=None) -> List[str]:
    """주 질문 생성 로직"""
    llm_span = None
    if trace:
        llm_span = trace.span(name="llm_call")
    
    llm_start_time = time.time()

    try:
        raw_response = await call_llm(prompt, trace_id=trace.id if trace else None)
        llm_execution_time = time.time() - llm_start_time
        
        if llm_span:
            llm_span.update(
                input={"prompt": prompt},
                output={"raw_response": raw_response},
                metadata={"execution_time_seconds": llm_execution_time},
            )
            llm_span.end()

        parsing_span = None
        if trace:
            parsing_span = trace.span(name="response_parsing")
        
        parsing_start_time = time.time()
        questions = parse_questions(raw_response)[:GENERATE_COUNT]
        parsing_execution_time = time.time() - parsing_start_time
        
        if parsing_span:
            parsing_span.update(
                input={"raw_response": raw_response},
                output={"parsed_questions": questions},
                metadata={"execution_time_seconds": parsing_execution_time},
            )
            parsing_span.end()

        return questions
    except Exception as e:
        llm_execution_time = time.time() - llm_start_time
        
        if llm_span:
            llm_span.end(
                error=str(e), 
                metadata={"execution_time_seconds": llm_execution_time}
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

    llm_span_api = None
    if trace:
        llm_span_api = trace.span(name="openai_api_call")
    
    llm_api_start_time = time.time()

    try:
        raw_response_api = await call_openai_api(prompt_api, trace_id=trace.id if trace else None)
        llm_api_execution_time = time.time() - llm_api_start_time
        
        if llm_span_api:
            llm_span_api.update(
                input={"prompt_api": prompt_api},
                output={"raw_response_api": raw_response_api},
                metadata={"execution_time_seconds": llm_api_execution_time},
            )
            llm_span_api.end()

        parsing_span_api = None
        if trace:
            parsing_span_api = trace.span(name="response_parsing_additional")
        
        parsing_api_start_time = time.time()
        additional_questions = parse_questions(raw_response_api)
        parsing_api_execution_time = time.time() - parsing_api_start_time
        
        if parsing_span_api:
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
        
        if llm_span_api:
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
            detail="질문 생성 모델(vLLM API)이 현재 사용할 수 없습니다. vLLM 서버가 실행 중인지 확인하세요.",
        )

    logger.info(f"vLLM API를 통한 질문 생성 요청: interview_id={req.interview_id}, req_data={req.dict()}")

    validate_request(req)

    trace_id = f"followup_{req.interview_id}_{uuid.uuid4().hex}"
    request_start_time = time.time()

    trace = None
    if langfuse:
        trace = langfuse.trace(
            id=trace_id,
            name="followup_generation_vllm_api",
            input={
                "interview_id": req.interview_id,
                "selected_question": req.selected_question,
                "keyword": req.keyword,
                "passed_questions": req.passed_questions or [],
                "api_mode": "vllm",
            },
        )

    try:
        context = prepare_context(req, trace)

        prompt_template = get_cached_prompt("followup_questions_generator")
        if prompt_template:
            prompt = prompt_template.compile(**context)
        else:
            # 기본 프롬프트 사용
            prompt = create_default_followup_prompt(context)

        generated_questions = await generate_primary_questions(prompt, trace)
        
        if len(generated_questions) < GENERATE_COUNT:
            remaining_count = GENERATE_COUNT - len(generated_questions)
            logger.info(f"vLLM API로 생성된 질문이 부족하여 OpenAI API로 추가 생성: {len(generated_questions)}/{GENERATE_COUNT}")
            generated_questions = await generate_additional_questions(
                req,
                context["passed_questions"],
                remaining_count,
                trace,
                generated_questions,
            )

        request_execution_time = time.time() - request_start_time
        
        if trace:
            trace.update(
                output={"followup_questions": generated_questions},
                metadata={
                    "total_execution_time_seconds": request_execution_time,
                    "primary_model": "vllm_api",
                    "fallback_used": len(generated_questions) < GENERATE_COUNT,
                },
            )
        
        logger.info(
            f"vLLM API 질문 생성 완료: interview_id={req.interview_id}, questions_count={len(generated_questions)}, execution_time={request_execution_time:.2f}초"
        )

        return FollowupResponse(
            message="followup_questions_generated",
            interview_id=req.interview_id,
            followup_questions=generated_questions,
        )

    except Exception as e:
        request_execution_time = time.time() - request_start_time
        error_msg = f"vLLM API 질문 생성 실패: {str(e)}"
        logger.error(f"{error_msg} (interview_id={req.interview_id})")
        
        if trace:
            trace.update(
                error={"message": str(e), "type": "vllm_api_error"},
                metadata={
                    "total_execution_time_seconds": request_execution_time,
                    "failed_model": "vllm_api",
                },
            )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg,
        )


# 질문 생성기 전용 상세 헬스 체크 (디버깅용)
@router.get("/status")
async def question_generator_status():
    """질문 생성기의 상세 상태를 확인합니다 (디버깅용)"""
    from app.main import is_model_available
    from app.api.question_generator.question_generator_model import check_vllm_health
    
    model_available = is_model_available("question_generator")
    api_healthy = False
    api_error = None
    
    if model_available:
        try:
            api_healthy = await check_vllm_health()
        except Exception as e:
            api_error = str(e)
            logger.error(f"vLLM API 상태 확인 실패: {e}")
    
    return {
        "service": "question_generator",
        "model_available": model_available,
        "vllm_api_healthy": api_healthy,
        "vllm_api_error": api_error,
        "vector_db_available": VECTOR_DB_AVAILABLE,
        "langfuse_configured": langfuse is not None,
        "status": "healthy" if (model_available and api_healthy) else "unhealthy",
        "config": {
            "api_base_url": VLLM_API_CONFIG["base_url"],
            "model_name": VLLM_API_CONFIG["model_name"],
            "timeout": VLLM_API_CONFIG["timeout"],
            "generate_count": GENERATE_COUNT,
            "max_history_questions": MAX_HISTORY_QUESTIONS,
        }
    }